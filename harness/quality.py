"""
quality.py — the quality gate, via EleutherAI lm-evaluation-harness.

Deliberately does NOT implement its own scorer. lm-eval is the standard
the quantization literature reports against (ARC, HellaSwag, PIQA,
WinoGrande, GSM8K, WikiText perplexity), and using it means results are
comparable to published numbers rather than to a bespoke rubric.

Three layers, reported separately and never averaged into one figure:
  1. standard battery  -- comparability with the literature
  2. long-context      -- relevance to the actual workload (15k+ tokens)
  3. domain            -- the applied anchor (GPRA clinical reasoning)

Averaging them would let a long-context collapse hide behind strong
short-prompt scores, which is precisely the failure mode that matters here.
"""

import json
from typing import Dict, Any, List, Optional


def _lm_eval_available() -> bool:
    try:
        import lm_eval  # noqa: F401
        return True
    except Exception:
        return False


def run_lm_eval(
    model,
    tokenizer,
    tasks: List[str],
    num_fewshot: int = 0,
    limit: Optional[int] = None,
    batch_size: int = 1,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate an ALREADY-LOADED model object.

    Passing the live model matters: reloading inside lm-eval would use its
    own loading path and could silently apply a different quantization or
    attention config than the one under test, so the thing evaluated would
    not be the thing profiled.

    max_length caps the sequence length used for rolling-loglikelihood tasks.
    This is not cosmetic. Perplexity scores EVERY token, so it materialises
    logits for the whole window at once: seq_len x vocab x 4 bytes. On a
    248,320-token vocabulary a 4096-token window is 4.07 GB, which OOMs
    alongside a 17 GB fp16 model on a 22 GB card. Multiple-choice tasks never
    hit this because they score short sequences.

    Perplexity values depend on the window, so runs compared against each
    other must use the SAME max_length. Relative comparison stays valid;
    absolute values are not comparable to published figures at other lengths.
    """
    if not _lm_eval_available():
        return {
            "status": "unavailable",
            "hint": "pip install lm-eval",
        }

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    hflm_kwargs = {"pretrained": model, "tokenizer": tokenizer,
                   "batch_size": batch_size}
    if max_length is not None:
        hflm_kwargs["max_length"] = max_length

    lm = HFLM(**hflm_kwargs)

    try:
        out = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            bootstrap_iters=0,   # we compare conditions, not report CIs per run
        )
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}

    scores: Dict[str, Any] = {}
    for task, metrics in out.get("results", {}).items():
        clean = {}
        for k, v in metrics.items():
            if k == "alias":
                continue
            if isinstance(v, (int, float)):
                clean[k] = round(float(v), 5)
        scores[task] = clean

    return {
        "status": "ok",
        "scores": scores,
        "n_samples": out.get("n-samples"),
        "tasks_requested": tasks,
        "num_fewshot": num_fewshot,
        "limit": limit,
    }


def headline_score(eval_result: Dict[str, Any]) -> Optional[float]:
    """
    Single comparable number for the standard battery: unweighted mean of
    per-task accuracy.

    Use for tracking and gating only. Report per-task scores in any writeup:
    a mean can stay flat while one task collapses, which is exactly the
    signal a quality gate exists to catch.
    """
    if eval_result.get("status") != "ok":
        return None

    accs = []
    for task, metrics in eval_result.get("scores", {}).items():
        for key in ("acc_norm,none", "acc,none", "exact_match,strict-match",
                    "exact_match,flexible-extract"):
            if key in metrics:
                accs.append(metrics[key])
                break

    if not accs:
        return None
    return round(sum(accs) / len(accs), 5)


def quality_gate(
    baseline_score: Optional[float],
    candidate_score: Optional[float],
    max_relative_drop: float = 0.02,
) -> Dict[str, Any]:
    """
    Pre-registered pass/fail against a baseline.

    Per the design doc: a method that reduces memory but drops quality below
    the threshold is a FAILED method, however large the memory saving. The
    threshold is set before running, which is what stops a marginal result
    being talked into acceptability afterwards.

    Default 2% relative drop mirrors the tolerance commonly claimed in
    quantization papers. Set it explicitly per experiment.
    """
    if baseline_score is None or candidate_score is None:
        return {"status": "indeterminate",
                "reason": "missing baseline or candidate score"}

    rel_drop = (baseline_score - candidate_score) / baseline_score

    return {
        "status": "pass" if rel_drop <= max_relative_drop else "fail",
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "relative_drop": round(rel_drop, 5),
        "threshold": max_relative_drop,
    }


# ---------------------------------------------------------------------------
# Domain evaluation stub.
#
# Left deliberately unimplemented rather than filled with placeholder items.
# A domain gate built from invented cases would produce a number that looks
# rigorous and means nothing. Populate from real GPRA cases with known
# correct reasoning, then wire in.
# ---------------------------------------------------------------------------

def run_domain_eval(model, tokenizer, cases_path: str) -> Dict[str, Any]:
    """
    Score domain reasoning against a curated case set.

    Expected JSONL format, one object per line:
      {"id": "...", "prompt": "...", "must_contain": [...],
       "must_not_contain": [...]}

    must_contain / must_not_contain is a crude rubric, but it is
    deterministic and auditable, which matters more here than nuance:
    the purpose is detecting degradation between runs, not grading
    clinical quality in absolute terms.
    """
    try:
        with open(cases_path) as f:
            cases = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return {"status": "unavailable",
                "hint": f"no case file at {cases_path}"}

    if not cases:
        return {"status": "unavailable", "hint": "case file is empty"}

    from .profiler import measure_generation

    passed, details = 0, []
    for case in cases:
        gen = measure_generation(
            model, tokenizer, case["prompt"], max_new_tokens=512
        )
        text = gen["sample_output"].lower()

        ok = all(s.lower() in text for s in case.get("must_contain", []))
        ok = ok and not any(
            s.lower() in text for s in case.get("must_not_contain", [])
        )

        passed += int(ok)
        details.append({"id": case.get("id"), "passed": ok})

    return {
        "status": "ok",
        "domain_score": round(passed / len(cases), 4),
        "n_cases": len(cases),
        "details": details,
    }
