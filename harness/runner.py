"""
runner.py — orchestrates one experimental condition end to end.

Order matters: load, profile, then evaluate. Profiling first means memory
numbers are not polluted by lm-eval's own allocations, which can be
substantial on long-context tasks.

Every result carries a provenance label -- measured, computed, or
estimated -- per design doc 5.5. Nothing in this file produces 'computed'
or 'estimated' values; everything here is measured. Derived figures belong
in analysis, tagged accordingly.
"""

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from .config import RunConfig, capture_environment
from .loader import load, free_gpu
from .profiler import (
    profile_context_scaling, make_synthetic_prompt, latency_repeats,
)
from .quality import run_lm_eval, headline_score, run_domain_eval


RESULTS_DIR = Path("results")


def run_experiment(
    cfg: RunConfig,
    results_dir: Path = RESULTS_DIR,
    domain_cases_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute one condition, write a result record, return it."""

    results_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()
    record: Dict[str, Any] = {
        "run_id": f"{cfg.experiment_id}-{cfg.fingerprint()}",
        "started_utc": started,
        "config": cfg.to_dict(),
        "environment": capture_environment(),
        "provenance": "measured",
    }

    t_start = time.perf_counter()

    try:
        # ---- load -------------------------------------------------------
        model, tok, load_info = load(cfg)
        record["load"] = load_info

        if not load_info["fast_path"].get("fla", False):
            record.setdefault("warnings", []).append(
                "flash-linear-attention missing: GDN layers on slow fallback. "
                "Timing not representative of a configured deployment."
            )
        if load_info.get("attn_impl_actual") != load_info.get("attn_impl_requested"):
            record.setdefault("warnings", []).append(
                f"attention impl requested "
                f"{load_info.get('attn_impl_requested')} but model reports "
                f"{load_info.get('attn_impl_actual')}"
            )

        # ---- profile (before eval, so memory is uncontaminated) ---------
        if cfg.run_perf:
            record["context_scaling"] = profile_context_scaling(
                model, tok,
                context_lengths=cfg.profile_context_lengths,
                max_new_tokens=64,
            )
            fixed_prompt = make_synthetic_prompt(tok, 1024)
            record["latency"] = latency_repeats(
                model, tok, fixed_prompt,
                max_new_tokens=cfg.max_new_tokens,
                repeats=3,
            )

        # ---- quality ----------------------------------------------------
        if cfg.run_standard_eval:
            std = run_lm_eval(
                model, tok,
                tasks=cfg.standard_tasks,
                num_fewshot=cfg.num_fewshot,
                limit=cfg.eval_limit,
                batch_size=cfg.batch_size,
            )
            record["standard_eval"] = std
            record["standard_headline"] = headline_score(std)

        if cfg.run_longcontext_eval:
            # Reported separately, never merged into the standard headline:
            # a long-context failure must not be maskable by short-prompt wins.
            record["longcontext_eval"] = run_lm_eval(
                model, tok,
                tasks=cfg.longcontext_tasks,
                num_fewshot=0,
                limit=cfg.eval_limit,
                batch_size=1,
            )

        if cfg.run_domain_eval and domain_cases_path:
            record["domain_eval"] = run_domain_eval(
                model, tok, domain_cases_path
            )

        record["status"] = "ok"

    except Exception as e:
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()

    finally:
        try:
            del model
        except Exception:
            pass
        free_gpu()

    record["wall_seconds"] = round(time.perf_counter() - t_start, 1)
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()

    out_path = results_dir / f"{record['run_id']}.json"
    with out_path.open("w") as f:
        json.dump(record, f, indent=2)

    ledger = results_dir / "ledger.jsonl"
    with ledger.open("a") as f:
        f.write(json.dumps({
            "run_id": record["run_id"],
            "experiment_id": cfg.experiment_id,
            "label": cfg.label,
            "base": cfg.base,
            "backend": cfg.backend,
            "attn_impl": cfg.attn_impl,
            "status": record["status"],
            "weights_gb": record.get("load", {}).get("weights_gb"),
            "standard_headline": record.get("standard_headline"),
            "finished_utc": record["finished_utc"],
        }) + "\n")

    return record


def run_matrix(configs, results_dir: Path = RESULTS_DIR, **kwargs):
    """
    Run several conditions sequentially.

    Sequential by design: concurrent runs share a GPU and would corrupt
    every memory measurement in the batch.
    """
    out = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {cfg.experiment_id} — {cfg.label}")
        rec = run_experiment(cfg, results_dir=results_dir, **kwargs)
        status = rec["status"]
        weights = rec.get("load", {}).get("weights_gb")
        head = rec.get("standard_headline")
        print(f"    status={status} weights={weights}GB headline={head}")
        out.append(rec)
    return out
