"""
analyse.py — read result records and produce comparisons.

Kept separate from running: analysis gets rerun and rewritten constantly,
and it should never require re-executing a GPU run to change how results
are read.

Every derived figure produced here is labelled 'computed' to distinguish
it from the 'measured' values in the raw records (design doc 5.5).
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from .quality import quality_gate


def load_results(results_dir: str = "results") -> List[Dict[str, Any]]:
    d = Path(results_dir)
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        if p.name == "ledger.jsonl":
            continue
        try:
            with p.open() as f:
                out.append(json.load(f))
        except Exception:
            continue
    return out


def summary_table(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per run, the columns you actually compare on."""
    rows = []
    for r in records:
        cfg = r.get("config", {})
        load = r.get("load", {})
        lat = r.get("latency", {})

        rows.append({
            "run_id": r.get("run_id"),
            "experiment": cfg.get("experiment_id"),
            "label": cfg.get("label"),
            "base": cfg.get("base"),
            "architecture": cfg.get("architecture"),
            "backend": cfg.get("backend"),
            "attn": load.get("attn_impl_actual", cfg.get("attn_impl")),
            "status": r.get("status"),
            "weights_gb": load.get("weights_gb"),
            "peak_gb_max": lat.get("peak_gb_max"),
            "tok_per_sec": lat.get("tokens_per_sec_median"),
            "headline": r.get("standard_headline"),
            "warnings": len(r.get("warnings", [])),
        })
    return rows


def context_curve(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract the memory-vs-context curve from one run.

    The interesting columns: kv_cache_gb should grow linearly with context,
    peak_above_weights_gb faster than linear if the attention score matrix
    is being materialised. Divergence between those two is the quadratic
    term showing up in real data.
    """
    rows = []
    for point in record.get("context_scaling", []):
        if point.get("status") != "ok":
            rows.append({
                "context": point.get("target_context_tokens"),
                "status": point.get("status"),
            })
            continue
        gen = point.get("generation", {})
        rows.append({
            "context": point.get("prompt_tokens"),
            "kv_cache_gb": point.get("kv_cache_gb"),
            "kv_bytes_per_token": point.get("kv_bytes_per_token"),
            "kv_layers": point.get("kv_layers"),
            # Constant in context length -- the hybrid architecture's
            # structural advantage, and the figure the design doc lists as
            # publicly unspecified.
            "gdn_state_gb": point.get("gdn_state_gb"),
            "gdn_layers": point.get("gdn_layers"),
            "prefill_peak_above_weights_gb": point.get("peak_above_weights_gb"),
            "prefill_tok_per_sec": point.get("prefill_tokens_per_sec"),
            "gen_peak_gb": gen.get("peak_gb"),
            "gen_tok_per_sec": gen.get("tokens_per_sec"),
            "status": "ok",
        })
    return rows


def compare(
    baseline_run_id: str,
    candidate_run_id: str,
    results_dir: str = "results",
    max_relative_drop: float = 0.02,
) -> Dict[str, Any]:
    """
    Head-to-head against a pre-registered quality threshold.

    Memory savings are reported only alongside the gate result. A method
    that saves memory and fails the gate is a failed method -- presenting
    the saving without the gate is how a bad result gets talked into
    looking acceptable.
    """
    records = {r["run_id"]: r for r in load_results(results_dir)}

    base = records.get(baseline_run_id)
    cand = records.get(candidate_run_id)
    if base is None or cand is None:
        return {"status": "error", "reason": "run_id not found"}

    gate = quality_gate(
        base.get("standard_headline"),
        cand.get("standard_headline"),
        max_relative_drop=max_relative_drop,
    )

    bw = base.get("load", {}).get("weights_gb")
    cw = cand.get("load", {}).get("weights_gb")
    bp = base.get("latency", {}).get("peak_gb_max")
    cp = cand.get("latency", {}).get("peak_gb_max")
    bt = base.get("latency", {}).get("tokens_per_sec_median")
    ct = cand.get("latency", {}).get("tokens_per_sec_median")

    def rel(a, b):
        if a in (None, 0) or b is None:
            return None
        return round((a - b) / a, 4)

    same_arch = (base["config"].get("architecture")
                 == cand["config"].get("architecture"))

    return {
        "provenance": "computed",
        "baseline": baseline_run_id,
        "candidate": candidate_run_id,
        "quality_gate": gate,
        "weights_gb": {"baseline": bw, "candidate": cw,
                       "reduction": rel(bw, cw)},
        "peak_gb": {"baseline": bp, "candidate": cp,
                    "reduction": rel(bp, cp)},
        "tokens_per_sec": {"baseline": bt, "candidate": ct,
                           "speedup": round(ct / bt, 3) if bt and ct else None},
        "verdict": (
            "accepted" if gate.get("status") == "pass" else
            "rejected: quality gate failed" if gate.get("status") == "fail" else
            "indeterminate"
        ),
        "caveats": [
            None if same_arch else
            "Runs use different architectures; this is a generalisation "
            "check, not a like-for-like comparison.",
            "Latency comparisons are hardware-specific. Decode is "
            "memory-bandwidth-bound, so speedups measured on one card do "
            "not transfer to a card with different bandwidth.",
        ],
    }


def print_summary(results_dir: str = "results"):
    rows = summary_table(load_results(results_dir))
    if not rows:
        print("No results found.")
        return

    cols = ["experiment", "backend", "attn", "weights_gb",
            "peak_gb_max", "tok_per_sec", "headline", "status"]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows))
              for c in cols}

    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
