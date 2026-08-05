"""
LLM inference efficiency harness.

Three jobs, deliberately separated:
  - profiler.py : memory + latency (measured, closes the peak-memory gap)
  - quality.py  : quality via lm-eval (standard, comparable to literature)
  - runner.py   : orchestration, provenance, result records

Quick start:
    from harness.experiments import smoke
    from harness.runner import run_matrix
    run_matrix(smoke())
    from harness.analyse import print_summary; print_summary()
"""

from .config import RunConfig, BASES, BACKENDS, ATTN_IMPLS, capture_environment
from .runner import run_experiment, run_matrix
from .analyse import print_summary, compare, context_curve, summary_table

__all__ = [
    "RunConfig", "BASES", "BACKENDS", "ATTN_IMPLS", "capture_environment",
    "run_experiment", "run_matrix",
    "print_summary", "compare", "context_curve", "summary_table",
]
