"""Evaluation module for trajectory error, conflict risk, and baseline execution."""

from __future__ import annotations


def __getattr__(name: str):
    if name in ("AuditReport", "run_label_audit"):
        from src.eval import label_audit

        return getattr(label_audit, name)
    if name in ("PositionErrorMetric", "evaluate_position_errors", "format_position_errors"):
        from src.eval import metrics

        return getattr(metrics, name)
    if name in (
        "ConflictEvaluationResult",
        "evaluate_conflict_prediction",
        "format_conflict_evaluation",
    ):
        from src.eval import conflict_eval

        return getattr(conflict_eval, name)
    if name in ("run_test_evaluation",):
        from src.eval import run_baseline

        return getattr(run_baseline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AuditReport",
    "run_label_audit",
    "PositionErrorMetric",
    "evaluate_position_errors",
    "format_position_errors",
    "ConflictEvaluationResult",
    "evaluate_conflict_prediction",
    "format_conflict_evaluation",
    "run_test_evaluation",
]
