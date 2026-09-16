"""Human-readable evaluation report formatting."""

from src.evaluation.metrics import EvaluationResult


def format_evaluation_report(
    result: EvaluationResult,
    title: str = "EVALUATION SUMMARY",
) -> str:
    """Format a deterministic console report for an evaluation result."""
    lines = [
        title,
        "",
        f"Threshold: {result.threshold:.2f}",
        "",
        "Aggregate Metrics",
        f"Micro Precision: {_format_optional(result.aggregate_metrics['micro_precision'])}",
        f"Micro Recall: {_format_optional(result.aggregate_metrics['micro_recall'])}",
        f"Micro F1: {_format_optional(result.aggregate_metrics['micro_f1'])}",
        f"Macro Precision: {_format_optional(result.aggregate_metrics['macro_precision'])}",
        f"Macro Recall: {_format_optional(result.aggregate_metrics['macro_recall'])}",
        f"Macro F1: {_format_optional(result.aggregate_metrics['macro_f1'])}",
        "",
        "Per-Label Metrics",
    ]

    for label in result.labels:
        metrics = result.per_label_metrics[label]
        lines.append(
            f"{label}: precision={_format_optional(metrics['precision'])}, "
            f"recall={_format_optional(metrics['recall'])}, "
            f"f1={_format_optional(metrics['f1'])}, "
            f"support={metrics['support']}"
        )

    lines.extend(
        [
            "",
            "ROC-AUC",
            f"Macro ROC-AUC: {_format_optional(result.roc_auc['macro'])}",
            f"Micro ROC-AUC: {_format_optional(result.roc_auc['micro'])}",
        ]
    )
    for label in result.labels:
        lines.append(
            f"{label}: {_format_optional(result.roc_auc['per_label'][label])}"
        )

    lines.extend(
        [
            "",
            "Average Precision",
            f"Macro Average Precision: {_format_optional(result.average_precision['macro'])}",
            f"Micro Average Precision: {_format_optional(result.average_precision['micro'])}",
        ]
    )
    for label in result.labels:
        lines.append(
            f"{label}: {_format_optional(result.average_precision['per_label'][label])}"
        )

    lines.append("")
    lines.append("Confusion Matrices")
    for label in result.labels:
        counts = result.confusion_matrices[label]
        lines.append(
            f"{label}: TN={counts['TN']}, FP={counts['FP']}, "
            f"FN={counts['FN']}, TP={counts['TP']}"
        )

    return "\n".join(lines)


def _format_optional(value) -> str:
    if value is None:
        return "undefined"
    return f"{value:.4f}"

