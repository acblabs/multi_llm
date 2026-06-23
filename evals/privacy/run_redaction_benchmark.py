import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from multi_model_agent.privacy import redact_sensitive_data


CASES_PATH = ROOT / "evals" / "privacy" / "labeled_phi_cases.jsonl"
REPORT_PATH = ROOT / "evals" / "privacy" / "redaction_benchmark_report.md"
GATED_RECALL_THRESHOLDS = {
    "email": 0.95,
    "phone": 0.90,
    "ssn": 0.90,
    "member_id": 0.80,
}
TRACKED_LIMITATION_KINDS = ("bare_name", "bare_date")


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        case = json.loads(line)
        if "case_id" not in case or "text" not in case or "labels" not in case:
            raise ValueError(f"Invalid benchmark case at line {line_number}")
        cases.append(case)
    return cases


def calculate_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    total_counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    passed_cases = 0

    for case in cases:
        case_counts = score_case(case)
        if _case_passed(case_counts):
            passed_cases += 1
        for kind, kind_counts in case_counts.items():
            target = counts.setdefault(
                kind,
                {"true_positive": 0, "false_positive": 0, "false_negative": 0},
            )
            for metric_name, value in kind_counts.items():
                target[metric_name] += value
                total_counts[metric_name] += value

    metrics_by_kind = {
        kind: _metrics_from_counts(kind_counts)
        for kind, kind_counts in sorted(counts.items())
    }
    failures_by_category = {
        kind: int(metrics["false_positive"] + metrics["false_negative"])
        for kind, metrics in metrics_by_kind.items()
        if int(metrics["false_positive"] + metrics["false_negative"]) > 0
    }
    overall = _metrics_from_counts(total_counts)
    gates = {
        kind: {
            "recall": metrics_by_kind.get(kind, {}).get("recall", 0.0),
            "threshold": threshold,
            "passed": metrics_by_kind.get(kind, {}).get("recall", 0.0) >= threshold,
        }
        for kind, threshold in GATED_RECALL_THRESHOLDS.items()
    }
    failed_gated_identifier_types = [
        kind for kind, gate in gates.items() if not gate["passed"]
    ]
    return {
        "total_cases": len(cases),
        "passed_cases": passed_cases,
        "failed_cases": len(cases) - passed_cases,
        "case_pass_rate": passed_cases / len(cases) if cases else 1.0,
        "failures_by_category": failures_by_category,
        "overall": overall,
        "metrics_by_identifier_type": metrics_by_kind,
        "false_positive_count": total_counts["false_positive"],
        "false_negative_count": total_counts["false_negative"],
        "gated_thresholds": gates,
        "gated_identifier_types_passed": len(gates) - len(failed_gated_identifier_types),
        "gated_identifier_types_failed": len(failed_gated_identifier_types),
        "failed_gated_identifier_types": failed_gated_identifier_types,
        "passed": not failed_gated_identifier_types,
    }


def score_case(case: dict[str, Any]) -> dict[str, dict[str, int]]:
    assessment = redact_sensitive_data(case["text"])
    labels = [
        {"kind": str(label["kind"]), "value": str(label["value"])}
        for label in case.get("labels", [])
    ]
    predictions = [
        {"kind": finding.kind, "value": finding.value}
        for finding in assessment.findings
    ]
    counts: dict[str, dict[str, int]] = {}
    matched_predictions: set[int] = set()

    for label in labels:
        kind = label["kind"]
        _ensure_kind(counts, kind)
        match_index = _matching_prediction_index(
            label=label,
            predictions=predictions,
            matched_predictions=matched_predictions,
        )
        if match_index is None:
            counts[kind]["false_negative"] += 1
        else:
            counts[kind]["true_positive"] += 1
            matched_predictions.add(match_index)

    for index, prediction in enumerate(predictions):
        if index in matched_predictions:
            continue
        kind = prediction["kind"]
        _ensure_kind(counts, kind)
        counts[kind]["false_positive"] += 1

    return counts


def write_report(summary: dict[str, Any], path: Path = REPORT_PATH) -> None:
    overall = summary["overall"]
    lines = [
        "# Redaction Benchmark Report",
        "",
        "## Summary",
        "",
        f"- total_cases: {summary['total_cases']}",
        f"- passed_cases: {summary['passed_cases']}",
        f"- failed_cases: {summary['failed_cases']}",
        f"- pass_rate: {summary['case_pass_rate']:.3f}",
        f"- gate_passed: {summary['passed']}",
        f"- gated_identifier_types_passed: {summary['gated_identifier_types_passed']}",
        f"- gated_identifier_types_failed: {summary['gated_identifier_types_failed']}",
        f"- failed_gated_identifier_types: {json.dumps(summary['failed_gated_identifier_types'])}",
        f"- failures_by_category: {json.dumps(summary['failures_by_category'], sort_keys=True)}",
        f"- redaction_precision: {overall['precision']:.3f}",
        f"- redaction_recall: {overall['recall']:.3f}",
        f"- redaction_f1: {overall['f1']:.3f}",
        f"- false_positive_count: {summary['false_positive_count']}",
        f"- false_negative_count: {summary['false_negative_count']}",
        "",
        "## Gated Thresholds",
        "",
        "| Identifier Type | Recall | Threshold | Passed |",
        "| --- | ---: | ---: | --- |",
    ]
    for kind, gate in sorted(summary["gated_thresholds"].items()):
        lines.append(
            f"| {kind} | {gate['recall']:.3f} | {gate['threshold']:.3f} | {gate['passed']} |"
        )

    lines.extend(
        [
            "",
            "## Metrics By Identifier Type",
            "",
            "| Identifier Type | Precision | Recall | F1 | TP | FP | FN |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for kind, metrics in summary["metrics_by_identifier_type"].items():
        lines.append(
            f"| {kind} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | "
            f"{metrics['f1']:.3f} | {metrics['true_positive']} | "
            f"{metrics['false_positive']} | {metrics['false_negative']} |"
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This benchmark does not claim production-grade PHI detection.",
            "- Case pass/fail counts are entity-level exactness checks; process exit is controlled by the gated identifier-type recall thresholds above.",
            "- Bare-name and bare-date recall are tracked but are not CI gates for this regex-only redactor.",
            "- Gated CI thresholds cover email, formatted phone, SSN, and member ID recall only.",
            "",
            "## Tracked Non-Gated Limitations",
            "",
        ]
    )
    metrics_by_kind = summary["metrics_by_identifier_type"]
    for kind in TRACKED_LIMITATION_KINDS:
        if kind in metrics_by_kind:
            lines.append(f"- {kind}_recall: {metrics_by_kind[kind]['recall']:.3f}")
    lines.extend(
        [
            f"- overall_recall: {overall['recall']:.3f}",
            "",
            "## Regression Notes",
            "",
            "- Fast CI can run this benchmark without credentials, network access, or external model calls.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _matching_prediction_index(
    *,
    label: dict[str, str],
    predictions: list[dict[str, str]],
    matched_predictions: set[int],
) -> int | None:
    candidates: list[tuple[int, int]] = []
    for index, prediction in enumerate(predictions):
        if index in matched_predictions:
            continue
        if prediction["kind"] != label["kind"]:
            continue
        if _values_match(label["value"], prediction["value"]):
            candidates.append(
                (abs(len(prediction["value"]) - len(label["value"])), index)
            )
    if not candidates:
        return None
    return sorted(candidates)[0][1]


def _values_match(left: str, right: str) -> bool:
    normalized_left = _normalize_value(left)
    normalized_right = _normalize_value(right)
    compact_left = _compact_identifier(left)
    compact_right = _compact_identifier(right)
    return (
        normalized_left == normalized_right
        or _contains_with_boundary(normalized_right, normalized_left)
        or bool(compact_left and compact_left in compact_right)
    )


def _normalize_value(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def _compact_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _contains_with_boundary(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return re.search(
        rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])",
        haystack,
        flags=re.IGNORECASE,
    ) is not None


def _case_passed(counts: dict[str, dict[str, int]]) -> bool:
    return all(
        kind_counts["false_positive"] == 0 and kind_counts["false_negative"] == 0
        for kind_counts in counts.values()
    )


def _ensure_kind(counts: dict[str, dict[str, int]], kind: str) -> None:
    counts.setdefault(
        kind,
        {"true_positive": 0, "false_positive": 0, "false_negative": 0},
    )


def _metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    true_positive = counts["true_positive"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 1.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 1.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def main() -> int:
    summary = calculate_metrics(load_cases())
    write_report(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
