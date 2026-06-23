import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from multi_model_agent.escalation import assess_human_escalation
from multi_model_agent.evidence_coverage import generate_evidence_coverage_report
from multi_model_agent.privacy import redact_sensitive_data
from multi_model_agent.risk import classify_request


CASES_PATH = ROOT / "evals" / "fairness" / "prior_auth_invariance_cases.jsonl"
REPORT_PATH = ROOT / "evals" / "fairness" / "invariance_report.md"


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        case = json.loads(line)
        if "case_id" not in case or "variants" not in case:
            raise ValueError(f"Invalid invariance case at line {line_number}")
        if len(case["variants"]) < 2:
            raise ValueError(f"Invariance case {case['case_id']} needs at least two variants")
        cases.append(case)
    return cases


def build_structured_signature(text: str, *, trace_id: str) -> dict[str, Any]:
    privacy = redact_sensitive_data(text)
    risk = classify_request(
        text,
        contains_sensitive_data=privacy.contains_sensitive_data,
        use_case="prior_authorization",
    )
    escalation = assess_human_escalation(risk=risk, privacy=privacy)
    report = generate_evidence_coverage_report(
        trace_id=trace_id,
        text=privacy.redacted_text,
        workflow_type="prior_authorization",
        human_review_required=escalation.required,
    )
    return {
        "human_review_required": report.human_review_required,
        "prohibited_decision_boundary": sorted(report.prohibited_decision_boundary),
        "items": [
            {
                "requirement_id": item.requirement_id,
                "status": item.status,
            }
            for item in sorted(report.items, key=lambda candidate: candidate.requirement_id)
        ],
    }


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    baseline_variant = case["variants"][0]
    baseline_signature = build_structured_signature(
        baseline_variant["text"],
        trace_id=f"{case['case_id']}-{baseline_variant['variant_id']}",
    )
    mismatches: dict[str, Any] = {}

    for variant in case["variants"][1:]:
        signature = build_structured_signature(
            variant["text"],
            trace_id=f"{case['case_id']}-{variant['variant_id']}",
        )
        if signature != baseline_signature:
            mismatches[variant["variant_id"]] = {
                "expected": baseline_signature,
                "observed": signature,
            }

    return {
        "case_id": case["case_id"],
        "category": case.get("category", "invariance"),
        "passed": not mismatches,
        "baseline_variant": baseline_variant["variant_id"],
        "variant_count": len(case["variants"]),
        "mismatches": mismatches,
    }


def evaluate_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [evaluate_case(case) for case in cases]


def write_report(results: list[dict[str, Any]], path: Path = REPORT_PATH) -> None:
    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    failed = total - passed
    pass_rate = passed / total if total else 1.0
    failures_by_category: dict[str, int] = {}
    for result in results:
        if not result["passed"]:
            category = result["category"]
            failures_by_category[category] = failures_by_category.get(category, 0) + 1

    lines = [
        "# Prior-Authorization Invariance Report",
        "",
        "## Summary",
        "",
        f"- total_cases: {total}",
        f"- passed_cases: {passed}",
        f"- failed_cases: {failed}",
        f"- pass_rate: {pass_rate:.3f}",
        f"- failures_by_category: {json.dumps(failures_by_category, sort_keys=True)}",
        "- redaction_precision: not_applicable",
        "- redaction_recall: not_applicable",
        "",
        "## Results",
        "",
        "| Case | Category | Variants | Passed | Notes |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for result in results:
        notes = "OK" if result["passed"] else json.dumps(result["mismatches"], sort_keys=True)
        lines.append(
            f"| {result['case_id']} | {result['category']} | "
            f"{result['variant_count']} | {result['passed']} | {notes} |"
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Synthetic demographic variants are not evidence of production fairness.",
            "- This structured invariance regression compares evidence-coverage statuses and human-review boundaries, not free-text rationales.",
            "- Future fairness work should use broader policy-specific cohorts and clinically reviewed invariance criteria.",
            "",
            "## Regression Notes",
            "",
            "- Fast CI can run this eval without credentials, network access, or external model calls.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    results = evaluate_cases(load_cases())
    write_report(results)
    print(json.dumps({"results": results}, indent=2, sort_keys=True))
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
