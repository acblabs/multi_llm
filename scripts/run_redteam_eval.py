import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multi_model_agent.escalation import assess_human_escalation
from multi_model_agent.governance import GovernanceBlockedError, prepare_provider_request
from multi_model_agent.policy import evaluate_provider_access
from multi_model_agent.privacy import redact_sensitive_data
from multi_model_agent.risk import classify_request


CASES_PATH = ROOT / "evals" / "redteam" / "prior_auth_redteam_cases.json"
REPORT_PATH = ROOT / "evals" / "redteam" / "prior_auth_redteam_report.md"


def evaluate_case(case: dict) -> dict:
    prompt = case["prompt"]
    expected = case["expected"]

    privacy = redact_sensitive_data(prompt)
    risk = classify_request(
        prompt,
        contains_sensitive_data=privacy.contains_sensitive_data,
    )
    escalation = assess_human_escalation(risk=risk, privacy=privacy)
    unredacted_decision = evaluate_provider_access(
        provider="openai",
        prompt=prompt,
        risk=risk,
        privacy=privacy,
    )

    provider_blocked = False
    try:
        prepare_provider_request(provider="openai", prompt=prompt)
    except GovernanceBlockedError:
        provider_blocked = True

    observed = {
        "risk_tier": risk.risk_tier.value,
        "redaction_required": privacy.contains_sensitive_data,
        "unredacted_egress_blocked": unredacted_decision.action.value == "block",
        "human_review_required": escalation.required,
        "provider_call_blocked": provider_blocked,
    }

    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }

    return {
        "case_id": case["id"],
        "category": case["category"],
        "passed": not mismatches,
        "observed": observed,
        "mismatches": mismatches,
    }


def write_report(results: list[dict]) -> None:
    passed = sum(1 for result in results if result["passed"])
    total = len(results)
    lines = [
        "# Prior-Authorization Red-Team Report",
        "",
        f"Result: {passed}/{total} cases passed.",
        "",
        "| Case | Category | Passed | Notes |",
        "| --- | --- | --- | --- |",
    ]

    for result in results:
        notes = "OK" if result["passed"] else json.dumps(result["mismatches"], sort_keys=True)
        lines.append(
            f"| {result['case_id']} | {result['category']} | {result['passed']} | {notes} |"
        )

    lines.extend(
        [
            "",
            "This eval verifies the MVP governance path only. It does not call external LLM providers.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    results = [evaluate_case(case) for case in cases]
    write_report(results)

    if not all(result["passed"] for result in results):
        print(json.dumps(results, indent=2, sort_keys=True))
        return 1

    print(f"prior-auth red-team eval passed: {len(results)}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
