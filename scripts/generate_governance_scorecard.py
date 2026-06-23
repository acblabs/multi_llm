import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_model_agent.audit_store import JsonlAuditStore  # noqa: E402
from multi_model_agent.evidence_packet import (  # noqa: E402
    DEFAULT_EVIDENCE_PACKET_ROOT,
    SAMPLE_PHI_REGRESSION_VALUES,
    read_text_for_scan,
)


DEFAULT_REDTEAM_REPORT = ROOT / "evals" / "redteam" / "prior_auth_redteam_report.md"
DEFAULT_PRIVACY_REPORT = ROOT / "evals" / "privacy" / "redaction_benchmark_report.md"
DEFAULT_FAIRNESS_REPORT = ROOT / "evals" / "fairness" / "invariance_report.md"
DEFAULT_SAMPLE_AUDIT_LOG = ROOT / "examples" / "audit" / "sample_audit_chain.jsonl"
DEFAULT_OUTPUT = ROOT / "governance" / "governance_scorecard.md"

_SUMMARY_LINE_RE = re.compile(r"^-\s+([A-Za-z0-9_]+):\s*(.+?)\s*$")


def generate_scorecard(
    *,
    redteam_report: str | Path = DEFAULT_REDTEAM_REPORT,
    privacy_report: str | Path = DEFAULT_PRIVACY_REPORT,
    fairness_report: str | Path = DEFAULT_FAIRNESS_REPORT,
    sample_audit_log: str | Path = DEFAULT_SAMPLE_AUDIT_LOG,
    evidence_packet_root: str | Path = DEFAULT_EVIDENCE_PACKET_ROOT,
    output: str | Path = DEFAULT_OUTPUT,
    unit_tests_passed: str | None = None,
    human_review_controls_passed: str | None = None,
    observability_tests_passed: str | None = None,
) -> dict[str, Any]:
    redteam = _parse_markdown_summary(Path(redteam_report))
    privacy = _parse_markdown_summary(Path(privacy_report))
    fairness = _parse_markdown_summary(Path(fairness_report))
    audit_verification = _verify_audit_chain(Path(sample_audit_log))
    sample_phi_regression_guard_passed = _sample_phi_regression_guard_passed(
        sample_audit_log=Path(sample_audit_log),
        evidence_packet_root=Path(evidence_packet_root),
    )
    report_parse_errors = _report_parse_errors(
        redteam=redteam,
        privacy=privacy,
        fairness=fairness,
    )

    scorecard = {
        "unit_tests_passed": _observed_status(
            explicit=unit_tests_passed,
            env_name="GOVERNANCE_UNIT_TESTS_PASSED",
        ),
        "redteam_pass_rate": _float_or_none(redteam.get("pass_rate")),
        "privacy_redaction_recall": _float_or_none(privacy.get("redaction_recall")),
        "privacy_redaction_precision": _float_or_none(
            privacy.get("redaction_precision")
        ),
        "privacy_gate_passed": _bool_or_none(privacy.get("gate_passed")),
        "fairness_invariance_pass_rate": _float_or_none(fairness.get("pass_rate")),
        "audit_chain_verified": audit_verification["valid"],
        "sample_phi_regression_guard_passed": sample_phi_regression_guard_passed,
        "human_review_controls_passed": _observed_status(
            explicit=human_review_controls_passed,
            env_name="GOVERNANCE_HUMAN_REVIEW_CONTROLS_PASSED",
        ),
        "observability_tests_passed": _observed_status(
            explicit=observability_tests_passed,
            env_name="GOVERNANCE_OBSERVABILITY_TESTS_PASSED",
        ),
    }
    sources = {
        "redteam_report": _relative(redteam_report),
        "privacy_report": _relative(privacy_report),
        "fairness_report": _relative(fairness_report),
        "sample_audit_log": _relative(sample_audit_log),
        "evidence_packet_root": _relative(evidence_packet_root),
    }
    result = {
        "scorecard": scorecard,
        "sources": sources,
        "audit_verification": audit_verification,
        "report_parse_errors": report_parse_errors,
        "critical_checks_passed": _critical_checks_passed(scorecard)
        and not report_parse_errors,
    }
    _write_scorecard(Path(output), result)
    return result


def _parse_markdown_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {"missing_report": str(path)}

    summary: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _SUMMARY_LINE_RE.match(line)
        if match:
            summary[match.group(1)] = match.group(2)
    return summary


def _verify_audit_chain(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "valid": False,
            "event_count": 0,
            "errors": [f"missing sample audit log: {path}"],
            "final_hash": None,
            "path": str(path),
        }

    result = JsonlAuditStore(path).verify_chain()
    return result.model_dump(mode="json")


def _sample_phi_regression_guard_passed(
    *,
    sample_audit_log: Path,
    evidence_packet_root: Path,
) -> bool:
    paths: list[Path] = []
    if sample_audit_log.exists():
        paths.append(sample_audit_log)
    if evidence_packet_root.exists():
        paths.extend(path for path in evidence_packet_root.rglob("*") if path.is_file())

    for path in paths:
        text = read_text_for_scan(path)
        if text is None:
            continue
        if any(raw_value in text for raw_value in SAMPLE_PHI_REGRESSION_VALUES):
            return False
    return True

def _observed_status(*, explicit: str | None, env_name: str) -> str:
    value = explicit if explicit is not None else os.getenv(env_name)
    if value is None:
        return "not_observed_by_generator"
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "passed", "pass"}:
        return "true"
    if lowered in {"0", "false", "no", "failed", "fail"}:
        return "false"
    return str(value)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _critical_checks_passed(scorecard: dict[str, Any]) -> bool:
    return (
        _observed_control_passed(scorecard["unit_tests_passed"])
        and _observed_control_passed(scorecard["human_review_controls_passed"])
        and _observed_control_passed(scorecard["observability_tests_passed"])
        and scorecard["redteam_pass_rate"] == 1.0
        and scorecard["privacy_gate_passed"] is True
        and scorecard["fairness_invariance_pass_rate"] == 1.0
        and scorecard["audit_chain_verified"] is True
        and scorecard["sample_phi_regression_guard_passed"] is True
    )


def _observed_control_passed(value: Any) -> bool:
    if value in {None, "not_observed_by_generator"}:
        return True
    return str(value).strip().lower() == "true"


def _write_scorecard(path: Path, result: dict[str, Any]) -> None:
    scorecard = result["scorecard"]
    sources = result["sources"]
    audit = result["audit_verification"]
    parse_errors = result.get("report_parse_errors", [])
    lines = [
        "# Governance Scorecard",
        "",
        "## Summary",
        "",
    ]
    for key in (
        "unit_tests_passed",
        "redteam_pass_rate",
        "privacy_redaction_recall",
        "privacy_redaction_precision",
        "fairness_invariance_pass_rate",
        "audit_chain_verified",
        "sample_phi_regression_guard_passed",
        "human_review_controls_passed",
        "observability_tests_passed",
    ):
        lines.append(f"- {key}: {_format_score_value(scorecard.get(key))}")

    lines.extend(
        [
            f"- privacy_gate_passed: {_format_score_value(scorecard.get('privacy_gate_passed'))}",
            f"- critical_checks_passed: {result['critical_checks_passed']}",
            "",
            "## Audit Chain",
            "",
            f"- sample_audit_event_count: {audit.get('event_count')}",
            f"- sample_audit_final_hash: {audit.get('final_hash')}",
            f"- sample_audit_errors: {json.dumps(audit.get('errors', []), sort_keys=True)}",
            f"- report_parse_errors: {json.dumps(parse_errors, sort_keys=True)}",
            "",
            "## Sources",
            "",
        ]
    )
    for key, value in sources.items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This scorecard summarizes deterministic local reports and a sample audit-chain verification.",
            "- sample_phi_regression_guard_passed is a regression check over known synthetic sample values, not a general PHI detector.",
            "- GitHub Actions supplies unit-test, human-review-control, and observability-test pass/fail status from discrete completed steps.",
            "- Cloud Build remains the GCP deployment and deployment-evidence path; this scorecard is repository governance evidence.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_score_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _report_parse_errors(
    *,
    redteam: dict[str, str],
    privacy: dict[str, str],
    fairness: dict[str, str],
) -> list[str]:
    requirements = {
        "redteam": (redteam, ("pass_rate",)),
        "privacy": (privacy, ("redaction_recall", "redaction_precision", "gate_passed")),
        "fairness": (fairness, ("pass_rate",)),
    }
    errors: list[str] = []
    for report_name, (summary, keys) in requirements.items():
        for key in keys:
            if key not in summary:
                errors.append(f"{report_name}: missing summary field {key}")
    return errors


def _relative(path: str | Path) -> str:
    target = Path(path)
    try:
        return target.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(target).replace("\\", "/")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic governance scorecard from local reports."
    )
    parser.add_argument("--redteam-report", default=str(DEFAULT_REDTEAM_REPORT))
    parser.add_argument("--privacy-report", default=str(DEFAULT_PRIVACY_REPORT))
    parser.add_argument("--fairness-report", default=str(DEFAULT_FAIRNESS_REPORT))
    parser.add_argument("--sample-audit-log", default=str(DEFAULT_SAMPLE_AUDIT_LOG))
    parser.add_argument(
        "--evidence-packet-root",
        default=str(DEFAULT_EVIDENCE_PACKET_ROOT),
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--unit-tests-passed")
    parser.add_argument("--human-review-controls-passed")
    parser.add_argument("--observability-tests-passed")
    args = parser.parse_args()

    result = generate_scorecard(
        redteam_report=args.redteam_report,
        privacy_report=args.privacy_report,
        fairness_report=args.fairness_report,
        sample_audit_log=args.sample_audit_log,
        evidence_packet_root=args.evidence_packet_root,
        output=args.output,
        unit_tests_passed=args.unit_tests_passed,
        human_review_controls_passed=args.human_review_controls_passed,
        observability_tests_passed=args.observability_tests_passed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["critical_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
