import json
import re
import shutil
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .audit_hashing import canonical_json
from .audit_store import (
    JsonlAuditStore,
    safe_audit_identifier,
    stored_event_to_audit_dict,
    verify_stored_events,
)
from .review import replay_trace_state
from .schemas import AuditVerificationResult, StoredAuditEvent, TraceState


DEFAULT_EVIDENCE_PACKET_ROOT = Path("examples/evidence_packets")

PACKET_FILENAMES = (
    "audit_events.jsonl",
    "audit_chain_verification.json",
    "trace_state.json",
    "governance_explanations.json",
    "evidence_coverage_report.json",
    "redaction_summary.json",
    "model_provenance.json",
    "human_review.json",
    "reviewer_summary.md",
)

SAMPLE_PHI_REGRESSION_VALUES = (
    "Jane Doe",
    "01/02/1960",
    "jane.doe@example.com",
    "555-123-4567",
    "ABC123456",
    "reviewer-123",
)

_SAFE_PATH_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class EvidencePacketExport:
    trace_id: str
    packet_dir: Path
    files: dict[str, Path]
    audit_chain_verification: AuditVerificationResult
    trace_state: TraceState
    event_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "packet_dir": _portable_report_path(self.packet_dir),
            "files": {
                name: _portable_report_path(path) for name, path in self.files.items()
            },
            "audit_chain_verification": self.audit_chain_verification.model_dump(
                mode="json"
            ),
            "trace_state": self.trace_state.model_dump(mode="json"),
            "event_count": self.event_count,
        }


def export_evidence_packet(
    *,
    trace_id: str,
    audit_log: str | Path,
    output_root: str | Path = DEFAULT_EVIDENCE_PACKET_ROOT,
) -> EvidencePacketExport:
    if not trace_id.strip():
        raise ValueError("trace_id is required")

    store = JsonlAuditStore(audit_log)
    all_events = store.list_events()
    verification = verify_stored_events(all_events, path=str(audit_log))
    safe_trace_id = safe_audit_identifier(trace_id, label="trace")
    stored_events = [event for event in all_events if event.trace_id == safe_trace_id]
    if not stored_events:
        raise ValueError(f"No audit events found for trace_id {trace_id!r}")

    stored_trace_id = stored_events[0].trace_id
    if any(event.trace_id != stored_trace_id for event in stored_events):
        raise ValueError(f"Audit trace {trace_id!r} resolved to inconsistent trace IDs")

    packet_dir = Path(output_root) / _packet_dir_name(stored_trace_id)
    _validate_packet_dir(packet_dir=packet_dir, output_root=Path(output_root))

    audit_events = [stored_event_to_audit_dict(event) for event in stored_events]
    trace_state = replay_trace_state(
        trace_id=stored_trace_id,
        events=audit_events,
        verification=verification,
    )
    packet = _build_packet(
        trace_id=stored_trace_id,
        stored_events=stored_events,
        audit_events=audit_events,
        verification=verification,
        trace_state=trace_state,
    )

    _rebuild_packet_dir(packet_dir=packet_dir, output_root=Path(output_root))
    files = _write_packet(packet_dir=packet_dir, packet=packet)
    return EvidencePacketExport(
        trace_id=stored_trace_id,
        packet_dir=packet_dir,
        files=files,
        audit_chain_verification=verification,
        trace_state=trace_state,
        event_count=len(stored_events),
    )


def packet_contains_sample_phi_values(packet_dir: str | Path) -> bool:
    root = Path(packet_dir)
    if not root.exists():
        return False
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        text = read_text_for_scan(path)
        if text is None:
            continue
        if any(raw_value in text for raw_value in SAMPLE_PHI_REGRESSION_VALUES):
            return True
    return False


def _build_packet(
    *,
    trace_id: str,
    stored_events: list[StoredAuditEvent],
    audit_events: list[dict[str, Any]],
    verification: AuditVerificationResult,
    trace_state: TraceState,
) -> dict[str, Any]:
    explanations = _collect_governance_explanations(audit_events)
    coverage_report = _latest_detail_value(
        audit_events,
        "evidence_coverage_report",
        default={"available": False},
    )
    redaction_summary = _aggregate_redaction_summary(audit_events)
    model_provenance = _collect_model_provenance(audit_events)
    human_review = _human_review_summary(audit_events, trace_state)
    reviewer_summary = _reviewer_summary_markdown(
        trace_id=trace_id,
        trace_state=trace_state,
        verification=verification,
        redaction_summary=redaction_summary,
        model_provenance=model_provenance,
        coverage_report=coverage_report,
        explanations=explanations,
        human_review=human_review,
    )

    return {
        "audit_events_jsonl": [
            canonical_json(event.model_dump(mode="json")) for event in stored_events
        ],
        "audit_chain_verification": verification.model_dump(mode="json"),
        "trace_state": trace_state.model_dump(mode="json"),
        "governance_explanations": explanations,
        "evidence_coverage_report": coverage_report,
        "redaction_summary": redaction_summary,
        "model_provenance": model_provenance,
        "human_review": human_review,
        "reviewer_summary": reviewer_summary,
    }


def _write_packet(
    *,
    packet_dir: Path,
    packet: dict[str, Any],
) -> dict[str, Path]:
    files = {name: packet_dir / name for name in PACKET_FILENAMES}
    files["audit_events.jsonl"].write_text(
        "\n".join(packet["audit_events_jsonl"]) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_json(
        files["audit_chain_verification.json"],
        packet["audit_chain_verification"],
    )
    _write_json(files["trace_state.json"], packet["trace_state"])
    _write_json(files["governance_explanations.json"], packet["governance_explanations"])
    _write_json(files["evidence_coverage_report.json"], packet["evidence_coverage_report"])
    _write_json(files["redaction_summary.json"], packet["redaction_summary"])
    _write_json(files["model_provenance.json"], packet["model_provenance"])
    _write_json(files["human_review.json"], packet["human_review"])
    files["reviewer_summary.md"].write_text(
        packet["reviewer_summary"],
        encoding="utf-8",
        newline="\n",
    )
    return files


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _collect_governance_explanations(
    audit_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    explanations: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for event in audit_events:
        details = event.get("details") or {}
        candidates: list[Any] = []
        if "governance_explanation" in details:
            candidates.append(details["governance_explanation"])
        if "governance_explanations" in details:
            value = details["governance_explanations"]
            if isinstance(value, list):
                candidates.extend(value)

        for candidate in candidates:
            if not isinstance(candidate, dict) or not candidate:
                continue
            dedupe_key = _explanation_dedupe_key(candidate)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            explanations.append(candidate)
    return explanations


def _explanation_dedupe_key(explanation: dict[str, Any]) -> str:
    decision_id = explanation.get("decision_id")
    if decision_id:
        return f"decision_id:{decision_id}"
    return "content:" + canonical_json(
        {
            "decision_type": explanation.get("decision_type"),
            "result": explanation.get("result"),
            "reason_codes": explanation.get("reason_codes") or [],
            "policy_ids": explanation.get("policy_ids") or [],
        }
    )


def _latest_detail_value(
    audit_events: list[dict[str, Any]],
    key: str,
    *,
    default: Any,
) -> Any:
    for event in reversed(audit_events):
        details = event.get("details") or {}
        value = details.get(key)
        if value:
            return value
    return default


def _aggregate_redaction_summary(audit_events: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    total_findings = 0
    contains_sensitive_data = False

    for event in audit_events:
        details = event.get("details") or {}
        summary = details.get("redaction_summary")
        if not isinstance(summary, dict):
            continue
        summaries.append(summary)
        total_findings += int(summary.get("total_findings") or 0)
        contains_sensitive_data = contains_sensitive_data or bool(
            summary.get("contains_sensitive_data")
        )
        by_kind = summary.get("finding_counts_by_kind") or {}
        if isinstance(by_kind, dict):
            for kind, value in by_kind.items():
                counts[str(kind)] = counts.get(str(kind), 0) + int(value or 0)

    return {
        "total_findings": total_findings,
        "total_findings_across_events": total_findings,
        "finding_counts_by_kind": dict(sorted(counts.items())),
        "contains_sensitive_data": contains_sensitive_data,
        "source_event_count": len(summaries),
        "counting_strategy": "sum_across_redaction_summary_events",
    }


def _collect_model_provenance(audit_events: list[dict[str, Any]]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for event in audit_events:
        details = event.get("details") or {}
        entry: dict[str, Any] = {
            "event_type": event.get("event_type"),
            "provider": event.get("provider"),
            "action": event.get("action"),
        }
        if isinstance(details.get("model_provenance"), dict):
            entry["model_provenance"] = details["model_provenance"]
        if details.get("model"):
            entry["model"] = details["model"]
        if details.get("token_counts"):
            entry["token_counts"] = details["token_counts"]
        if details.get("tokens") is not None:
            entry["tokens"] = details["tokens"]

        if (
            entry.get("provider")
            or entry.get("model_provenance")
            or entry.get("model")
            or entry.get("token_counts")
            or entry.get("tokens") is not None
        ):
            entries.append({key: value for key, value in entry.items() if value is not None})

    providers = sorted(
        {
            str(entry["provider"])
            for entry in entries
            if entry.get("provider")
        }
    )
    return {
        "providers": providers,
        "events": entries,
    }


def _human_review_summary(
    audit_events: list[dict[str, Any]],
    trace_state: TraceState,
) -> dict[str, Any]:
    review_events = [
        {
            "event_type": event.get("event_type"),
            "action": event.get("action"),
            "details": event.get("details") or {},
        }
        for event in audit_events
        if str(event.get("event_type", "")).startswith("human_")
    ]
    return {
        "required": trace_state.human_review_required,
        "assigned": trace_state.human_review_assigned,
        "completed": trace_state.human_review_completed,
        "final_decision": trace_state.final_human_review_decision,
        "reviewer_role": trace_state.reviewer_role,
        "reviewer_id_hmac": trace_state.reviewer_id_hmac,
        "events": review_events,
    }


def _reviewer_summary_markdown(
    *,
    trace_id: str,
    trace_state: TraceState,
    verification: AuditVerificationResult,
    redaction_summary: dict[str, Any],
    model_provenance: dict[str, Any],
    coverage_report: Any,
    explanations: list[dict[str, Any]],
    human_review: dict[str, Any],
) -> str:
    workflow_type = "unknown"
    coverage_summary = "No evidence coverage report was included for this trace."
    coverage_counts: dict[str, int] = {}
    if isinstance(coverage_report, dict) and coverage_report.get("available") is not False:
        workflow_type = str(coverage_report.get("workflow_type") or workflow_type)
        coverage_summary = str(
            coverage_report.get("overall_summary")
            or "Evidence coverage report included."
        )
        for item in coverage_report.get("items") or []:
            if isinstance(item, dict):
                status = str(item.get("status") or "unknown")
                coverage_counts[status] = coverage_counts.get(status, 0) + 1

    if trace_state.latest_risk_tier:
        risk_tier = trace_state.latest_risk_tier
    else:
        risk_tier = "unknown"

    provider_names = ", ".join(model_provenance.get("providers") or []) or "none recorded"
    policy_ids = _collect_policy_ids(explanations, human_review)
    policy_text = ", ".join(policy_ids) if policy_ids else "none recorded"
    redaction_event_count = int(redaction_summary.get("source_event_count") or 0)
    coverage_count_text = (
        ", ".join(f"{status}: {count}" for status, count in sorted(coverage_counts.items()))
        if coverage_counts
        else "none"
    )

    lines = [
        "# Evidence Packet Reviewer Summary",
        "",
        f"- Trace ID: {trace_id}",
        f"- Workflow type: {workflow_type}",
        f"- Risk tier: {risk_tier}",
        f"- Human review status: {_human_review_status(trace_state)}",
        f"- Redaction summary: {redaction_summary['total_findings']} finding observations across {redaction_event_count} audit event(s); sensitive data detected: {redaction_summary['contains_sensitive_data']}",
        f"- Provider/model provenance: {provider_names}",
        f"- Policy decisions: {policy_text}",
        f"- Evidence coverage summary: {coverage_summary}",
        f"- Evidence coverage counts: {coverage_count_text}",
        f"- Audit chain verification result: {'valid' if verification.valid else 'invalid'}",
        "",
        "## Known Limitations",
        "",
        "- This packet is generated from sanitized audit events and safe views only.",
        "- The audit chain is integrity-verifiable, not tamper-proof.",
        "- Evidence coverage is documentation triage for human review, not a coverage, diagnosis, treatment, or medical-necessity decision.",
        "- The MVP redactor and evidence classifier are deterministic heuristics, not production-grade clinical controls.",
    ]
    return "\n".join(lines) + "\n"


def _collect_policy_ids(
    explanations: list[dict[str, Any]],
    human_review: dict[str, Any],
) -> list[str]:
    policy_ids: list[str] = []
    seen: set[str] = set()

    for explanation in explanations:
        for policy_id in explanation.get("policy_ids") or []:
            value = str(policy_id)
            if value and value not in seen:
                policy_ids.append(value)
                seen.add(value)

    for event in human_review.get("events") or []:
        details = event.get("details") if isinstance(event, dict) else None
        if not isinstance(details, dict):
            continue
        for policy_id in details.get("policy_ids") or []:
            value = str(policy_id)
            if value and value not in seen:
                policy_ids.append(value)
                seen.add(value)

    return policy_ids


def _human_review_status(trace_state: TraceState) -> str:
    if trace_state.human_review_completed:
        return f"completed ({trace_state.final_human_review_decision or 'decision recorded'})"
    if trace_state.human_review_assigned:
        return "assigned"
    if trace_state.human_review_required:
        return "required"
    return "not required"


def _validate_packet_dir(*, packet_dir: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_root = output_root.resolve()
    resolved_packet_dir = packet_dir.resolve()
    if resolved_packet_dir == resolved_root or resolved_root not in resolved_packet_dir.parents:
        raise ValueError(f"Refusing to rebuild packet directory outside output root: {packet_dir}")


def _rebuild_packet_dir(*, packet_dir: Path, output_root: Path) -> None:
    _validate_packet_dir(packet_dir=packet_dir, output_root=output_root)

    if packet_dir.exists():
        if not packet_dir.is_dir():
            raise ValueError(f"Packet output path exists and is not a directory: {packet_dir}")
        shutil.rmtree(packet_dir)

    packet_dir.mkdir(parents=True, exist_ok=False)


def _packet_dir_name(stored_trace_id: str) -> str:
    prefix = _safe_path_component(stored_trace_id)[:96]
    digest = sha256(stored_trace_id.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}-{digest}" if prefix else digest


def read_text_for_scan(path: str | Path) -> str | None:
    target = Path(path)
    try:
        return target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _safe_path_component(value: str) -> str:
    cleaned = _SAFE_PATH_COMPONENT_RE.sub("_", value).strip("._")
    return cleaned[:120] if cleaned else "trace"


def _portable_report_path(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/")
