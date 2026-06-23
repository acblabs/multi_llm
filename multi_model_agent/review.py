import hmac
import os
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from .audit import append_audit_event, get_audit_log, verify_audit_chain
from .audit_store import safe_audit_identifier
from .schemas import (
    AuditVerificationResult,
    HumanReviewDecision,
    HumanReviewDecisionValue,
    RiskTier,
    TraceState,
    sanitize_review_rationale_text,
)


REVIEW_HMAC_KEY_ENV = "MULTI_LLM_REVIEW_HMAC_KEY"
REVIEW_HMAC_SALT_ENV = "MULTI_LLM_REVIEW_HMAC_SALT"
DEFAULT_REVIEW_HMAC_SALT = "multi-llm-human-reviewer-v1"
OVERRIDE_DECISIONS: set[HumanReviewDecisionValue] = {
    "modified",
    "rejected",
    "escalated",
}


class ReviewConfigurationError(RuntimeError):
    pass


def record_human_review_assignment(
    *,
    trace_id: str,
    target_queue: str | None,
    risk_tier: RiskTier | None,
    reason_codes: list[str] | None = None,
    policy_ids: list[str] | None = None,
    assigned_at: datetime | None = None,
) -> None:
    assigned_at = assigned_at or datetime.now(timezone.utc)
    append_audit_event(
        trace_id=trace_id,
        event_type="human_review_assigned",
        action="assigned",
        risk_tier=risk_tier,
        details={
            "review_status": "assigned",
            "human_review_required": True,
            "human_review_assigned": True,
            "target_queue": target_queue,
            "assigned_at": _format_datetime_utc(assigned_at),
            "reason_codes": reason_codes or ["HUMAN_REVIEW_REQUIRED"],
            "policy_ids": policy_ids or ["human_review_assignment_recorded"],
        },
    )


def record_human_review_decision(
    *,
    trace_id: str,
    reviewer_id: str,
    decision: HumanReviewDecisionValue,
    rationale: str,
    reviewer_role: str = "clinical_operations",
    reviewed_at: datetime | None = None,
) -> HumanReviewDecision:
    reviewed_at = reviewed_at or datetime.now(timezone.utc)
    review_decision = HumanReviewDecision(
        trace_id=trace_id,
        reviewer_role=reviewer_role,
        reviewer_id_hmac=hmac_reviewer_id(reviewer_id),
        decision=decision,
        rationale=sanitize_review_rationale(rationale),
        reviewed_at=reviewed_at,
    )
    safe_decision = review_decision.to_safe_dict()

    append_audit_event(
        trace_id=trace_id,
        event_type="human_review_completed",
        action=decision,
        details={
            "review_status": "completed",
            "human_review_completed": True,
            "reviewer_role": safe_decision["reviewer_role"],
            "reviewer_id_hmac": safe_decision["reviewer_id_hmac"],
            "review_decision": safe_decision["decision"],
            "review_rationale": safe_decision["rationale"],
            "reviewed_at": safe_decision["reviewed_at"],
            "human_review_decision": safe_decision,
            "reason_codes": ["HUMAN_REVIEW_COMPLETED"],
            "policy_ids": ["human_review_closure_required"],
        },
    )

    if decision in OVERRIDE_DECISIONS:
        append_audit_event(
            trace_id=trace_id,
            event_type="human_override_recorded",
            action=decision,
            details={
                "review_status": "override_recorded",
                "reviewer_role": safe_decision["reviewer_role"],
                "reviewer_id_hmac": safe_decision["reviewer_id_hmac"],
                "review_decision": safe_decision["decision"],
                "override_rationale": safe_decision["rationale"],
                "reviewed_at": safe_decision["reviewed_at"],
                "reason_codes": ["HUMAN_OVERRIDE_RECORDED"],
                "policy_ids": ["human_override_rationale_required"],
            },
        )

    return review_decision


def resolve_trace_state(trace_id: str) -> TraceState:
    """Replay sanitized trace events and include whole-store chain verification.

    The JSONL hash chain links every event in the audit store, so
    audit_chain_valid/audit_chain_errors describe the store-wide chain that
    contains this trace, not an independently verifiable per-trace subchain.
    """
    events = get_audit_log(trace_id)
    verification = verify_audit_chain()
    resolved_trace_id = (
        str(events[0].get("trace_id")) if events else safe_audit_identifier(trace_id, label="trace")
    )
    return replay_trace_state(
        trace_id=resolved_trace_id,
        events=events,
        verification=verification,
    )


def replay_trace_state(
    *,
    trace_id: str,
    events: list[dict[str, Any]],
    verification: AuditVerificationResult,
) -> TraceState:
    """Derive terminal trace state from sanitized audit event dictionaries."""
    state: dict[str, Any] = {
        "trace_id": trace_id,
        "latest_risk_tier": None,
        "latest_policy_action": None,
        "human_review_required": False,
        "human_review_assigned": False,
        "human_review_completed": False,
        "final_human_review_decision": None,
        "final_human_review_rationale": None,
        "reviewer_role": None,
        "reviewer_id_hmac": None,
        "audit_chain_valid": verification.valid,
        "audit_chain_errors": verification.errors,
        "event_count": len(events),
    }

    for event in events:
        event_type = str(event.get("event_type", ""))
        action = event.get("action")
        details = event.get("details") or {}

        risk_tier = event.get("risk_tier") or details.get("risk_tier")
        if risk_tier:
            state["latest_risk_tier"] = risk_tier

        if event_type == "risk_classification" and action:
            state["latest_risk_tier"] = action

        if event_type == "policy_decision" and action:
            state["latest_policy_action"] = action

        if event_type == "human_escalation":
            state["human_review_required"] = (
                action == "required"
                or bool(details.get("human_review_required"))
                or bool(details.get("requires_human_review"))
            )

        if bool(details.get("human_review_required")):
            state["human_review_required"] = True

        if event_type == "human_review_assigned":
            state["human_review_required"] = True
            state["human_review_assigned"] = True

        if event_type == "human_review_completed":
            state["human_review_completed"] = True
            decision = details.get("human_review_decision")
            state["final_human_review_decision"] = details.get("review_decision")
            state["final_human_review_rationale"] = details.get("review_rationale")
            state["reviewer_role"] = details.get("reviewer_role")
            state["reviewer_id_hmac"] = details.get("reviewer_id_hmac")
            if isinstance(decision, dict):
                state["final_human_review_decision"] = decision.get(
                    "decision",
                    state["final_human_review_decision"],
                )
                state["final_human_review_rationale"] = decision.get(
                    "rationale",
                    state["final_human_review_rationale"],
                )
                state["reviewer_role"] = decision.get(
                    "reviewer_role",
                    state["reviewer_role"],
                )
                state["reviewer_id_hmac"] = decision.get(
                    "reviewer_id_hmac",
                    state["reviewer_id_hmac"],
                )

    return TraceState.model_validate(state)


def hmac_reviewer_id(
    reviewer_id: str,
    *,
    key: str | None = None,
    salt: str | None = None,
) -> str:
    normalized_reviewer_id = reviewer_id.strip()
    if not normalized_reviewer_id:
        raise ValueError("reviewer_id is required")

    secret = key if key is not None else os.getenv(REVIEW_HMAC_KEY_ENV)
    if not secret:
        raise ReviewConfigurationError(
            f"{REVIEW_HMAC_KEY_ENV} must be set before recording reviewer identity"
        )

    namespace = salt or os.getenv(REVIEW_HMAC_SALT_ENV, DEFAULT_REVIEW_HMAC_SALT)
    message = f"{namespace}:{normalized_reviewer_id}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), message, sha256).hexdigest()
    return f"hmac-sha256:v1:{digest}"


def sanitize_review_rationale(rationale: str) -> str:
    stripped = rationale.strip()
    if not stripped:
        raise ValueError("rationale is required")

    return sanitize_review_rationale_text(stripped)


def _format_datetime_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00",
        "Z",
    )
