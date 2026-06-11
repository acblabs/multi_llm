from typing import Any

from .schemas import AuditEvent, RiskTier


_AUDIT_LOG: list[AuditEvent] = []


def append_audit_event(
    *,
    trace_id: str,
    event_type: str,
    provider: str | None = None,
    action: str | None = None,
    risk_tier: RiskTier | None = None,
    details: dict[str, Any] | None = None,
) -> AuditEvent:
    event = AuditEvent(
        trace_id=trace_id,
        event_type=event_type,
        provider=provider,
        action=action,
        risk_tier=risk_tier,
        details=details or {},
    )
    _AUDIT_LOG.append(event)
    return event


def get_audit_log(trace_id: str | None = None) -> list[dict[str, Any]]:
    events = _AUDIT_LOG
    if trace_id is not None:
        events = [event for event in events if event.trace_id == trace_id]
    return [event.model_dump(mode="json") for event in events]


def clear_audit_log() -> None:
    _AUDIT_LOG.clear()
