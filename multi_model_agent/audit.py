import os
import sys
from pathlib import Path
from typing import Any

from .audit_store import (
    AuditStore,
    InMemoryAuditStore,
    JsonlAuditStore,
    stored_event_to_audit_dict,
)
from .schemas import AuditEvent, AuditVerificationResult, RiskTier


def _default_audit_store() -> AuditStore:
    if _use_memory_store_by_default():
        return InMemoryAuditStore()
    path = Path(os.getenv("MULTI_LLM_AUDIT_LOG_PATH", "audit_logs/dev_audit.jsonl"))
    return JsonlAuditStore(path)


def _use_memory_store_by_default() -> bool:
    configured_store = os.getenv("MULTI_LLM_AUDIT_STORE", "").strip().lower()
    if configured_store in {"memory", "in_memory", "test"}:
        return True
    if configured_store in {"jsonl", "persistent"}:
        return False
    if os.getenv("MULTI_LLM_AUDIT_LOG_PATH"):
        return False

    # Unit tests should not need per-test discipline to avoid local JSONL writes.
    return any(
        module_name == "unittest" or module_name.startswith("unittest.")
        for module_name in sys.modules
    )


_AUDIT_STORE: AuditStore = _default_audit_store()


def set_audit_store(store: AuditStore) -> None:
    global _AUDIT_STORE
    _AUDIT_STORE = store


def reset_audit_store() -> None:
    set_audit_store(_default_audit_store())


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
    _AUDIT_STORE.append(event)
    return event


def get_audit_log(trace_id: str | None = None) -> list[dict[str, Any]]:
    if trace_id is None:
        events = _AUDIT_STORE.list_events()
    else:
        events = _AUDIT_STORE.query_by_trace_id(trace_id)
    return [stored_event_to_audit_dict(event) for event in events]


def verify_audit_chain() -> AuditVerificationResult:
    return _AUDIT_STORE.verify_chain()


def clear_audit_log() -> None:
    _AUDIT_STORE.clear()
