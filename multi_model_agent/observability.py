from typing import Any
from uuid import uuid4

from .audit import append_audit_event
from .schemas import RiskTier


def new_trace_id() -> str:
    return str(uuid4())


def get_state_value(state: Any, key: str) -> Any:
    """Read a value from an ADK State, a dict, or a plain object."""
    if state is None:
        return None

    getter = getattr(state, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except TypeError:
            pass

    try:
        return state[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(state, key, None)


def set_state_value(state: Any, key: str, value: str) -> None:
    """Write a value to an ADK State, a dict, or a plain object."""
    if state is None:
        return

    try:
        state[key] = value
        return
    except (TypeError, AttributeError):
        pass

    setter = getattr(state, "set", None)
    if callable(setter):
        try:
            setter(key, value)
            return
        except TypeError:
            pass

    try:
        setattr(state, key, value)
    except Exception:
        return


def ensure_trace_id(state: Any) -> str:
    """Return the trace_id stored in state, creating and storing one if absent.

    This correlates the pre-router redaction event and the provider-tool
    governance events under a single trace ID across one ADK invocation.
    """
    existing = get_state_value(state, "trace_id")
    if existing:
        return str(existing)

    trace_id = new_trace_id()
    set_state_value(state, "trace_id", trace_id)
    return trace_id


def record_metric(
    *,
    trace_id: str,
    name: str,
    value: int | float | str | bool,
    provider: str | None = None,
    risk_tier: RiskTier | None = None,
) -> None:
    append_audit_event(
        trace_id=trace_id,
        event_type="metric",
        provider=provider,
        risk_tier=risk_tier,
        details={"name": name, "value": value},
    )
