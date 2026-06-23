from threading import RLock
from typing import Any

from .config import PRICING


_usage_log: list[dict[str, float | int | str]] = []
_usage_lock = RLock()


def log_usage(provider: str, tokens: int) -> str:
    """Record an ephemeral local usage summary.

    This is not audit evidence. Durable metric events flow through
    observability.record_metric(), which appends sanitized audit events with a
    trace ID. This helper supports local demos and human-readable tool output.
    """
    safe_provider = _safe_provider(provider)
    safe_tokens = _safe_token_count(tokens)
    cost = safe_tokens * PRICING.get(safe_provider, 0)

    with _usage_lock:
        _usage_log.append(
            {
                "provider": safe_provider,
                "tokens": safe_tokens,
                "cost": cost,
            }
        )

    return f"[{safe_provider}: {safe_tokens} tokens | ${round(cost, 4)}]"


def get_usage_summary() -> dict[str, Any]:
    with _usage_lock:
        calls = [dict(entry) for entry in _usage_log]

    total_cost = sum(entry["cost"] for entry in calls)
    total_tokens = sum(entry["tokens"] for entry in calls)

    return {
        "total_tokens": total_tokens,
        "total_cost": round(total_cost, 4),
        "calls": calls,
        "ephemeral": True,
        "audit_evidence": False,
    }


def clear_usage_log() -> None:
    with _usage_lock:
        _usage_log.clear()


def _safe_provider(provider: str) -> str:
    return provider if provider in PRICING else "unknown"


def _safe_token_count(tokens: int) -> int:
    if not isinstance(tokens, int):
        return 0
    return max(tokens, 0)
