import random
import time
from collections.abc import Callable
from typing import TypeVar

from .audit import append_audit_event
from .explainer import explain_retry_decision
from .telemetry import METRIC_RETRY_COUNT, record_governance_metric


T = TypeVar("T")


def classify_error(error: Exception) -> str:
    msg = str(error).lower()

    if any(token in msg for token in ("timeout", "connection", "rate_limit", "429")):
        return "retry"

    if any(token in msg for token in ("overloaded", "internal_server_error", "500")):
        return "fallback"

    if any(token in msg for token in ("authentication", "api key", "invalid_request", "400")):
        return "fail"

    return "fallback"


def retry_with_backoff(
    func: Callable[[], T],
    *,
    trace_id: str,
    provider: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> T:
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as error:
            error_type = classify_error(error)
            if error_type == "retry":
                record_governance_metric(
                    METRIC_RETRY_COUNT,
                    1,
                    {
                        "governance.trace_id": trace_id,
                        "governance.provider": provider,
                        "governance.retry_count": attempt + 1,
                    },
                )
            explanation = explain_retry_decision(
                trace_id=trace_id,
                provider=provider,
                action=error_type,
                attempt=attempt + 1,
                max_retries=max_retries,
            )
            append_audit_event(
                trace_id=trace_id,
                event_type="provider_retry_decision",
                provider=provider,
                action=error_type,
                details={
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "error": str(error),
                    "governance_explanation": explanation.to_safe_dict(),
                },
            )

            if error_type != "retry" or attempt == max_retries - 1:
                raise

            delay = base_delay * (2**attempt)
            jitter = random.uniform(0.1, 0.3) * delay
            sleeper(delay + jitter)

    raise RuntimeError("retry loop exited without returning or raising")
