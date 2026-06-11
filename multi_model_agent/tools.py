import os

from .config import MODEL_CONFIG
from .governance import (
    GovernanceBlockedError,
    governance_failure_message,
    prepare_provider_request,
    record_provider_failure,
    record_provider_success,
)
from .metrics import log_usage
from .observability import ensure_trace_id
from .reliability import classify_error, retry_with_backoff


FALLBACK_CHAIN = {
    "claude": ["openai", "grok"],
    "openai": ["claude", "grok"],
    "grok": ["openai", "claude"],
}


def _api_key_for(provider: str) -> str | None:
    env_vars = {
        "openai": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "grok": "XAI_API_KEY",
    }
    return os.getenv(env_vars[provider])


def _call_litellm_with_retry(
    provider: str,
    model: str,
    api_key: str | None,
    prompt: str,
    trace_id: str,
):
    def _inner():
        import litellm

        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
        )

        content = response["choices"][0]["message"]["content"]
        tokens = response.get("usage", {}).get("total_tokens", 0)
        return content, tokens

    return retry_with_backoff(_inner, trace_id=trace_id, provider=provider)


def graceful_failure() -> str:
    return "Unable to complete request due to provider issues. Please try again."


def _handle_fallback(
    provider: str,
    prompt: str,
    error: Exception,
    trace_id: str | None,
) -> str:
    if classify_error(error) == "fail":
        return graceful_failure()

    for fallback in FALLBACK_CHAIN.get(provider, []):
        try:
            return _call_provider(
                fallback,
                prompt,
                fallback_allowed=False,
                trace_id=trace_id,
                propagate_failure=True,
            )
        except Exception:
            continue

    return graceful_failure()


def _call_provider(
    provider: str,
    prompt: str,
    fallback_allowed: bool = True,
    trace_id: str | None = None,
    propagate_failure: bool = False,
) -> str:
    request = None
    context = None

    try:
        request, context, _ = prepare_provider_request(
            provider=provider,
            prompt=prompt,
            trace_id=trace_id,
        )
        content, tokens = _call_litellm_with_retry(
            provider,
            MODEL_CONFIG[provider],
            _api_key_for(provider),
            request.prompt,
            request.trace_id,
        )
        log_usage(provider, tokens)
        record_provider_success(
            trace_id=request.trace_id,
            provider=provider,
            tokens=tokens,
            risk_tier=context.risk.risk_tier,
        )
        return content

    except GovernanceBlockedError as error:
        return governance_failure_message(error)

    except Exception as error:
        if request is not None:
            record_provider_failure(
                trace_id=request.trace_id,
                provider=provider,
                error=error,
                action=classify_error(error),
                risk_tier=request.risk_tier,
            )

        if propagate_failure:
            raise

        if fallback_allowed:
            return _handle_fallback(
                provider,
                prompt,
                error,
                trace_id=request.trace_id if request is not None else trace_id,
            )

        return graceful_failure()


def _trace_id_from_tool_context(tool_context) -> str | None:
    """Reuse the trace ID the pre-router callback stored in ADK session state so
    tool-call governance events correlate with the rest of the invocation.

    Returns None when no ADK tool context/state is available (direct calls or
    tests), in which case the governance layer mints a fresh trace ID. The
    parameter is named ``tool_context`` so ADK injects its ToolContext and
    excludes it from the model-visible tool schema; this module stays free of
    any ADK import.
    """
    if tool_context is None:
        return None
    state = getattr(tool_context, "state", None)
    if state is None:
        return None
    return ensure_trace_id(state)


def call_openai(prompt: str, fallback_allowed: bool = True, tool_context=None) -> str:
    return _call_provider(
        "openai",
        prompt,
        fallback_allowed=fallback_allowed,
        trace_id=_trace_id_from_tool_context(tool_context),
    )


def call_claude(prompt: str, fallback_allowed: bool = True, tool_context=None) -> str:
    return _call_provider(
        "claude",
        prompt,
        fallback_allowed=fallback_allowed,
        trace_id=_trace_id_from_tool_context(tool_context),
    )


def call_grok(prompt: str, fallback_allowed: bool = True, tool_context=None) -> str:
    return _call_provider(
        "grok",
        prompt,
        fallback_allowed=fallback_allowed,
        trace_id=_trace_id_from_tool_context(tool_context),
    )
