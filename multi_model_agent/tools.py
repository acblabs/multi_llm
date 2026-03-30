import os
import time
import random
from .config import MODEL_CONFIG
from .metrics import log_usage

# =====================================================
# ERROR CLASSIFICATION
# =====================================================

def classify_error(e: Exception) -> str:
    msg = str(e).lower()

    # --- Retryable errors ---
    if "timeout" in msg or "connection" in msg:
        return "retry"

    if "rate_limit" in msg or "429" in msg:
        return "retry"

    # --- Immediate fallback errors ---
    if "overloaded" in msg:
        return "fallback"

    if "internal_server_error" in msg or "500" in msg:
        return "fallback"

    # --- Fail fast ---
    if "authentication" in msg or "api key" in msg:
        return "fail"

    if "invalid_request" in msg or "400" in msg:
        return "fail"

    return "fallback"


# =====================================================
# RETRY WITH BACKOFF (ERROR-AWARE)
# =====================================================

def retry_with_backoff(func, max_retries=3, base_delay=1.0):
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                error_type = classify_error(e)

                # Only retry retryable errors
                if error_type != "retry":
                    raise e

                if attempt == max_retries - 1:
                    raise e

                delay = base_delay * (2 ** attempt)
                jitter = random.uniform(0.1, 0.3) * delay
                time.sleep(delay + jitter)

        return None
    return wrapper


# =====================================================
# CORE LITELLM CALL
# =====================================================

def _call_litellm_with_retry(model, api_key, prompt):

    @retry_with_backoff
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

    return _inner()


# =====================================================
# FALLBACK CHAIN (PRIORITY ORDER)
# =====================================================

FALLBACK_CHAIN = {
    "claude": ["openai", "grok"],
    "openai": ["claude", "grok"],
    "grok": ["openai", "claude"],
}


# =====================================================
# GRACEFUL FAILURE
# =====================================================

def graceful_failure():
    return "⚠️ Unable to complete request due to provider issues. Please try again."


# =====================================================
# FALLBACK HANDLER (ERROR-AWARE)
# =====================================================

def _handle_fallback(provider: str, prompt: str, error: Exception) -> str:
    error_type = classify_error(error)

    # --- Fail fast ---
    if error_type == "fail":
        return graceful_failure()

    # --- Fallback ---
    fallback_providers = FALLBACK_CHAIN.get(provider, [])

    for fallback in fallback_providers:
        try:
            if fallback == "openai":
                return call_openai(prompt, fallback_allowed=False)

            elif fallback == "claude":
                return call_claude(prompt, fallback_allowed=False)

            elif fallback == "grok":
                return call_grok(prompt, fallback_allowed=False)

        except Exception:
            continue

    return graceful_failure()


# =====================================================
# TOOL FUNCTIONS
# =====================================================

def call_openai(prompt: str, fallback_allowed=True) -> str:
    try:
        content, tokens = _call_litellm_with_retry(
            MODEL_CONFIG["openai"],
            os.getenv("OPENAI_API_KEY"),
            prompt
        )
        log_usage("openai", tokens)
        return content

    except Exception as e:
        if fallback_allowed:
            return _handle_fallback("openai", prompt, e)
        return graceful_failure()


def call_claude(prompt: str, fallback_allowed=True) -> str:
    try:
        content, tokens = _call_litellm_with_retry(
            MODEL_CONFIG["claude"],
            os.getenv("ANTHROPIC_API_KEY"),
            prompt
        )
        log_usage("claude", tokens)
        return content

    except Exception as e:
        if fallback_allowed:
            return _handle_fallback("claude", prompt, e)
        return graceful_failure()


def call_grok(prompt: str, fallback_allowed=True) -> str:
    try:
        content, tokens = _call_litellm_with_retry(
            MODEL_CONFIG["grok"],
            os.getenv("XAI_API_KEY"),
            prompt
        )
        log_usage("grok", tokens)
        return content

    except Exception as e:
        if fallback_allowed:
            return _handle_fallback("grok", prompt, e)
        return graceful_failure()
