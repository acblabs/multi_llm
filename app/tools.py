from __future__ import annotations

from threading import Lock
from typing import Any
from litellm import completion, completion_cost

from .config import OPENAI_MODEL, CLAUDE_MODEL, GROK_MODEL


_USAGE_LOCK = Lock()

USAGE_TOTALS = {
    "calls": 0,
    "total_tokens": 0,
    "cost_usd": 0.0,
}


def _extract_content(response: Any) -> str:
    content = response.choices[0].message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif hasattr(item, "text"):
                parts.append(getattr(item, "text", ""))
        return "\n".join(p for p in parts if p)

    return str(content)


def _provider_name(model: str) -> str:
    if model.startswith("openai/"):
        return "OpenAI"
    if model.startswith("anthropic/"):
        return "Claude"
    if model.startswith("xai/"):
        return "Grok"
    return model


def _update_totals(total_tokens: int, cost_usd: float) -> None:
    with _USAGE_LOCK:
        USAGE_TOTALS["calls"] += 1
        USAGE_TOTALS["total_tokens"] += total_tokens
        USAGE_TOTALS["cost_usd"] += cost_usd


def _call_model(model: str, prompt: str) -> str:
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        timeout=60,
    )

    usage = getattr(response, "usage", None)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

    cost_usd = float(completion_cost(completion_response=response, model=model) or 0.0)

    _update_totals(total_tokens, cost_usd)

    content = _extract_content(response)

    provider = _provider_name(model)

    usage_footer = f"\n\n[{provider}: {total_tokens} tokens | ${cost_usd:.4f}]"

    return content + usage_footer


def ask_openai(prompt: str) -> str:
    """Use OpenAI (reasoning, coding)."""
    return _call_model(OPENAI_MODEL, prompt)


def ask_claude(prompt: str) -> str:
    """Use Claude (writing, summarization)."""
    return _call_model(CLAUDE_MODEL, prompt)


def ask_grok(prompt: str) -> str:
    """Use Grok (only when explicitly needed)."""
    return _call_model(GROK_MODEL, prompt)


def get_usage_summary() -> str:
    with _USAGE_LOCK:
        return (
            f"Calls: {USAGE_TOTALS['calls']}\n"
            f"Total tokens: {USAGE_TOTALS['total_tokens']}\n"
            f"Estimated cost: ${USAGE_TOTALS['cost_usd']:.4f}"
        )
