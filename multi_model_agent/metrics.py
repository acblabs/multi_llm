from .config import PRICING

usage_log = []

def log_usage(provider: str, tokens: int):
    cost = tokens * PRICING.get(provider, 0)

    usage_log.append({
        "provider": provider,
        "tokens": tokens,
        "cost": cost,
    })

    return f"[{provider}: {tokens} tokens | ${round(cost, 4)}]"


def get_usage_summary():
    total_cost = sum(x["cost"] for x in usage_log)
    total_tokens = sum(x["tokens"] for x in usage_log)

    return {
        "total_tokens": total_tokens,
        "total_cost": round(total_cost, 4),
        "calls": usage_log,
    }
