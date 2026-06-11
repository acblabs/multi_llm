import os


MODEL_CONFIG = {
    "openai": os.getenv("OPENAI_MODEL", "gpt-5.5"),
    "claude": os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8"),
    "grok": os.getenv("XAI_MODEL", "xai/grok-4.3"),
    "gemini": os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
}

PRICING = {
    "openai": 0.00001,     # placeholder per token
    "claude": 0.000008,
    "grok": 0.000006,
}
