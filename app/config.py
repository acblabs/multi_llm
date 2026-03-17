import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "GCP_REGION")
STAGING_BUCKET = os.environ["GOOGLE_CLOUD_STAGING_BUCKET"]

# Native Gemini on Vertex
GEMINI_MODEL = "gemini-2.5-flash"

# Non-Google providers via LiteLLM
OPENAI_MODEL = "openai/gpt-5.1"
CLAUDE_MODEL = "anthropic/claude-sonnet-4-6"
GROK_MODEL = "xai/grok-4"
