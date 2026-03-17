import os
import vertexai
from vertexai import agent_engines

from .config import PROJECT_ID, LOCATION
from app.agent import root_agent

STAGING_BUCKET = os.environ["GOOGLE_CLOUD_STAGING_BUCKET"]  # e.g. gs://my-agent-staging-bucket

def main():
    vertexai.init(
        project=PROJECT_ID,
        location=LOCATION,
	staging_bucket=STAGING_BUCKET,
    )

    app = agent_engines.AdkApp(
        agent=root_agent,
        enable_tracing=True,
    )

    remote_app = agent_engines.create(
        app,
        display_name="vertex-multi-model-agent",
        requirements=[
            "google-cloud-aiplatform[agent_engines,adk]>=1.112",
            "litellm>=1.63.0",
            "python-dotenv>=1.0.1",
        ],
        extra_packages=["./app"],
    )

    print("Deployed:")
    print(remote_app)

if __name__ == "__main__":
    main()
