# Managed GCP Agent Runtime Profile

This profile documents how the ADK agent would be deployed to a managed GCP agent runtime. Verify the current product name and status before publishing or deploying.

## MVP Status

Documentation-only. Local tests and red-team evals do not require GCP access.

## Runtime Assumptions

- Runtime: managed GCP agent runtime for ADK agents.
- Identity: least-privilege workload identity or service account.
- Secrets: Secret Manager, not plaintext environment variables.
- Observability: Cloud Logging, Cloud Trace, Cloud Monitoring, and eval artifacts, with message-content capture disabled by default.
- Networking: approved endpoints only; private connectivity where required.
- Session/log privacy: production must verify whether raw user input is written to managed runtime session state, traces, or logs before callback redaction. Add ingest-time redaction or logging suppression before real PHI is used.

## Required Before Live Deployment

- Current product-name verification.
- Service account role review.
- Provider contract/privacy review.
- Safety-screening policy validation.
- Cloud Build approval gate.
- Session retention, trace/log redaction, and content-capture controls.
