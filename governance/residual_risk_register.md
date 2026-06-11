# Residual Risk Register

| Risk | Severity | Status | Owner | Control | Residual risk |
| --- | --- | --- | --- | --- | --- |
| Heuristic redaction misses uncommon PHI formats | High | Open | Privacy | Regex redaction, policy block, evals | Requires managed sensitive-data inspection before production |
| Raw PHI persists in ADK session state, history, or platform logs before model-request redaction | High | Open | Privacy/InfoSec | Pre-router model-request redaction, `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=FALSE`, no real PHI in MVP tests | Production requires ingest-time redaction before ADK session write, content-logging suppression, retention limits, access controls, and Cloud Logging/Trace review |
| Destructive redaction loses referential continuity across multiple patients or entities | Medium | Accepted for MVP | Product/Privacy | `[PATIENT_NAME]`, `[DOB]`, and `[MEMBER_ID]` replacement before model calls | Defensible for prior-auth summarization where the human reviewer has source documents; production may require pseudonymous stable tokens such as `[PATIENT_1]` |
| User over-trusts generated prior-auth summary | High | Mitigated | Product | HITL escalation, system card, README limits | Human process must enforce review |
| External provider stores or logs redacted context | Medium | Open | Legal/Privacy | Provider policy, BAA/contract review | Production requires provider review |
| Prompt injection attempts to bypass policy | High | Mitigated | InfoSec | Red-team eval, safety-screening profile | Managed screening recommended before production |
| Fallback provider changes output behavior | Medium | Mitigated | Engineering | Fallback audit events, provider attribution | Monitor fallback rates and quality |
