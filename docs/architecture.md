# Architecture

```mermaid
flowchart TD
    U[User prompt] --> BAA
    subgraph BAA[Managed GCP / ADK boundary]
        ADK[ADK runtime and session state]
        PR[before_model_callback PHI/PII redaction]
        GEM[Gemini orchestrator]
        ADK --> PR
        PR -->|redacted model request| GEM
    end
    GEM --> G[Third-party egress control plane]
    G --> R[Risk tiering]
    G --> P[Defense-in-depth PHI/PII redaction]
    G --> E[Approved-provider egress policy]
    G --> H[HITL escalation decision]
    G --> A[Audit trace]
    E -->|redacted prompt only| O[OpenAI]
    E -->|redacted prompt only| C[Claude]
    E -->|redacted prompt only| X[Grok]
    O --> GEM
    C --> GEM
    X --> GEM
    GEM --> V[Schema/output validation]
    V --> A
    V --> F[Final decision-support answer]
```

## Data Plane

The data plane keeps the original multi-LLM architecture: provider diversity, retry/fallback, cost tracking, and Gemini synthesis.

## Control Plane

The Responsible AI control plane runs in two places:

- `pre_router.py` redacts text parts in the ADK model request before Gemini receives them. Raw user input may still exist inside the ADK runtime/session boundary before this callback runs.
- The third-party egress gate redacts and validates again before OpenAI, Claude, or Grok calls.

- `risk.py` classifies the request.
- `privacy.py` redacts sensitive identifiers.
- `policy.py` enforces approved-provider and redaction requirements.
- `escalation.py` determines human review.
- `audit.py` records every control decision.

## Traceability

A single trace ID correlates every governance event for one request. The `before_model_callback` stores a trace ID in ADK session state on the first model call (`observability.ensure_trace_id`), and the provider tools reuse it through their injected `tool_context`. As a result, the pre-router redaction event, risk classification, policy decision, HITL escalation, and each provider call/outcome for one invocation share the same trace ID in the audit log and can be retrieved together with `get_audit_log(trace_id)`.

When no ADK session state is available (direct calls or unit tests), the governance layer mints a fresh trace ID per request rather than failing.

## Trust Boundaries

- Local ADK runtime.
- Managed GCP agent runtime deployment profile.
- External model providers.
- Logs, audit events, metrics, and secrets.

The local ADK process receives the user input and may write it to session state/history before `before_model_callback=redact_before_model` mutates the ADK `LlmRequest`. The callback prevents raw PHI/PII from reaching the Gemini model request by default, and the MVP redacts again before egress to non-Google third-party LLM providers such as OpenAI, Claude, and Grok. Production use still requires ingest-time redaction before session write, content-logging suppression, retention controls, telemetry review, and BAA/contract review for the managed GCP boundary and every external provider.

MVP redaction scope is text-only. Attachments, images, files, and other `inline_data` parts require extraction/OCR and managed sensitive-data inspection before production use.

The MVP keeps GCP product names inside the deployment profile so the governance pattern remains portable.
