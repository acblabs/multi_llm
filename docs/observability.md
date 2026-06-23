# Observability

This project supports optional OpenTelemetry-style governance observability while
keeping local and test runs dependency-free. Telemetry is operational debugging
evidence; the PHI-safe JSONL audit chain remains the durable audit evidence.

## Runtime Modes

Default mode is no-op:

```bash
python -m unittest discover -s tests
```

No collector, SDK, exporter, or OpenTelemetry package is required for imports or
normal local execution.

Enable OpenTelemetry API usage with:

```bash
set MULTI_LLM_OTEL_ENABLED=true
set OTEL_SERVICE_NAME=multi-llm-governance
```

If `opentelemetry-api` is available, spans and metrics are sent to the active
provider. If only the API is installed and no SDK/provider is configured, the API
behaves as a no-op.

To let this repo configure a simple OTLP HTTP trace exporter when optional SDK
packages are installed:

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
set MULTI_LLM_OTEL_ENABLED=true
set MULTI_LLM_OTEL_AUTO_CONFIGURE=true
set OTEL_SERVICE_NAME=multi-llm-governance
set OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

The auto-configure path is best for local demos. Production should configure
OpenTelemetry providers in the host runtime so sampling, exporters, credentials,
and resource attributes are centrally managed.

## Span Model

The telemetry wrapper emits these control-plane span names where the repo owns
the execution boundary:

```text
request
pre_router.redaction
risk.classification
policy.egress_check
provider.openai.call
provider.claude.call
provider.grok.call
synthesis.gemini
output.schema_validation
escalation.hitl
```

`synthesis.gemini` is a safe handoff marker because the actual Gemini model call
is managed by ADK outside this repository's Python wrapper. The pre-router span
captures the redaction work this repo controls before that handoff.

## Attributes

The allowlist includes GenAI semantic convention fields when available:

```text
gen_ai.system
gen_ai.request.model
gen_ai.response.model
gen_ai.usage.input_tokens
gen_ai.usage.output_tokens
```

It also includes governance attributes:

```text
governance.trace_id
governance.risk_tier
governance.human_review_required
governance.redaction_count
governance.policy_ids
governance.escalation_reason
governance.fallback_used
governance.retry_count
governance.estimated_cost_usd
```

Prompts, responses, raw PHI, raw excerpts, source text, reviewer IDs, exception
messages, and arbitrary unallowlisted fields are not attached as span or metric
attributes. Attribute values are redacted and constrained before export.

## Metrics

The wrapper can emit these operational metric names:

```text
request_count
error_count
provider_latency_ms
retry_count
fallback_count
estimated_cost_usd
risk_tier_distribution
redaction_count_distribution
policy_violation_attempts
human_escalations_total
```

Rate-shaped values are derived downstream, not emitted as counters:

```text
provider_error_rate = error_count / request_count
human_escalation_rate = human_escalations_total / request_count
```

Local usage summaries in `multi_model_agent/metrics.py` remain ephemeral demo
state. Durable metric events that are intentionally persisted through
`observability.record_metric()` flow through the Phase 1 audit sanitizer and hash
chain.

## Correlation With Audit Evidence

Telemetry spans are useful for runtime debugging and latency/cost inspection, but
they are not the system of record. The durable evidence path is the sanitized
audit chain plus exported evidence packets. Use the shared trace ID to correlate
telemetry with audit events, and treat the audit event as authoritative when the
two disagree.

Production telemetry hardening should include collector access control,
retention limits, sampling policy review, exporter credential management, and
explicit disabling of any platform feature that captures prompts, responses,
request bodies, raw exception payloads, or model message content.

## Test Capture

Unit tests use in-memory capture rather than a collector:

```python
from multi_model_agent.telemetry import configure_telemetry

configure_telemetry(
    enabled=True,
    capture_spans=True,
    capture_metrics=True,
    use_otel=False,
)
```

This mode is deterministic and verifies the same attribute sanitizer used before
OpenTelemetry export.

## Limitations

Telemetry is not audit evidence unless the same event is explicitly persisted
through the sanitized audit sink. Collector storage, sampling, retention,
cross-service correlation, and dashboarding are deployment concerns outside this
MVP. Production should review collector access controls and disable any platform
feature that captures request bodies, model prompts, model responses, or raw
exception payloads.
