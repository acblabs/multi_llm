# Governed Multi-LLM Agent Architecture

Responsible AI control plane for a multi-LLM prior-authorization decision-support workflow.

This repo demonstrates how to keep the core multi-LLM engineering story intact while adding the governance controls expected in regulated healthcare AI: pre-router PHI/PII redaction, risk tiering, defense-in-depth redaction before third-party egress, approved-provider egress policy, structured governance explanations, prior-auth evidence coverage reports, human-in-the-loop escalation, retry/fallback safety, schema contracts for governance artifacts, observability helpers, and audit evidence.

This is an architecture MVP, not a production medical or coverage-decision system.

## Architecture At A Glance

```text
User request
  -> pre-router PHI/PII redaction
  -> ADK/Gemini orchestrator inside managed GCP boundary
  -> Responsible AI egress control plane
       -> risk tiering
       -> PHI/PII redaction for third-party providers
       -> approved-provider egress policy
       -> structured governance explanations
       -> evidence coverage report for prior authorization
       -> HITL escalation decision
       -> audit trace
  -> Multi-LLM data plane
       -> OpenAI / Claude / Grok perspectives
       -> Gemini synthesis
       -> retry/fallback and cost tracking
  -> decision-support output for human review
```

The first vertical slice is prior-authorization summarization:

```text
Prior-auth request
  -> pre-router PHI/PII redaction
  -> Gemini router
  -> risk tiering
  -> defense-in-depth PHI/PII redaction before third-party egress
  -> approved-provider egress policy
  -> structured governance explanations
  -> evidence coverage report
  -> multi-LLM orchestration
  -> HITL escalation
  -> sanitized audit events
  -> red-team eval evidence
```

## Multi-LLM Data Plane

The agent still demonstrates modern AI orchestration:

- provider diversity through LiteLLM;
- Gemini as orchestrator and synthesizer;
- provider-specific strengths from OpenAI, Claude, and Grok;
- retry with backoff, fallback chains, and graceful degradation;
- token and cost tracking.

The local usage summary in `metrics.py` is ephemeral demo telemetry, not audit evidence. Durable metric events should flow through `observability.record_metric()`, which writes sanitized audit events with a trace ID.

Default model versions:

| Role | Provider | LiteLLM model ID |
| --- | --- | --- |
| Orchestrator / synthesizer | Gemini | `gemini-3.5-flash` |
| Structured implementation perspective | OpenAI | `gpt-5.5` |
| Deep reasoning perspective | Claude | `claude-opus-4-8` |
| Alternative exploration perspective | Grok | `xai/grok-4.3` |

## Responsible AI Control Plane

The MVP governance path is implemented in code:

| Control | File |
| --- | --- |
| Pre-router PHI/PII redaction | `multi_model_agent/pre_router.py` |
| Risk tiering | `multi_model_agent/risk.py` |
| PHI/PII redaction before third-party egress | `multi_model_agent/privacy.py` |
| Provider egress policy | `multi_model_agent/policy.py` |
| Structured governance explanations | `multi_model_agent/explainer.py` |
| Prior-auth evidence coverage report | `multi_model_agent/evidence_coverage.py` |
| HITL escalation | `multi_model_agent/escalation.py` |
| PHI-safe audit trace and hash-chain verification | `multi_model_agent/audit.py`, `multi_model_agent/audit_store.py`, `multi_model_agent/audit_hashing.py` |
| Schema contracts | `multi_model_agent/schemas.py` |
| Retry/fallback safety | `multi_model_agent/reliability.py` |
| Provider tool integration | `multi_model_agent/tools.py` |

The ADK agent uses `before_model_callback=redact_before_model`, which redacts text in the Gemini router request before the model call. External provider calls are then prepared through `prepare_provider_request()`, which redacts sensitive data again and records privacy, risk, policy, evidence coverage, and escalation events before egress to non-Google third-party LLMs.

For prior-authorization workflows, `evidence_coverage.py` produces a deterministic `EvidenceCoverageReport` that labels required documentation elements as `present`, `missing`, `insufficient`, or `not_applicable`. The report stores source references and hashes over redacted excerpts, not raw excerpts. It is support for human review only; it must not approve care, deny coverage, determine medical necessity, diagnose, or recommend treatment.

Trust boundary note: the local ADK process receives user input, but no model should receive raw PHI/PII by default. The MVP redacts the ADK model request before the Gemini router call and redacts again before third-party provider calls. Raw PHI/PII may still exist in ADK session state/history or platform logs before callback redaction; production requires ingest-time redaction before session write and strict content-logging controls.

MVP redaction is text-only and destructive. That is intentional for the prior-auth demo because the human reviewer retains the source document, but production may require stable pseudonymous tokens and managed inspection for files, images, and attachments.

## Current MVP Limitations

The current controls are intentionally deterministic and testable, but they are not production-grade clinical or coverage-decision controls:

- `privacy.py` uses regular expressions and misses many real PHI forms, including unlabeled names, bare DOBs, international formats, context-free identifiers, homoglyphs, encoded/spaced values, and PII embedded in arbitrary JSON.
- `risk.py` uses lexical keyword heuristics and can be bypassed by paraphrase or indirect phrasing.
- `evidence_coverage.py` uses deterministic keyword heuristics. It is useful for reviewer-facing coverage checks, but it is not a payer-policy engine, clinical classifier, or medical-necessity model.
- Provider responses are plain text today. There is no enforced provider `response_format`, JSON schema, or post-response clinical-boundary validator on model outputs yet.
- Runtime objects such as `GovernanceContext` and `PrivacyAssessment` may contain raw PHI during request processing. Safe views and audit persistence prevent those values from being written to durable artifacts, but production needs stricter runtime data-minimization and logging controls.
- The deterministic red-team suite is still small and should be expanded before claiming broad adversarial robustness.

## PHI-Safe Audit Chain

Audit events are sanitized before persistence and before hashing. The persistence allowlist favors structured fields such as provider, action, risk tier, reason codes, policy IDs, model provenance, redaction summaries, token counts, evidence coverage reports, hashed redacted excerpts, and safe error categories. Generic free-text fields such as raw reasons, names, arbitrary values, prompts, responses, excerpts, and reviewer IDs are not persisted.

The default local sink writes JSONL to `audit_logs/dev_audit.jsonl`; tests can swap in an in-memory adapter through the audit facade. Verify a local log with:

```bash
python scripts/verify_audit_chain.py audit_logs/dev_audit.jsonl
```

The JSONL hash chain is integrity-verifiable, not tamper-proof. It detects accidental edits, event reordering, middle deletion, and insertion when the verifier checks the chain links and hashes. A process-local thread lock protects demo appends; multi-process writers need a file lock, database, append-only object store, or external audit service. Appending reloads and verifies the existing chain, which is simple and defensible for this MVP but O(n) per write. A partial final-line write from a crash requires manual log rotation or recovery before appends resume. A user with filesystem write access could still rewrite the whole file and recompute hashes unless production storage adds protected HMAC/signing keys, immutable storage, object-store versioning, append-only controls, or external digest anchoring.

If raw PHI is accidentally persisted, rotate the affected log, preserve any restricted incident copy only as policy requires, write a documented scrub event to a new log, and accept the chain break rather than silently rewriting history.

## Evidence

High-signal artifacts:

- [Architecture](docs/architecture.md)
- [ADR: governed prior-authorization slice](docs/adr/0001-governed-prior-authorization-slice.md)
- [Model risk tiering](governance/model_risk_tiering.md)
- [System card](governance/system_card.md)
- [AI impact assessment](governance/ai_impact_assessment.md)
- [Standards crosswalk](governance/standards_crosswalk.md)
- [EU AI Act applicability reasoning](governance/eu_ai_act_applicability.md)
- [FDA SaMD boundary reasoning](governance/fda_samd_boundary.md)
- [Governance operating model](governance/governance_operating_model.md)
- [Board risk report](governance/board_risk_report.md)
- [Residual risk register](governance/residual_risk_register.md)
- [Prior-auth walkthrough](examples/prior_authorization/governance_walkthrough.md)
- [Evidence coverage sample](examples/prior_authorization/evidence_coverage_sample.json)

## Red-Team Eval

Run the focused MVP red-team suite:

```bash
python scripts/run_redteam_eval.py
```

It checks prompt-injection and PHI-exfiltration cases without calling external LLM providers.

Run unit tests:

```bash
python -m unittest discover -s tests
```

## Optional Git Hooks

This repo includes opt-in hooks for local guardrails:

```bash
git config core.hooksPath .githooks
```

The pre-commit hook blocks generated audit logs by path and by parsing staged JSON/JSONL for `audit.v1` stored events. It also requires at least one core reviewer-facing doc to be staged whenever governance or implementation files are staged: `README.md`, `docs/architecture.md`, `examples/prior_authorization/governance_walkthrough.md`, `governance/system_card.md`, `governance/model_risk_tiering.md`, or `governance/ai_impact_assessment.md`. Test-only commits do not trigger this documentation gate. The hook then runs compile checks. The pre-push hook runs unit tests and the deterministic red-team eval. Set `PYTHON=/path/to/python` before invoking Git if your shell does not expose `python` on `PATH`. These hooks are useful local tripwires, but CI should remain the source of record for required governance checks.

## Deployment Profile

The repo is designed for Google ADK and a managed GCP agent runtime, but product names are intentionally contained in `deployment/gcp/` because Google Cloud agent services are evolving.

MVP deployment artifacts:

- `cloudbuild.yaml`
- `deployment/gcp/managed_agent_runtime_profile.md`
- `deployment/gcp/cloud_build_pipeline.md`
- `deployment/gcp/agent_inventory_manifest.yaml`
- `deployment/gcp/safety_screening_policy.yaml`

Cloud Build is validation-only in the MVP. Live deployment, agent inventory registration, governed capability registration, and managed safety-screening integration are expansion items until implemented for real.

## Standards Scope

Core MVP mappings focus on:

- NIST AI RMF
- ISO/IEC 42001
- SOC 2 Type 2
- HIPAA
- OWASP Top 10 for LLM Applications
- EU AI Act applicability reasoning
- FDA SaMD / GMLP boundary reasoning
- FHIR provenance and interoperability considerations

CMMC, NIST SP 800-171, and FedRAMP are explicitly out of scope unless a federal/CUI scenario is added.

## Local Setup

```bash
pip install -r multi_model_agent/requirements.txt
cp multi_model_agent/.env.example multi_model_agent/.env
adk web
```

Set real provider keys only in local secrets or a managed secret store. Do not commit `.env`. The default model IDs are also shown in `multi_model_agent/.env.example` and can be overridden with environment variables.
