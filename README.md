# Governed Multi-LLM Agent Architecture

Responsible AI control plane for a multi-LLM prior-authorization decision-support workflow.

This repo demonstrates how to keep the core multi-LLM engineering story intact while adding the governance controls expected in regulated healthcare AI: pre-router PHI/PII redaction, risk tiering, defense-in-depth redaction before third-party egress, approved-provider egress policy, human-in-the-loop escalation, retry/fallback safety, schema validation, observability, and audit evidence.

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
  -> multi-LLM orchestration
  -> HITL escalation
  -> audit event
  -> red-team eval evidence
```

## Multi-LLM Data Plane

The agent still demonstrates modern AI orchestration:

- provider diversity through LiteLLM;
- Gemini as orchestrator and synthesizer;
- provider-specific strengths from OpenAI, Claude, and Grok;
- retry with backoff, fallback chains, and graceful degradation;
- token and cost tracking.

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
| HITL escalation | `multi_model_agent/escalation.py` |
| Audit trace | `multi_model_agent/audit.py` |
| Schema contracts | `multi_model_agent/schemas.py` |
| Retry/fallback safety | `multi_model_agent/reliability.py` |
| Provider tool integration | `multi_model_agent/tools.py` |

The ADK agent uses `before_model_callback=redact_before_model`, which redacts text in the Gemini router request before the model call. External provider calls are then prepared through `prepare_provider_request()`, which redacts sensitive data again and records privacy, risk, policy, and escalation events before egress to non-Google third-party LLMs.

Trust boundary note: the local ADK process receives user input, but no model should receive raw PHI/PII by default. The MVP redacts the ADK model request before the Gemini router call and redacts again before third-party provider calls. Raw PHI/PII may still exist in ADK session state/history or platform logs before callback redaction; production requires ingest-time redaction before session write and strict content-logging controls.

MVP redaction is text-only and destructive. That is intentional for the prior-auth demo because the human reviewer retains the source document, but production may require stable pseudonymous tokens and managed inspection for files, images, and attachments.

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
