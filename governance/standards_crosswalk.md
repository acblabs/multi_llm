# Standards Crosswalk

This is an architecture-readiness crosswalk, not a certification claim. Each row maps a concrete repo control to relevant Responsible AI, healthcare, and security expectations.

## Core MVP Scope

| Repo control | Evidence | NIST AI RMF | ISO/IEC 42001 | SOC 2 | HIPAA | OWASP LLM Top 10 | Healthcare/FDA/FHIR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Risk tiering | `risk.py`, `model_risk_tiering.md` | Map, Govern | AI risk assessment | Risk assessment | Risk analysis | Overreliance, excessive agency | SaMD boundary reasoning |
| Pre-router PHI/PII redaction | `pre_router.py` | Measure, Manage | Data governance | Confidentiality, Privacy | Minimum necessary, access control | Sensitive information disclosure | FHIR provenance/data minimization |
| Third-party egress redaction | `privacy.py`, `policy.py` | Measure, Manage | Operational controls | Confidentiality, Privacy | Minimum necessary, transmission security | Sensitive information disclosure | External provider boundary |
| Provider egress policy | `policy.py` | Manage | Operational controls | Security, Confidentiality | Transmission security | Insecure plugin/tool design | External provider boundary |
| HITL escalation | `escalation.py` | Govern, Manage | Human oversight | Control activities | Workforce/process safeguards | Overreliance | Prior-auth decision-support boundary |
| Audit trace (single trace ID per request) | `audit.py`, `observability.py` (`ensure_trace_id`), `pre_router.py` | Govern, Measure | Monitoring and evidence | Logging/monitoring | Audit controls | Logging and monitoring | Traceability to source documents |
| Reviewer evidence packet | `evidence_packet.py`, `scripts/export_audit_packet.py`, `examples/evidence_packets/` | Govern, Measure | Evidence management | Monitoring, processing integrity | Audit controls | Logging and monitoring | Reviewer-ready trace package |
| Governance scorecard and repository CI | `.github/workflows/governance-ci.yml`, `scripts/generate_governance_scorecard.py`, `governance/governance_scorecard.md` | Govern, Measure | Monitoring and continual improvement | Monitoring | Risk management | Logging and monitoring | CI evidence for governance controls |
| Retry/fallback safety | `reliability.py`, `tools.py` | Manage | Operational resilience | Availability | Contingency operations | Model denial of service | Safe degradation |
| Schema enforcement | `schemas.py` | Measure | Lifecycle controls | Processing integrity | Integrity controls | Insecure output handling | Structured evidence records |
| Red-team eval | `scripts/run_redteam_eval.py`, `evals/redteam/prior_auth_redteam_cases.json`, `evals/redteam/prior_auth_redteam_report.md` | Measure | Evaluation | Monitoring | Risk management | Prompt injection, sensitive disclosure, excessive agency, model denial of service | High-risk scenario testing |
| Redaction benchmark | `evals/privacy/run_redaction_benchmark.py`, `evals/privacy/labeled_phi_cases.jsonl`, `evals/privacy/redaction_benchmark_report.md` | Measure | Evaluation, data governance | Privacy monitoring | Minimum necessary, risk analysis | Sensitive information disclosure | PHI detection regression evidence |
| Prior-auth structured invariance regression | `evals/fairness/run_invariance_eval.py`, `evals/fairness/prior_auth_invariance_cases.jsonl`, `evals/fairness/invariance_report.md` | Measure, Manage | Bias and performance monitoring | Risk monitoring | Risk analysis | Overreliance, excessive agency | Synthetic invariance regression evidence |

## Conditional Scope

CMMC Level 2, NIST SP 800-171, and FedRAMP are out of scope for the base commercial-healthcare MVP unless a federal/CUI deployment scenario is added. If added later, each must map to real evidence rather than keyword-only rows.

GDPR and ISO/IEC 27001 can be included where concrete evidence exists, such as data minimization, retention, access control, incident response, and privacy-impact review.
