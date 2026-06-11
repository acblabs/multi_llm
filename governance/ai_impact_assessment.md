# AI Impact Assessment: Governed Multi-LLM Prior-Authorization Assistant

## Intended Use

The MVP supports prior-authorization summarization and documentation review. It is intended to help a human reviewer organize facts, identify missing materials, and prepare a draft summary.

## Out-Of-Scope Uses

- Final coverage approval or denial.
- Diagnosis, treatment, prescribing, or medical-necessity determination.
- Autonomous submission to payers.
- Use with unredacted PHI/PII sent to external providers.

## Impacted Stakeholders

- Patients whose access to care could be affected by poor summaries.
- Clinical operations reviewers.
- Privacy, compliance, legal, and information security teams.
- Engineering teams responsible for model orchestration and release controls.

## Key Risks And Controls

| Risk | Control |
| --- | --- |
| PHI/PII leakage to model providers | `pre_router.py` redaction before Gemini plus `privacy.py` redaction and `policy.py` egress block before third-party providers |
| Automation bias | HITL escalation and system-card limitations |
| Prompt injection | Red-team eval and safety-screening deployment profile |
| Hallucinated or unsupported claims | Multi-LLM synthesis boundary and audit trace |
| Unsafe fallback | `reliability.py` retry/fallback audit events |
| Lack of accountability | RACI, residual risk register, board risk report |

## Release Decision

MVP status: architecture demonstration only. Production use requires formal privacy review, security review, clinical operations approval, managed GCP boundary review, provider contracting/BAA review, and deployment evidence.
