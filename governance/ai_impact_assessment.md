# AI Impact Assessment: Governed Multi-LLM Prior-Authorization Assistant

## Intended Use

The MVP supports prior-authorization summarization and documentation review. It is intended to help a human reviewer organize facts, identify missing materials, and prepare a draft summary.

The evidence coverage report is a structured documentation checklist. It identifies supplied, missing, insufficient, or not-applicable documentation elements for human review, but it does not decide coverage, medical necessity, diagnosis, or treatment.

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
| Prompt injection | 36-case deterministic red-team eval and safety-screening deployment profile |
| Redaction regression | Privacy benchmark with gated email, formatted phone, SSN, and member ID recall plus non-gated bare-name and bare-date limitation reporting |
| Demographic or identity sensitivity in prior-auth documentation triage | Synthetic prior-auth structured invariance regression comparing evidence statuses, human-review requirement, and prohibited decision boundaries |
| Hallucinated or unsupported claims | Evidence coverage report, decision-support boundary, structured governance explanations, audit trace, and red-team hallucinated-evidence cases |
| Unsafe fallback | `reliability.py` retry/fallback audit events |
| Lack of accountability | RACI, residual risk register, board risk report |

## Release Decision

MVP status: architecture demonstration only. The deterministic evals provide regression evidence for the implemented control plane, but they do not establish production-grade PHI detection, production fairness, or adversarial robustness. Production use requires formal privacy review, security review, clinical operations approval, managed GCP boundary review, provider contracting/BAA review, and deployment evidence.
