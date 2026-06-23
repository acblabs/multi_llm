# Model Risk Tiering

This project uses risk tiering to decide which controls must run before a multi-LLM provider call is allowed.

## Tiers

| Tier | Use case | Required controls |
| --- | --- | --- |
| Minimal | General productivity with no sensitive data or healthcare impact | Basic logging, schema validation |
| Limited | Healthcare-adjacent support with no direct access-to-care impact | Privacy scan, policy check, audit event |
| High | Prior authorization, benefit access, medical-necessity support, or PHI/PII | PHI/PII redaction, provider policy, structured explanations, evidence coverage report, HITL escalation, audit trace, red-team evidence, privacy benchmark evidence, structured invariance evidence |
| Prohibited | Autonomous diagnosis, treatment, final coverage approval/denial, or bypassing human review | Block execution and route to governance review |

## MVP Classification

The prior-authorization summarization path is classified as high risk because it can influence access to care if misused. The MVP keeps it inside an administrative decision-support boundary:

- the agent may summarize and organize provided evidence;
- the agent may identify missing documentation;
- the agent may label prior-authorization documentation elements as present, missing, insufficient, or not applicable for human review;
- the agent must not approve, deny, diagnose, prescribe, or determine medical necessity;
- a human reviewer is required before operational use.

## Evidence In Code

- `multi_model_agent/risk.py` classifies prior-authorization requests as high risk.
- `multi_model_agent/pre_router.py` redacts PHI/PII before the Gemini router model call.
- `multi_model_agent/privacy.py` redacts PHI/PII before provider egress.
- `multi_model_agent/policy.py` blocks unredacted sensitive data from external providers.
- `multi_model_agent/explainer.py` attaches reason codes, policy IDs, and safe rationales.
- `multi_model_agent/evidence_coverage.py` creates the prior-auth documentation coverage report without raw source excerpts.
- `multi_model_agent/escalation.py` requires human review for high-risk requests.
- `multi_model_agent/audit.py` records sanitized control decisions.
- `scripts/run_redteam_eval.py` measures deterministic prompt-injection, PHI-exfiltration, egress, autonomy-boundary, fallback, fanout, and hallucinated-evidence controls.
- `evals/privacy/run_redaction_benchmark.py` measures redaction precision, recall, F1, and per-identifier metrics with realistic gates for the current regex redactor.
- `evals/fairness/run_invariance_eval.py` compares structured evidence-coverage outputs across synthetic demographic variants rather than free-text rationales.
