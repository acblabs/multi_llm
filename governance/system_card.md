# System Card

## System Name

Governed Multi-LLM Prior-Authorization Assistant

## Summary

This system demonstrates how a multi-LLM agent can be governed with pre-router PHI/PII redaction, risk tiering, defense-in-depth redaction before third-party egress, approved-provider egress policy, human escalation, retry/fallback controls, schema validation, and audit logging.

## Intended Users

- AI architects
- Engineering reviewers
- Responsible AI reviewers
- Healthcare operations stakeholders

## Intended Use

Summarize prior-authorization documentation and identify missing administrative evidence for human review.

## Limitations

- Does not make final coverage decisions.
- Does not determine medical necessity.
- Does not replace clinical or payer review.
- Uses heuristic PHI/PII redaction in the MVP; production requires managed sensitive-data inspection and validation.
- External provider calls use verified LiteLLM model IDs: `gemini-3.5-flash`, `gpt-5.5`, `claude-opus-4-8`, and `xai/grok-4.3`.
- The local ADK process receives user input, but `before_model_callback=redact_before_model` redacts the ADK model request before Gemini receives it. The MVP redacts again before egress to non-Google third-party LLM providers.
- Production use requires privacy, logging, telemetry, and BAA/contract review for the managed GCP boundary and every external provider.
- Raw PHI/PII may still exist in ADK session state, request history, or platform logs before model-request redaction. Production requires ingest-time redaction before session write and strict content-logging controls.
- MVP redaction is destructive rather than pseudonymous. It protects model inputs but can reduce referential continuity across multiple people or entities.
- MVP redaction covers text parts only. Files, images, attachments, and other inline data are out of scope until extraction/OCR and managed sensitive-data inspection are added.

## Explainability And Traceability

For the LLM MVP, explainability is provided through:

- provider attribution;
- risk classification rationale;
- privacy findings;
- policy decisions;
- HITL escalation decisions;
- a single audit trace ID that correlates the pre-router redaction, risk classification, policy decision, HITL escalation, and provider-call events for one request (propagated through ADK session state and tool context).

SHAP, LIME, and counterfactual explanations are applicable to predictive/classical ML components if added later, but they are not claimed as implemented in this LLM-only MVP.

## Human Oversight

Prior-authorization requests require human review before operational use. The agent is decision support only.
