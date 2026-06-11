# FDA SaMD Boundary Reasoning

The MVP is intentionally scoped outside autonomous diagnosis, treatment, and medical-necessity determination. It summarizes provided administrative documentation for a human reviewer.

## Inside MVP Scope

- Organizing prior-authorization information.
- Redacting sensitive identifiers before external model calls.
- Identifying missing administrative evidence.
- Producing a draft summary for human review.

## Outside MVP Scope

- Diagnosis or treatment recommendations.
- Medication dosing recommendations.
- Final medical-necessity determinations.
- Automated coverage approval or denial.

## Boundary Controls

- High-risk tier for prior-authorization workflows.
- Human review required before operational use.
- Policy block for requests that ask the agent to make final coverage decisions.
- Audit trail for every governance decision.
