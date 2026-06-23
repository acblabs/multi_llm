# Governance Operating Model

## RACI

| Activity | AI Architecture | Engineering | Product | Privacy | Compliance | InfoSec | Legal | Operations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Risk tiering | A | R | C | C | C | C | C | C |
| PHI/PII controls | C | R | C | A | C | C | C | C |
| Provider egress policy | A | R | C | C | C | C | C | I |
| Red-team eval | A | R | C | C | C | C | I | I |
| Privacy redaction benchmark | C | R | C | A | C | C | I | I |
| Prior-auth structured invariance regression | A | R | C | C | C | C | I | C |
| HITL process | C | R | A | C | C | I | I | R |
| Evidence packet generation and retention | C | R | C | C | A | C | C | R |
| Governance scorecard review | A | R | C | C | C | C | I | I |
| Telemetry and audit storage controls | C | R | I | C | C | A | C | C |
| Release approval | A | R | A | A | A | A | C | C |
| Incident response | C | R | C | A | A | A | C | R |

Legend: R = Responsible, A = Accountable, C = Consulted, I = Informed.

## Operating Principles

- Architecture defines the control points.
- Engineering implements and tests the controls.
- Deterministic evals are release evidence for the MVP control plane; they must document limitations and avoid production-grade claims.
- Reviewer evidence packets are support artifacts derived from sanitized audit events; they are not a substitute for access control, retention, legal hold, or incident-scrub procedures.
- Observability is operational evidence. Durable audit evidence is the sanitized audit chain plus exported evidence packets.
- Privacy, Compliance, Legal, and InfoSec approve production boundaries.
- Operations owns the human-review workflow once deployed.
