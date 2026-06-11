# Governance Operating Model

## RACI

| Activity | AI Architecture | Engineering | Product | Privacy | Compliance | InfoSec | Legal | Operations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Risk tiering | A | R | C | C | C | C | C | C |
| PHI/PII controls | C | R | C | A | C | C | C | C |
| Provider egress policy | A | R | C | C | C | C | C | I |
| Red-team eval | A | R | C | C | C | C | I | I |
| HITL process | C | R | A | C | C | I | I | R |
| Release approval | A | R | A | A | A | A | C | C |
| Incident response | C | R | C | A | A | A | C | R |

Legend: R = Responsible, A = Accountable, C = Consulted, I = Informed.

## Operating Principles

- Architecture defines the control points.
- Engineering implements and tests the controls.
- Privacy, Compliance, Legal, and InfoSec approve production boundaries.
- Operations owns the human-review workflow once deployed.
