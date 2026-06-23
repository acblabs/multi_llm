# ADR 0001: Governed Prior-Authorization Vertical Slice

## Status

Accepted for MVP.

## Context

The original repo demonstrated multi-LLM orchestration, retry/fallback, and cost tracking. For a Responsible AI architecture portfolio, the stronger signal is a governed healthcare workflow with real control points.

## Decision

Build one prior-authorization vertical slice before expanding horizontally:

1. Classify risk.
2. Redact PHI/PII.
3. Enforce provider egress policy.
4. Preserve retry/fallback.
5. Produce structured governance explanations.
6. Produce a prior-auth evidence coverage report for human review.
7. Require HITL for high-risk workflows.
8. Emit PHI-safe audit events with integrity verification.
9. Produce deterministic red-team, privacy redaction benchmark, and prior-auth invariance reports.

## Consequences

- The repo stays focused and reviewer-friendly.
- The multi-LLM story remains intact.
- Governance controls are implemented in code instead of only described in documents.
- Evals are deterministic and fast enough for local review and CI, but they remain MVP regression evidence rather than production safety, privacy, or fairness proof.
- Broader framework mappings and GCP integrations remain expansion items until evidence exists.
