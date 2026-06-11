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
5. Require HITL for high-risk workflows.
6. Emit audit events.
7. Produce a red-team eval report.

## Consequences

- The repo stays focused and reviewer-friendly.
- The multi-LLM story remains intact.
- Governance controls are implemented in code instead of only described in documents.
- Broader framework mappings and GCP integrations remain expansion items until evidence exists.
