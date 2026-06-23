# Board Risk Report: AI Governance Snapshot

## Executive Summary

The MVP demonstrates a high-risk healthcare support workflow with controls for privacy, human oversight, auditability, red-team evaluation, redaction benchmarking, and structured prior-auth invariance regression. It is not approved for production use.

## Key Risk Indicators

| KRI | Current MVP status | Target before production |
| --- | --- | --- |
| High-risk requests without HITL | 0 expected | 0 |
| External calls with unredacted PHI/PII | 0 expected | 0 |
| Prompt-injection red-team failures | 36 deterministic cases pass in `evals/redteam/prior_auth_redteam_report.md` | 0 critical failures |
| Gated redaction recall failures | Email, formatted phone, SSN, and member ID gates pass in `evals/privacy/redaction_benchmark_report.md` | 0 gated failures |
| Known redaction limitations | Bare-name and bare-date recall reported as non-gated limitations | Managed sensitive-data inspection or improved detector before production |
| Prior-auth structured invariance failures | Synthetic structured-status invariance cases pass in `evals/fairness/invariance_report.md` | Clinically reviewed fairness plan before production |
| Audit trace coverage | Required for governed calls | 100% |
| Provider fallback without audit event | 0 expected | 0 |

## Open Decisions

- Whether external providers are permitted for PHI-adjacent workflows after redaction.
- Whether managed sensitive-data inspection is mandatory before production.
- Which GCP managed agent runtime product name and status applies at implementation time.
- Whether payer/provider legal review is required for each deployment context.

## Recommended Governance Action

Approve continued prototype work only if the MVP keeps final decisions with humans and every README claim remains backed by code, evals, or documented scope limits.
