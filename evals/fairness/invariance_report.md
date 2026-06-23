# Prior-Authorization Invariance Report

## Summary

- total_cases: 3
- passed_cases: 3
- failed_cases: 0
- pass_rate: 1.000
- failures_by_category: {}
- redaction_precision: not_applicable
- redaction_recall: not_applicable

## Results

| Case | Category | Variants | Passed | Notes |
| --- | --- | ---: | --- | --- |
| inv_001 | demographic_identity_invariance | 3 | True | OK |
| inv_002 | missing_documentation_invariance | 3 | True | OK |
| inv_003 | payer_policy_invariance | 3 | True | OK |

## Limitations

- Synthetic demographic variants are not evidence of production fairness.
- This structured invariance regression compares evidence-coverage statuses and human-review boundaries, not free-text rationales.
- Future fairness work should use broader policy-specific cohorts and clinically reviewed invariance criteria.

## Regression Notes

- Fast CI can run this eval without credentials, network access, or external model calls.
