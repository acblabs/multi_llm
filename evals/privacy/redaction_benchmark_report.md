# Redaction Benchmark Report

## Summary

- total_cases: 19
- passed_cases: 17
- failed_cases: 2
- pass_rate: 0.895
- gate_passed: True
- gated_identifier_types_passed: 4
- gated_identifier_types_failed: 0
- failed_gated_identifier_types: []
- failures_by_category: {"bare_date": 1, "bare_name": 1}
- redaction_precision: 1.000
- redaction_recall: 0.929
- redaction_f1: 0.963
- false_positive_count: 0
- false_negative_count: 2

## Gated Thresholds

| Identifier Type | Recall | Threshold | Passed |
| --- | ---: | ---: | --- |
| email | 1.000 | 0.950 | True |
| member_id | 1.000 | 0.800 | True |
| phone | 1.000 | 0.900 | True |
| ssn | 1.000 | 0.900 | True |

## Metrics By Identifier Type

| Identifier Type | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bare_date | 1.000 | 0.000 | 0.000 | 0 | 0 | 1 |
| bare_name | 1.000 | 0.000 | 0.000 | 0 | 0 | 1 |
| date_of_birth | 1.000 | 1.000 | 1.000 | 2 | 0 | 0 |
| email | 1.000 | 1.000 | 1.000 | 5 | 0 | 0 |
| member_id | 1.000 | 1.000 | 1.000 | 8 | 0 | 0 |
| patient_name | 1.000 | 1.000 | 1.000 | 2 | 0 | 0 |
| phone | 1.000 | 1.000 | 1.000 | 5 | 0 | 0 |
| ssn | 1.000 | 1.000 | 1.000 | 4 | 0 | 0 |

## Limitations

- This benchmark does not claim production-grade PHI detection.
- Case pass/fail counts are entity-level exactness checks; process exit is controlled by the gated identifier-type recall thresholds above.
- Bare-name and bare-date recall are tracked but are not CI gates for this regex-only redactor.
- Gated CI thresholds cover email, formatted phone, SSN, and member ID recall only.

## Tracked Non-Gated Limitations

- bare_name_recall: 0.000
- bare_date_recall: 0.000
- overall_recall: 0.929

## Regression Notes

- Fast CI can run this benchmark without credentials, network access, or external model calls.
