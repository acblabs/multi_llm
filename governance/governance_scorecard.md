# Governance Scorecard

## Summary

- unit_tests_passed: not_observed_by_generator
- redteam_pass_rate: 1.000
- privacy_redaction_recall: 0.929
- privacy_redaction_precision: 1.000
- fairness_invariance_pass_rate: 1.000
- audit_chain_verified: True
- sample_phi_regression_guard_passed: True
- human_review_controls_passed: not_observed_by_generator
- observability_tests_passed: not_observed_by_generator
- privacy_gate_passed: True
- critical_checks_passed: True

## Audit Chain

- sample_audit_event_count: 9
- sample_audit_final_hash: e2519df966eaaf1ebfdc513891f95adb5b7d6bfaca805941f9b32ec30274869b
- sample_audit_errors: []
- report_parse_errors: []

## Sources

- redteam_report: evals/redteam/prior_auth_redteam_report.md
- privacy_report: evals/privacy/redaction_benchmark_report.md
- fairness_report: evals/fairness/invariance_report.md
- sample_audit_log: examples/audit/sample_audit_chain.jsonl
- evidence_packet_root: examples/evidence_packets

## Limitations

- This scorecard summarizes deterministic local reports and a sample audit-chain verification.
- sample_phi_regression_guard_passed is a regression check over known synthetic sample values, not a general PHI detector.
- GitHub Actions supplies unit-test, human-review-control, and observability-test pass/fail status from discrete completed steps.
- Cloud Build remains the GCP deployment and deployment-evidence path; this scorecard is repository governance evidence.
