# Prior-Authorization Governance Walkthrough

## Example Request

```text
Patient: Jane Doe. DOB: 04/12/1975. Member ID: ABC123456.
Summarize this prior authorization request for medication coverage and identify missing documentation.
```

## Control Flow

1. `pre_router.py` redacts patient name, DOB, and member ID before the Gemini router model call.
2. `risk.py` classifies the workflow as high risk because it is prior-authorization support.
3. `privacy.py` redacts again before third-party provider egress.
4. `policy.py` allows only the redacted prompt to leave the managed GCP/ADK boundary for non-Google third-party providers.
5. `explainer.py` attaches deterministic reason codes, policy IDs, and safe reviewer-facing rationales to governance decisions.
6. `evidence_coverage.py` produces an `EvidenceCoverageReport` for prior authorization documentation.
7. `escalation.py` requires human review.
8. `tools.py` sends the redacted prompt to approved third-party providers with retry/fallback.
9. `audit.py` records privacy, risk, policy, evidence coverage, escalation, provider, and metric events.
10. The deterministic eval scripts measure the same control path without external model calls.

## Evidence Coverage Output

The evidence coverage report identifies documentation elements as `present`, `missing`, `insufficient`, or `not_applicable`:

- diagnosis or condition;
- requested service or procedure;
- clinical rationale;
- relevant history;
- prior conservative therapy;
- imaging or lab documentation;
- medication history, when relevant;
- provider notes;
- payer policy references, when supplied.

The report stores source references and hashes over redacted excerpts rather than raw source excerpts. See `evidence_coverage_sample.json` for a sanitized example.

## Regression Evidence

Run the Phase 5 evals after changing governance controls or fixtures:

```bash
python scripts/run_redteam_eval.py
python evals/privacy/run_redaction_benchmark.py
python evals/fairness/run_invariance_eval.py
```

The red-team eval exercises prior-auth prompt-injection, PHI-exfiltration, egress-policy, autonomy-boundary, fallback, fanout, and hallucinated-evidence cases. The privacy benchmark gates only email, formatted phone, SSN, and member ID recall for the current regex redactor and reports bare-name and bare-date limitations. The structured invariance regression compares evidence-coverage statuses and decision boundaries across synthetic demographic variants rather than comparing free-text rationales.

## Expected Boundary

The agent may summarize and organize facts and identify missing or insufficient documentation for human review. It must not approve or deny coverage, determine medical necessity, diagnose, recommend treatment, or bypass human review.
