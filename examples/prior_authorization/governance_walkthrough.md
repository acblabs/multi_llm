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
5. `escalation.py` requires human review.
6. `tools.py` sends the redacted prompt to approved third-party providers with retry/fallback.
7. `audit.py` records privacy, risk, policy, escalation, provider, and metric events.

## Expected Boundary

The agent may summarize and organize facts. It must not approve or deny coverage, determine medical necessity, or bypass human review.
