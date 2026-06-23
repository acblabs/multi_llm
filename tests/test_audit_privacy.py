import json
import tempfile
import unittest
from pathlib import Path

from multi_model_agent.audit import (
    append_audit_event,
    clear_audit_log,
    get_audit_log,
    set_audit_store,
)
from multi_model_agent.audit_store import (
    InMemoryAuditStore,
    JsonlAuditStore,
    sanitize_for_persistence,
)
from multi_model_agent.escalation import assess_human_escalation
from multi_model_agent.policy import evaluate_provider_access
from multi_model_agent.privacy import redact_sensitive_data, safe_privacy_assessment
from multi_model_agent.risk import classify_request
from multi_model_agent.schemas import AuditEvent, GovernanceContext


PHI_TEXT = (
    "Patient: Jane Doe. DOB: 01/02/1960. "
    "Email jane.doe@example.com. Phone 555-123-4567. "
    "Member ID ABC123456."
)
RAW_PHI_STRINGS = [
    "Jane Doe",
    "01/02/1960",
    "jane.doe@example.com",
    "555-123-4567",
    "ABC123456",
]


class AuditPrivacyTests(unittest.TestCase):
    def setUp(self):
        set_audit_store(InMemoryAuditStore())
        clear_audit_log()

    def test_persisted_jsonl_does_not_contain_raw_phi(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"
            store = JsonlAuditStore(path)
            privacy = redact_sensitive_data(PHI_TEXT)

            store.append(
                AuditEvent(
                    trace_id="trace-privacy",
                    event_type="privacy_assessment",
                    details={
                        "original_prompt": PHI_TEXT,
                        "original_text": PHI_TEXT,
                        "prompt": PHI_TEXT,
                        "full_model_response": PHI_TEXT,
                        "privacy": privacy,
                        "findings": privacy.findings,
                        "error": RuntimeError(f"Provider payload included {PHI_TEXT}"),
                        "reason": f"Patient Jane Doe submitted DOB 01/02/1960.",
                    },
                )
            )

            persisted = path.read_text(encoding="utf-8")

            for raw_value in RAW_PHI_STRINGS:
                self.assertNotIn(raw_value, persisted)
            self.assertIn("redaction_summary", persisted)
            self.assertIn("error_category", persisted)
            self.assertNotIn("original_prompt", persisted)
            self.assertNotIn("original_text", persisted)

    def test_facade_get_audit_log_returns_sanitized_events_only(self):
        privacy = redact_sensitive_data(PHI_TEXT)

        append_audit_event(
            trace_id="trace-facade-privacy",
            event_type="privacy_assessment",
            details={
                "value": PHI_TEXT,
                "findings": privacy.findings,
                "original_text": PHI_TEXT,
            },
        )

        serialized = json.dumps(get_audit_log())

        self.assertIn("privacy_assessment", serialized)
        self.assertIn("redaction_summary", serialized)
        for raw_value in RAW_PHI_STRINGS:
            self.assertNotIn(raw_value, serialized)

    def test_metric_events_keep_only_structured_numeric_values(self):
        append_audit_event(
            trace_id="trace-metric-privacy",
            event_type="metric",
            details={
                "name": "provider_tokens",
                "value": PHI_TEXT,
                "tokens": 42,
            },
        )

        events = get_audit_log("trace-metric-privacy")

        self.assertEqual(events[0]["details"], {"name": "provider_tokens", "tokens": 42})
        serialized = json.dumps(events)
        for raw_value in RAW_PHI_STRINGS:
            self.assertNotIn(raw_value, serialized)

    def test_sanitize_for_persistence_drops_raw_prompt_and_response_fields(self):
        event = AuditEvent(
            trace_id="trace-sanitize",
            event_type="provider_call_failed",
            details={
                "original_prompt": PHI_TEXT,
                "raw_response": PHI_TEXT,
                "reviewer_id": "reviewer-jane-doe",
                "error": f"Provider failed while handling {PHI_TEXT}",
                "reason_codes": ["PHI_DETECTED"],
            },
        )

        payload = sanitize_for_persistence(event)
        serialized = json.dumps(payload)

        self.assertIn("reason_codes", serialized)
        self.assertIn("error_category", serialized)
        self.assertNotIn("original_prompt", serialized)
        self.assertNotIn("raw_response", serialized)
        self.assertNotIn("reviewer-jane-doe", serialized)
        for raw_value in RAW_PHI_STRINGS:
            self.assertNotIn(raw_value, serialized)

    def test_free_text_allowlisted_gaps_are_not_persisted(self):
        adversarial_phi_values = [
            "JANE DOE",
            "jose ramirez",
            "seen by dr smith",
            "MRN 7781234",
            "Patient O'Connor",
            "Patient José Ramírez",
        ]
        event = AuditEvent(
            trace_id="trace-adversarial-phi",
            event_type="policy_decision",
            details={
                "reason": "JANE DOE and jose ramirez need review",
                "name": "Patient O'Connor",
                "value": "MRN 7781234",
                "category": "seen by dr smith",
                "replacement": "Patient José Ramírez",
                "reason_codes": ["PHI_DETECTED"],
                "policy_ids": ["high_risk_prior_auth_requires_review"],
            },
        )

        payload = sanitize_for_persistence(event)
        serialized = json.dumps(payload)

        self.assertIn("reason_codes", serialized)
        self.assertIn("policy_ids", serialized)
        details = payload["details"]
        self.assertNotIn("reason", details)
        self.assertNotIn("name", details)
        self.assertNotIn("value", details)
        self.assertNotIn("category", details)
        self.assertNotIn("replacement", details)
        for raw_value in adversarial_phi_values:
            self.assertNotIn(raw_value, serialized)

    def test_model_provenance_preserves_structured_fields_only(self):
        event = AuditEvent(
            trace_id="trace-provenance",
            event_type="provider_call_succeeded",
            details={
                "model_provenance": {
                    "provider": "openai",
                    "model": "gpt-5.5",
                    "version": "2026-06-01",
                    "snapshot_id": "snap_abc-123",
                    "request_id": "req_abc-123",
                    "region": "us-central1",
                    "unsafe_note": "Patient JANE DOE",
                    "prompt": PHI_TEXT,
                    "token_counts": {
                        "input_tokens": 12,
                        "output_tokens": 7,
                        "patient_name": "JANE DOE",
                    },
                }
            },
        )

        payload = sanitize_for_persistence(event)
        provenance = payload["details"]["model_provenance"]

        self.assertEqual(provenance["provider"], "openai")
        self.assertEqual(provenance["model"], "gpt-5.5")
        self.assertEqual(provenance["version"], "2026-06-01")
        self.assertEqual(provenance["snapshot_id"], "snap_abc-123")
        self.assertEqual(provenance["request_id"], "req_abc-123")
        self.assertEqual(provenance["region"], "us-central1")
        self.assertEqual(
            provenance["token_counts"],
            {"input_tokens": 12, "output_tokens": 7},
        )
        serialized = json.dumps(payload)
        self.assertNotIn("unsafe_note", serialized)
        self.assertNotIn("prompt", serialized)
        self.assertNotIn("JANE DOE", serialized)

    def test_privacy_safe_view_excludes_original_text_and_finding_values(self):
        privacy = redact_sensitive_data(PHI_TEXT)

        safe_view = safe_privacy_assessment(privacy)
        serialized = json.dumps(safe_view, sort_keys=True)

        self.assertIn("redaction_summary", serialized)
        self.assertNotIn("original_text", serialized)
        self.assertNotIn("redacted_text", serialized)
        for finding in privacy.findings:
            self.assertNotIn(finding.value, serialized)
        for raw_value in RAW_PHI_STRINGS:
            self.assertNotIn(raw_value, serialized)

    def test_governance_context_safe_view_excludes_prompts_and_phi_values(self):
        privacy = redact_sensitive_data(PHI_TEXT)
        risk = classify_request(PHI_TEXT, contains_sensitive_data=True)
        escalation = assess_human_escalation(risk=risk, privacy=privacy)
        policy = evaluate_provider_access(
            provider="openai",
            prompt=privacy.redacted_text,
            risk=risk,
            privacy=privacy,
        )
        context = GovernanceContext(
            trace_id="trace-context",
            original_prompt=PHI_TEXT,
            governed_prompt=privacy.redacted_text,
            privacy=privacy,
            risk=risk,
            escalation=escalation,
            policy_decisions=[policy],
        )

        safe_view = context.to_safe_dict()
        serialized = json.dumps(safe_view, sort_keys=True)

        self.assertNotIn("original_prompt", serialized)
        self.assertNotIn("governed_prompt", serialized)
        self.assertNotIn("original_text", serialized)
        for raw_value in RAW_PHI_STRINGS:
            self.assertNotIn(raw_value, serialized)


if __name__ == "__main__":
    unittest.main()
