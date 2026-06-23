import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multi_model_agent.audit import clear_audit_log, get_audit_log, set_audit_store
from multi_model_agent.audit_store import InMemoryAuditStore, JsonlAuditStore
from multi_model_agent.governance import prepare_provider_request
from multi_model_agent.review import (
    REVIEW_HMAC_KEY_ENV,
    ReviewConfigurationError,
    hmac_reviewer_id,
    record_human_review_decision,
    resolve_trace_state,
)


ROOT = Path(__file__).resolve().parents[1]
PHI_PROMPT = (
    "Patient: Jane Doe. DOB: 01/02/1960. Email jane.doe@example.com. "
    "Phone 555-123-4567. Member ID ABC123456. "
    "Summarize this prior authorization request for MRI lumbar spine. "
    "Diagnosis: radiculopathy. Clinical rationale: persistent symptoms. "
    "Provider notes: office visit note supplied."
)
RAW_VALUES = [
    "Jane Doe",
    "01/02/1960",
    "jane.doe@example.com",
    "555-123-4567",
    "ABC123456",
    "reviewer-123",
]


class HumanReviewTests(unittest.TestCase):
    def setUp(self):
        set_audit_store(InMemoryAuditStore())
        clear_audit_log()

    def test_high_risk_prior_auth_creates_review_assignment(self):
        prepare_provider_request(
            provider="openai",
            prompt=PHI_PROMPT,
            trace_id="trace-review-assignment",
        )

        events = get_audit_log("trace-review-assignment")
        assigned = next(
            event for event in events if event["event_type"] == "human_review_assigned"
        )
        serialized = json.dumps(events, sort_keys=True)

        self.assertEqual(assigned["action"], "assigned")
        self.assertTrue(assigned["details"]["human_review_required"])
        self.assertTrue(assigned["details"]["human_review_assigned"])
        self.assertEqual(
            assigned["details"]["target_queue"],
            "prior-auth-clinical-operations-review",
        )
        for raw_value in RAW_VALUES[:-1]:
            self.assertNotIn(raw_value, serialized)

    def test_review_decision_is_hmaced_and_sanitized(self):
        with patch.dict(os.environ, {REVIEW_HMAC_KEY_ENV: "test-secret"}, clear=False):
            review = record_human_review_decision(
                trace_id="trace-review-complete",
                reviewer_id="reviewer-123",
                reviewer_role="clinical_operations",
                decision="accepted",
                rationale=(
                    "Reviewed Patient Jane Doe DOB: 01/02/1960 and "
                    "Member ID ABC123456 against supplied documentation."
                ),
            )

        events = get_audit_log("trace-review-complete")
        completed = next(
            event for event in events if event["event_type"] == "human_review_completed"
        )
        serialized = json.dumps({"events": events, "review": review.to_safe_dict()})

        self.assertEqual(
            completed["details"]["reviewer_id_hmac"],
            hmac_reviewer_id("reviewer-123", key="test-secret"),
        )
        self.assertEqual(completed["details"]["review_decision"], "accepted")
        self.assertEqual(completed["details"]["review_rationale"], review.rationale)
        self.assertIn("[PATIENT_NAME]", completed["details"]["review_rationale"])
        for raw_value in RAW_VALUES:
            self.assertNotIn(raw_value, serialized)

    def test_review_decision_requires_hmac_key(self):
        with patch.dict(os.environ, {REVIEW_HMAC_KEY_ENV: ""}, clear=False):
            with self.assertRaises(ReviewConfigurationError):
                record_human_review_decision(
                    trace_id="trace-review-missing-key",
                    reviewer_id="reviewer-123",
                    decision="accepted",
                    rationale="Reviewed against supplied documentation.",
                )

        self.assertEqual(get_audit_log("trace-review-missing-key"), [])

    def test_review_decision_rejects_empty_rationale(self):
        with patch.dict(os.environ, {REVIEW_HMAC_KEY_ENV: "test-secret"}, clear=False):
            with self.assertRaises(ValueError):
                record_human_review_decision(
                    trace_id="trace-review-empty-rationale",
                    reviewer_id="reviewer-123",
                    decision="accepted",
                    rationale="   ",
                )

        self.assertEqual(get_audit_log("trace-review-empty-rationale"), [])

    def test_modified_review_records_override_event(self):
        with patch.dict(os.environ, {REVIEW_HMAC_KEY_ENV: "test-secret"}, clear=False):
            record_human_review_decision(
                trace_id="trace-review-override",
                reviewer_id="reviewer-123",
                decision="modified",
                rationale="Prior Authorization reviewed by Clinical Operations.",
            )

        events = get_audit_log("trace-review-override")
        event_types = [event["event_type"] for event in events]
        override = next(
            event for event in events if event["event_type"] == "human_override_recorded"
        )

        self.assertIn("human_review_completed", event_types)
        self.assertEqual(override["action"], "modified")
        self.assertEqual(override["details"]["review_decision"], "modified")
        self.assertIn("override_rationale", override["details"])
        self.assertEqual(
            override["details"]["override_rationale"],
            "Prior Authorization reviewed by Clinical Operations.",
        )

    def test_audit_chain_remains_valid_after_review_lifecycle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"
            set_audit_store(JsonlAuditStore(path))
            clear_audit_log()

            prepare_provider_request(
                provider="openai",
                prompt=PHI_PROMPT,
                trace_id="trace-review-chain",
            )
            with patch.dict(
                os.environ,
                {REVIEW_HMAC_KEY_ENV: "test-secret"},
                clear=False,
            ):
                record_human_review_decision(
                    trace_id="trace-review-chain",
                    reviewer_id="reviewer-123",
                    decision="accepted",
                    rationale="Reviewed against supplied documentation.",
                )

            result = JsonlAuditStore(path).verify_chain()

            self.assertTrue(result.valid, result.errors)

    def test_resolve_trace_state_returns_terminal_review_status(self):
        prepare_provider_request(
            provider="openai",
            prompt=PHI_PROMPT,
            trace_id="trace-review-state",
        )
        with patch.dict(os.environ, {REVIEW_HMAC_KEY_ENV: "test-secret"}, clear=False):
            record_human_review_decision(
                trace_id="trace-review-state",
                reviewer_id="reviewer-123",
                decision="accepted",
                rationale="Reviewed against supplied documentation.",
            )

        state = resolve_trace_state("trace-review-state")

        self.assertEqual(state.latest_risk_tier, "high")
        self.assertEqual(state.latest_policy_action, "allow")
        self.assertTrue(state.human_review_required)
        self.assertTrue(state.human_review_assigned)
        self.assertTrue(state.human_review_completed)
        self.assertEqual(state.final_human_review_decision, "accepted")
        self.assertEqual(
            state.reviewer_id_hmac,
            hmac_reviewer_id("reviewer-123", key="test-secret"),
        )
        self.assertTrue(state.audit_chain_valid)

    def test_resolve_trace_state_returns_canonical_trace_id(self):
        raw_trace_id = "trace review/state with spaces"
        prepare_provider_request(
            provider="openai",
            prompt=PHI_PROMPT,
            trace_id=raw_trace_id,
        )

        state = resolve_trace_state(raw_trace_id)

        self.assertTrue(state.trace_id.startswith("trace:"))
        self.assertNotEqual(state.trace_id, raw_trace_id)
        self.assertTrue(state.human_review_assigned)

    def test_record_human_review_cli_records_sanitized_completion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"
            set_audit_store(JsonlAuditStore(path))
            clear_audit_log()
            prepare_provider_request(
                provider="openai",
                prompt=PHI_PROMPT,
                trace_id="trace-review-cli",
            )

            env = dict(os.environ)
            env[REVIEW_HMAC_KEY_ENV] = "test-secret"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/record_human_review.py",
                    "--trace-id",
                    "trace-review-cli",
                    "--reviewer-id",
                    "reviewer-123",
                    "--decision",
                    "accepted",
                    "--rationale",
                    "Reviewed Patient Jane Doe DOB: 01/02/1960 documentation.",
                    "--audit-log",
                    str(path),
                ],
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            persisted = path.read_text(encoding="utf-8")

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            output = json.loads(result.stdout)
            self.assertTrue(output["trace_state"]["human_review_completed"])
            self.assertEqual(
                output["human_review_decision"]["reviewer_id_hmac"],
                hmac_reviewer_id("reviewer-123", key="test-secret"),
            )
            self.assertTrue(JsonlAuditStore(path).verify_chain().valid)
            for raw_value in RAW_VALUES:
                self.assertNotIn(raw_value, persisted)
                self.assertNotIn(raw_value, result.stdout)


if __name__ == "__main__":
    unittest.main()
