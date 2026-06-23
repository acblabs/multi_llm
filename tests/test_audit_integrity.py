import json
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import multi_model_agent.audit as audit_module
from multi_model_agent.audit import (
    append_audit_event,
    clear_audit_log,
    get_audit_log,
    reset_audit_store,
    set_audit_store,
)
from multi_model_agent.audit_hashing import canonical_json, compute_payload_hash
from multi_model_agent.audit_store import InMemoryAuditStore, JsonlAuditStore
from multi_model_agent.schemas import AuditEvent, RiskTier


ROOT = Path(__file__).resolve().parents[1]


class AuditIntegrityTests(unittest.TestCase):
    def setUp(self):
        set_audit_store(InMemoryAuditStore())
        clear_audit_log()

    def test_existing_audit_facade_returns_sanitized_compatibility_view(self):
        append_audit_event(
            trace_id="trace-facade",
            event_type="risk_classification",
            action="high",
            risk_tier=RiskTier.HIGH,
            details={"use_case": "prior_authorization", "requires_human_review": True},
        )

        events = get_audit_log("trace-facade")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["trace_id"], "trace-facade")
        self.assertEqual(events[0]["event_type"], "risk_classification")
        self.assertEqual(events[0]["risk_tier"], "high")
        self.assertEqual(events[0]["details"]["use_case"], "prior_authorization")
        self.assertIn("event_hash", events[0])

    def test_clear_audit_log_isolates_memory_store_tests(self):
        append_audit_event(trace_id="trace-one", event_type="metric")
        self.assertEqual(len(get_audit_log()), 1)

        clear_audit_log()

        self.assertEqual(get_audit_log(), [])

    def test_default_audit_store_uses_memory_under_unittest(self):
        with patch.dict(
            "os.environ",
            {
                "MULTI_LLM_AUDIT_LOG_PATH": "",
                "MULTI_LLM_AUDIT_STORE": "",
            },
            clear=False,
        ):
            reset_audit_store()

        self.assertIsInstance(audit_module._AUDIT_STORE, InMemoryAuditStore)

    def test_appending_one_jsonl_event_creates_valid_chain(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"
            store = JsonlAuditStore(path)

            stored = store.append(
                AuditEvent(
                    trace_id="trace-one",
                    event_type="provider_call_succeeded",
                    provider="openai",
                    action="allow",
                    details={"tokens": 12},
                )
            )

            result = store.verify_chain()
            self.assertTrue(result.valid, result.errors)
            self.assertEqual(result.event_count, 1)
            self.assertIsNone(stored.previous_hash)
            self.assertEqual(result.final_hash, stored.event_hash)

    def test_appending_multiple_events_links_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonlAuditStore(Path(temp_dir) / "audit.jsonl")

            first = store.append(AuditEvent(trace_id="trace-link", event_type="privacy"))
            second = store.append(AuditEvent(trace_id="trace-link", event_type="risk"))
            third = store.append(AuditEvent(trace_id="trace-link", event_type="policy"))

            self.assertIsNone(first.previous_hash)
            self.assertEqual(second.previous_hash, first.event_hash)
            self.assertEqual(third.previous_hash, second.event_hash)
            self.assertTrue(store.verify_chain().valid)

    def test_query_by_trace_id_returns_only_matching_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonlAuditStore(Path(temp_dir) / "audit.jsonl")
            store.append(AuditEvent(trace_id="trace-a", event_type="privacy"))
            store.append(AuditEvent(trace_id="trace-b", event_type="privacy"))
            store.append(AuditEvent(trace_id="trace-a", event_type="policy"))

            events = store.query_by_trace_id("trace-a")

            self.assertEqual([event.trace_id for event in events], ["trace-a", "trace-a"])

    def test_modifying_payload_causes_verification_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"
            store = JsonlAuditStore(path)
            store.append(
                AuditEvent(
                    trace_id="trace-tamper",
                    event_type="provider_call_succeeded",
                    details={"tokens": 12},
                )
            )

            line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            line["payload"]["details"]["tokens"] = 99
            path.write_text(json.dumps(line) + "\n", encoding="utf-8")

            result = store.verify_chain()
            self.assertFalse(result.valid)
            self.assertTrue(any("payload_hash mismatch" in error for error in result.errors))

    def test_reordering_events_causes_verification_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"
            store = JsonlAuditStore(path)
            for event_type in ("privacy", "risk", "policy"):
                store.append(AuditEvent(trace_id="trace-reorder", event_type=event_type))

            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join([lines[1], lines[0], lines[2]]) + "\n", encoding="utf-8")

            result = store.verify_chain()
            self.assertFalse(result.valid)
            self.assertTrue(any("previous_hash mismatch" in error for error in result.errors))

    def test_deleting_middle_event_causes_verification_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"
            store = JsonlAuditStore(path)
            for event_type in ("privacy", "risk", "policy"):
                store.append(AuditEvent(trace_id="trace-delete", event_type=event_type))

            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")

            result = store.verify_chain()
            self.assertFalse(result.valid)
            self.assertTrue(any("previous_hash mismatch" in error for error in result.errors))

    def test_hashing_occurs_over_canonical_sanitized_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"
            store = JsonlAuditStore(path)
            store.append(
                AuditEvent(
                    trace_id="trace-hash",
                    event_type="metric",
                    details={
                        "name": "provider_tokens",
                        "value": "Patient Jane Doe 01/02/1960 jane.doe@example.com",
                    },
                )
            )

            stored = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

            self.assertNotIn("Jane Doe", json.dumps(stored))
            self.assertNotIn("01/02/1960", json.dumps(stored))
            self.assertNotIn("jane.doe@example.com", json.dumps(stored))
            self.assertEqual(stored["payload_hash"], compute_payload_hash(stored["payload"]))

    def test_datetime_and_enum_canonicalization_is_deterministic(self):
        timestamp = datetime(2026, 1, 1, 12, 34, 56, 123456, tzinfo=timezone.utc)

        serialized = canonical_json({"timestamp": timestamp, "risk_tier": RiskTier.HIGH})

        self.assertEqual(
            serialized,
            '{"risk_tier":"high","timestamp":"2026-01-01T12:34:56.123456Z"}',
        )

    def test_invalid_timestamp_string_defaults_to_safe_utc_timestamp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"
            store = JsonlAuditStore(path)

            store.append(
                {
                    "timestamp": "Patient JANE DOE visited dr smith",
                    "trace_id": "trace-timestamp",
                    "event_type": "privacy",
                    "details": {},
                }
            )

            stored = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

            self.assertRegex(
                stored["timestamp"],
                r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$",
            )
            self.assertNotIn("JANE DOE", json.dumps(stored))
            self.assertNotIn("dr smith", json.dumps(stored))

    def test_concurrent_jsonl_appends_keep_valid_chain(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"

            def append_event(index: int) -> None:
                JsonlAuditStore(path).append(
                    AuditEvent(
                        trace_id="trace-concurrent",
                        event_type="metric",
                        details={"name": "provider_tokens", "value": index},
                    )
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(append_event, range(25)))

            result = JsonlAuditStore(path).verify_chain()

            self.assertTrue(result.valid, result.errors)
            self.assertEqual(result.event_count, 25)

    def test_verify_audit_chain_script_succeeds_for_valid_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"
            JsonlAuditStore(path).append(
                AuditEvent(trace_id="trace-cli", event_type="privacy")
            )

            result = subprocess.run(
                [sys.executable, "scripts/verify_audit_chain.py", str(path)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn('"valid": true', result.stdout)
            output = json.loads(result.stdout)
            self.assertNotIn("\\", output["path"])


if __name__ == "__main__":
    unittest.main()
