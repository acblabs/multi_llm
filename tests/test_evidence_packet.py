import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multi_model_agent.audit import clear_audit_log, set_audit_store
from multi_model_agent.audit_store import JsonlAuditStore
from multi_model_agent.evidence_packet import (
    PACKET_FILENAMES,
    export_evidence_packet,
    packet_contains_sample_phi_values,
)
from multi_model_agent.governance import prepare_provider_request, record_provider_success
from multi_model_agent.review import REVIEW_HMAC_KEY_ENV, record_human_review_decision
from multi_model_agent.schemas import RiskTier


ROOT = Path(__file__).resolve().parents[1]
TRACE_ID = "trace-evidence-packet"
PHI_PROMPT = (
    "Patient: Jane Doe. DOB: 01/02/1960. Email jane.doe@example.com. "
    "Phone 555-123-4567. Member ID ABC123456. "
    "Summarize this prior authorization request for MRI lumbar spine. "
    "Diagnosis: lumbar radiculopathy. "
    "Clinical rationale: persistent symptoms and functional impairment. "
    "History: symptoms for 10 weeks. "
    "Prior conservative therapy: physical therapy trial. "
    "MRI report shows nerve root compression. "
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


class EvidencePacketTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.audit_log = Path(self.temp_dir.name) / "audit.jsonl"
        set_audit_store(JsonlAuditStore(self.audit_log))
        clear_audit_log()

    def _record_sample_trace(self, trace_id: str = TRACE_ID) -> None:
        prepare_provider_request(
            provider="openai",
            prompt=PHI_PROMPT,
            trace_id=trace_id,
        )
        record_provider_success(
            trace_id=trace_id,
            provider="openai",
            tokens=128,
            risk_tier=RiskTier.HIGH,
        )
        with patch.dict(os.environ, {REVIEW_HMAC_KEY_ENV: "test-secret"}, clear=False):
            record_human_review_decision(
                trace_id=trace_id,
                reviewer_id="reviewer-123",
                reviewer_role="clinical_operations",
                decision="accepted",
                rationale="Reviewed Patient Jane Doe DOB: 01/02/1960 documentation.",
            )

    def test_packet_generation_contains_expected_safe_files(self):
        self._record_sample_trace()
        output_root = Path(self.temp_dir.name) / "packets"

        result = export_evidence_packet(
            trace_id=TRACE_ID,
            audit_log=self.audit_log,
            output_root=output_root,
        )

        self.assertTrue(result.audit_chain_verification.valid)
        self.assertEqual(set(PACKET_FILENAMES), set(result.files))
        for filename in PACKET_FILENAMES:
            self.assertTrue((result.packet_dir / filename).exists(), filename)

        trace_state = json.loads((result.packet_dir / "trace_state.json").read_text())
        explanations = json.loads(
            (result.packet_dir / "governance_explanations.json").read_text()
        )
        coverage = json.loads(
            (result.packet_dir / "evidence_coverage_report.json").read_text()
        )
        redaction = json.loads((result.packet_dir / "redaction_summary.json").read_text())
        provenance = json.loads((result.packet_dir / "model_provenance.json").read_text())
        human_review = json.loads((result.packet_dir / "human_review.json").read_text())
        reviewer_summary = (result.packet_dir / "reviewer_summary.md").read_text()
        serialized_packet = "\n".join(
            path.read_text(encoding="utf-8")
            for path in result.packet_dir.iterdir()
            if path.is_file()
        )

        self.assertEqual(trace_state["latest_risk_tier"], "high")
        self.assertEqual(trace_state["latest_policy_action"], "allow")
        self.assertTrue(trace_state["human_review_completed"])
        self.assertTrue(any(item["decision_type"] == "risk_classification" for item in explanations))
        self.assertEqual(coverage["workflow_type"], "prior_authorization")
        self.assertTrue(coverage["human_review_required"])
        self.assertGreater(redaction["total_findings"], 0)
        self.assertIn("openai", provenance["providers"])
        self.assertTrue(human_review["completed"])
        self.assertEqual(human_review["final_decision"], "accepted")
        self.assertIn("Audit chain verification result: valid", reviewer_summary)
        self.assertIn("finding observations across", reviewer_summary)
        self.assertEqual(redaction["counting_strategy"], "sum_across_redaction_summary_events")
        self.assertFalse(packet_contains_sample_phi_values(result.packet_dir))
        for raw_value in RAW_VALUES:
            self.assertNotIn(raw_value, serialized_packet)

    def test_packet_export_rebuilds_directory_and_uses_canonical_trace_id(self):
        trace_id = "trace/unsafe packet id"
        self._record_sample_trace(trace_id=trace_id)
        output_root = Path(self.temp_dir.name) / "packets"

        first = export_evidence_packet(
            trace_id=trace_id,
            audit_log=self.audit_log,
            output_root=output_root,
        )
        stale_file = first.packet_dir / "stale.txt"
        stale_file.write_text("stale packet content", encoding="utf-8")

        second = export_evidence_packet(
            trace_id=trace_id,
            audit_log=self.audit_log,
            output_root=output_root,
        )

        self.assertEqual(first.packet_dir, second.packet_dir)
        self.assertTrue(second.trace_id.startswith("trace:"))
        self.assertFalse(stale_file.exists())
        self.assertFalse(packet_contains_sample_phi_values(second.packet_dir))

    def test_sample_phi_scan_ignores_binary_files(self):
        packet_dir = Path(self.temp_dir.name) / "binary-packet"
        packet_dir.mkdir()
        (packet_dir / "artifact.bin").write_bytes(b"\xff\xfe\x00\x00")

        self.assertFalse(packet_contains_sample_phi_values(packet_dir))

    def test_export_cli_writes_packet_for_trace(self):
        trace_id = "trace-evidence-packet-cli"
        self._record_sample_trace(trace_id=trace_id)
        output_root = Path(self.temp_dir.name) / "cli-packets"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/export_audit_packet.py",
                "--trace-id",
                trace_id,
                "--audit-log",
                str(self.audit_log),
                "--output-dir",
                str(output_root),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        output = json.loads(result.stdout)
        packet_dir = Path(output["packet_dir"])
        self.assertTrue((packet_dir / "reviewer_summary.md").exists())
        self.assertTrue(output["audit_chain_verification"]["valid"])
        for raw_value in RAW_VALUES:
            self.assertNotIn(raw_value, result.stdout)

    def test_export_cli_reports_missing_trace_without_traceback(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/export_audit_packet.py",
                "--trace-id",
                "missing-trace",
                "--audit-log",
                str(self.audit_log),
                "--output-dir",
                str(Path(self.temp_dir.name) / "missing-packets"),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("No audit events found", result.stderr)
        self.assertNotIn("Traceback", result.stderr + result.stdout)

    def test_empty_trace_id_is_rejected_before_packet_export(self):
        with self.assertRaisesRegex(ValueError, "trace_id is required"):
            export_evidence_packet(
                trace_id="   ",
                audit_log=self.audit_log,
                output_root=Path(self.temp_dir.name) / "empty-trace-packets",
            )

    def test_scorecard_generation_reads_local_reports(self):
        output = Path(self.temp_dir.name) / "governance_scorecard.md"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_governance_scorecard.py",
                "--output",
                str(output),
                "--evidence-packet-root",
                str(Path(self.temp_dir.name) / "empty-packets"),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        summary = json.loads(result.stdout)
        scorecard_text = output.read_text(encoding="utf-8")

        self.assertTrue(summary["critical_checks_passed"])
        self.assertTrue(summary["scorecard"]["audit_chain_verified"])
        self.assertEqual(summary["report_parse_errors"], [])
        self.assertEqual(summary["scorecard"]["redteam_pass_rate"], 1.0)
        self.assertEqual(summary["scorecard"]["fairness_invariance_pass_rate"], 1.0)
        for field in (
            "unit_tests_passed",
            "redteam_pass_rate",
            "privacy_redaction_recall",
            "privacy_redaction_precision",
            "fairness_invariance_pass_rate",
            "audit_chain_verified",
            "sample_phi_regression_guard_passed",
            "human_review_controls_passed",
            "observability_tests_passed",
        ):
            self.assertIn(field, scorecard_text)


if __name__ == "__main__":
    unittest.main()
