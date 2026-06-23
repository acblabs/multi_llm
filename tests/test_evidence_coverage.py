import json
import unittest

from multi_model_agent.audit import clear_audit_log, get_audit_log, set_audit_store
from multi_model_agent.audit_store import InMemoryAuditStore, safe_audit_details
from multi_model_agent.evidence_coverage import (
    assert_evidence_report_boundary,
    find_prohibited_decision_language,
    generate_evidence_coverage_report,
    validate_evidence_report_boundaries,
)
from multi_model_agent.governance import prepare_provider_request
from multi_model_agent.schemas import EvidenceCoverageItem, SourceReference


PHI_VALUES = [
    "Jane Doe",
    "01/02/1960",
    "jane.doe@example.com",
    "555-123-4567",
    "ABC123456",
]


def _item_by_id(report, requirement_id: str) -> EvidenceCoverageItem:
    return next(item for item in report.items if item.requirement_id == requirement_id)


class EvidenceCoverageTests(unittest.TestCase):
    def setUp(self):
        set_audit_store(InMemoryAuditStore())
        clear_audit_log()

    def test_present_missing_and_insufficient_documentation_are_distinguished(self):
        text = (
            "Patient: Jane Doe. DOB: 01/02/1960. Member ID ABC123456. "
            "Prior authorization request for MRI lumbar spine. "
            "Diagnosis: lumbar radiculopathy. "
            "Clinical rationale: persistent symptoms and functional impairment. "
            "History: symptoms for 10 weeks with previous flare. "
            "Prior conservative therapy: not documented. "
            "Provider notes: office visit note supplied."
        )

        report = generate_evidence_coverage_report(
            trace_id="trace-coverage-status",
            text=text,
        )

        self.assertEqual(_item_by_id(report, "diagnosis_condition").status, "present")
        self.assertEqual(_item_by_id(report, "requested_service").status, "present")
        self.assertEqual(_item_by_id(report, "clinical_rationale").status, "present")
        self.assertEqual(_item_by_id(report, "relevant_history").status, "present")
        self.assertEqual(
            _item_by_id(report, "prior_conservative_therapy").status,
            "insufficient",
        )
        self.assertEqual(
            _item_by_id(report, "imaging_or_lab_documentation").status,
            "missing",
        )
        self.assertEqual(_item_by_id(report, "provider_notes").status, "present")
        self.assertEqual(
            _item_by_id(report, "payer_policy_references").status,
            "not_applicable",
        )

    def test_requested_imaging_does_not_count_as_imaging_documentation(self):
        report = generate_evidence_coverage_report(
            trace_id="trace-requested-imaging",
            text=(
                "Prior authorization request for MRI lumbar spine. "
                "Diagnosis: lumbar radiculopathy. Provider notes: office visit note."
            ),
        )

        self.assertEqual(_item_by_id(report, "requested_service").status, "present")
        self.assertEqual(
            _item_by_id(report, "imaging_or_lab_documentation").status,
            "missing",
        )

    def test_result_oriented_imaging_and_lab_text_counts_as_documentation(self):
        report = generate_evidence_coverage_report(
            trace_id="trace-imaging-results",
            text=(
                "Prior authorization request for MRI lumbar spine. "
                "Diagnosis: lumbar radiculopathy. MRI report shows nerve root compression. "
                "Lab results: A1c 7.2. Provider notes: office visit note."
            ),
        )

        self.assertEqual(
            _item_by_id(report, "imaging_or_lab_documentation").status,
            "present",
        )

    def test_requested_medication_is_relevant_but_not_medication_history(self):
        report = generate_evidence_coverage_report(
            trace_id="trace-medication-request",
            text=(
                "Prior authorization request for medication coverage. "
                "Diagnosis: inflammatory arthritis. Provider notes: office visit note."
            ),
        )

        self.assertEqual(_item_by_id(report, "requested_service").status, "present")
        self.assertEqual(_item_by_id(report, "medication_history").status, "missing")

    def test_missing_documentation_is_flagged_for_sparse_packets(self):
        report = generate_evidence_coverage_report(
            trace_id="trace-coverage-missing",
            text="Prior authorization packet received for review.",
        )

        self.assertEqual(_item_by_id(report, "diagnosis_condition").status, "missing")
        self.assertEqual(_item_by_id(report, "requested_service").status, "missing")
        self.assertEqual(_item_by_id(report, "provider_notes").status, "missing")
        self.assertIn("missing", report.overall_summary)

    def test_source_references_are_available_without_raw_excerpts(self):
        report = generate_evidence_coverage_report(
            trace_id="trace-source-ref",
            text=(
                "Prior authorization request for medication coverage. "
                "Diagnosis: inflammatory arthritis. Current medications: NSAID trial. "
                "Provider notes: clinical note supplied."
            ),
            source_locations={
                "diagnosis_condition": SourceReference(
                    document_name="clinical_notes.pdf",
                    page=2,
                    section="assessment",
                    locator="line-14",
                )
            },
        )

        item = _item_by_id(report, "diagnosis_condition")
        safe = item.to_safe_dict()

        self.assertEqual(item.status, "present")
        self.assertIsNotNone(item.source_reference)
        self.assertEqual(safe["source_reference"]["document_name"], "clinical_notes.pdf")
        self.assertIn("source_excerpt_hash", safe)
        self.assertNotIn("source_excerpt", safe)

    def test_report_safe_views_and_audit_sanitizer_exclude_raw_phi(self):
        report = generate_evidence_coverage_report(
            trace_id="trace-phi-safe",
            text=(
                "Patient: Jane Doe. DOB: 01/02/1960. Email jane.doe@example.com. "
                "Phone 555-123-4567. Member ID ABC123456. "
                "Prior authorization request for CT abdomen. Diagnosis: abdominal pain. "
                "Clinical rationale: persistent symptoms. Provider notes: office visit."
            ),
            source_locations={
                "requested_service": {
                    "document_name": "Jane Doe CT request.pdf",
                    "page": 1,
                    "section": "Patient Jane Doe order",
                }
            },
        )

        serialized_report = json.dumps(report.to_safe_dict(), sort_keys=True)
        serialized_audit_details = json.dumps(
            safe_audit_details({"evidence_coverage_report": report}),
            sort_keys=True,
        )

        self.assertIn("evidence_coverage_report", serialized_audit_details)
        self.assertIn("source_excerpt_hash", serialized_report)
        self.assertNotIn('"source_excerpt":', serialized_report)
        for raw_value in PHI_VALUES:
            self.assertNotIn(raw_value, serialized_report)
            self.assertNotIn(raw_value, serialized_audit_details)

    def test_prohibited_language_validation_flags_decision_terms(self):
        report = generate_evidence_coverage_report(
            trace_id="trace-boundary-validation",
            text="Prior authorization request for MRI. Diagnosis: radiculopathy.",
        )
        unsafe = report.model_copy(
            update={"overall_summary": "This request is approved."}
        )

        self.assertIn(
            "coverage_approval_language",
            find_prohibited_decision_language("This request is approved."),
        )
        self.assertTrue(validate_evidence_report_boundaries(unsafe))
        with self.assertRaises(ValueError):
            assert_evidence_report_boundary(
                report,
                reviewer_summary="The service is medically necessary.",
            )

        safe_boundary_sentence = "This report does not make an approval or denial."
        self.assertEqual(find_prohibited_decision_language(safe_boundary_sentence), [])

    def test_prior_auth_governance_path_records_evidence_coverage_report(self):
        _, context, _ = prepare_provider_request(
            provider="openai",
            prompt=(
                "Patient: Jane Doe. DOB: 01/02/1960. Member ID ABC123456. "
                "Summarize this prior authorization request for medication coverage. "
                "Diagnosis: rheumatoid arthritis. Clinical rationale: persistent symptoms. "
                "Current medications: methotrexate 15 mg weekly. "
                "Provider notes: office visit note supplied."
            ),
            trace_id="trace-governance-evidence",
        )

        events = get_audit_log("trace-governance-evidence")
        report_event = next(
            event for event in events if event["event_type"] == "evidence_coverage_report"
        )
        report = report_event["details"]["evidence_coverage_report"]
        serialized = json.dumps({"context": context.to_safe_dict(), "audit": events})

        self.assertIsNotNone(context.evidence_coverage_report)
        self.assertEqual(report["workflow_type"], "prior_authorization")
        self.assertTrue(report["human_review_required"])
        self.assertIn("prohibited_decision_boundary", report)
        self.assertTrue(any(item["status"] == "present" for item in report["items"]))
        for raw_value in PHI_VALUES:
            self.assertNotIn(raw_value, serialized)

    def test_non_prior_auth_request_does_not_generate_report(self):
        _, context, _ = prepare_provider_request(
            provider="openai",
            prompt="Summarize this non-sensitive operations request.",
            trace_id="trace-no-evidence-report",
        )

        self.assertIsNone(context.evidence_coverage_report)
        self.assertFalse(
            any(
                event["event_type"] == "evidence_coverage_report"
                for event in get_audit_log("trace-no-evidence-report")
            )
        )


if __name__ == "__main__":
    unittest.main()
