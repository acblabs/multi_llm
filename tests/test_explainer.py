import json
import unittest

from multi_model_agent.audit import (
    append_audit_event,
    clear_audit_log,
    get_audit_log,
    set_audit_store,
)
from multi_model_agent.audit_store import InMemoryAuditStore
from multi_model_agent.escalation import assess_human_escalation
from multi_model_agent.explainer import (
    explain_fallback_decision,
    explain_final_output_boundary,
    explain_human_escalation_decision,
    explain_redaction_decision,
    explain_risk_classification,
)
from multi_model_agent.governance import (
    prepare_provider_request,
    record_provider_failure,
    record_provider_success,
)
from multi_model_agent.privacy import redact_sensitive_data
from multi_model_agent.reliability import retry_with_backoff
from multi_model_agent.risk import classify_request
from multi_model_agent.schemas import (
    APPROVED_PROVIDER_NAMES,
    DataSensitivity,
    DecisionFactor,
    EscalationDecision,
    GovernanceDecisionExplanation,
    RiskAssessment,
    RiskTier,
)


PHI_PROMPT = (
    "Patient: Jane Doe. DOB: 01/02/1960. Email jane.doe@example.com. "
    "Phone 555-123-4567. Member ID ABC123456. Summarize this prior authorization."
)
RAW_PHI_VALUES = [
    "Jane Doe",
    "01/02/1960",
    "jane.doe@example.com",
    "555-123-4567",
    "ABC123456",
]


class ExplainerTests(unittest.TestCase):
    def setUp(self):
        set_audit_store(InMemoryAuditStore())
        clear_audit_log()

    def test_risk_classification_includes_reason_codes(self):
        risk = classify_request(PHI_PROMPT, contains_sensitive_data=True)

        self.assertEqual(risk.risk_tier.value, "high")
        self.assertIn("PRIOR_AUTH_WORKFLOW", risk.reason_codes)
        self.assertIn("PHI_DETECTED", risk.reason_codes)
        self.assertIn("HUMAN_REVIEW_REQUIRED", risk.reason_codes)

    def test_hitl_escalation_includes_policy_ids(self):
        privacy = redact_sensitive_data(PHI_PROMPT)
        risk = classify_request(
            PHI_PROMPT,
            contains_sensitive_data=privacy.contains_sensitive_data,
        )

        escalation = assess_human_escalation(risk=risk, privacy=privacy)

        self.assertTrue(escalation.required)
        self.assertIn("high_risk_prior_auth_requires_human_review", escalation.policy_ids)

    def test_redaction_explanation_uses_counts_not_raw_values(self):
        privacy = redact_sensitive_data(PHI_PROMPT)

        explanation = explain_redaction_decision(
            trace_id="trace-explain-redaction",
            privacy=privacy,
        )

        serialized = explanation.model_dump_json()
        self.assertIn("redaction_count", serialized)
        self.assertIn("PHI_DETECTED", serialized)
        for raw_value in RAW_PHI_VALUES:
            self.assertNotIn(raw_value, serialized)

    def test_explanations_are_serializable(self):
        _, context, _ = prepare_provider_request(
            provider="openai",
            prompt=PHI_PROMPT,
            trace_id="trace-serializable",
        )

        serialized = json.dumps(
            [
                explanation.model_dump(mode="json")
                for explanation in context.explanations
            ],
            sort_keys=True,
        )

        self.assertIn("human_review_required", serialized)
        self.assertIn("provider_egress", serialized)

    def test_explanations_are_attached_to_sanitized_audit_events(self):
        _, context, _ = prepare_provider_request(
            provider="openai",
            prompt=PHI_PROMPT,
            trace_id="trace-audit-explanations",
        )

        events = get_audit_log("trace-audit-explanations")
        explained_events = {
            event["event_type"]
            for event in events
            if "governance_explanation" in event["details"]
        }
        privacy_event = next(
            event for event in events if event["event_type"] == "privacy_assessment"
        )
        context_view = context.to_safe_dict()

        self.assertIn("privacy_assessment", explained_events)
        self.assertIn("risk_classification", explained_events)
        self.assertIn("routing_decision", explained_events)
        self.assertIn("policy_decision", explained_events)
        self.assertIn("human_escalation", explained_events)
        self.assertIn("redaction_summary", privacy_event["details"])
        self.assertNotIn("findings", privacy_event["details"])
        self.assertTrue(context_view["governance_explanations"])

        serialized = json.dumps(
            {"audit": events, "context": context_view},
            sort_keys=True,
        )
        for raw_value in RAW_PHI_VALUES:
            self.assertNotIn(raw_value, serialized)

    def test_existing_audit_callers_still_work_without_explanations(self):
        append_audit_event(
            trace_id="trace-existing-caller",
            event_type="metric",
            details={"name": "provider_tokens", "value": 7},
        )

        events = get_audit_log("trace-existing-caller")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["details"], {"name": "provider_tokens", "value": 7})

    def test_fallback_reason_codes_cover_manual_schema_objects(self):
        privacy = redact_sensitive_data("General operations request.")
        risk = RiskAssessment(
            risk_tier=RiskTier.HIGH,
            use_case="prior_authorization",
            rationale="Compatibility object without explicit reason codes.",
            data_sensitivity=DataSensitivity.INTERNAL,
            requires_human_review=True,
        )
        escalation = EscalationDecision(
            required=True,
            reason="Compatibility object without explicit reason codes.",
            target_queue="manual-review",
        )

        risk_explanation = explain_risk_classification(
            trace_id="trace-fallback-risk",
            risk=risk,
        )
        hitl_explanation = explain_human_escalation_decision(
            trace_id="trace-fallback-hitl",
            escalation=escalation,
            risk=risk,
            privacy=privacy,
        )

        self.assertIn("PRIOR_AUTH_WORKFLOW", risk_explanation.reason_codes)
        self.assertIn("HUMAN_REVIEW_REQUIRED", hitl_explanation.reason_codes)

    def test_explanation_schema_safe_view_redacts_and_supports_numeric_results(self):
        explanation = GovernanceDecisionExplanation(
            decision_id="decision with spaces and Patient Jane Doe",
            decision_type="manual_test",
            result=3,
            reason_codes=["PHI_DETECTED"],
            human_rationale=(
                "Patient: Jane Doe. DOB: 01/02/1960. "
                "Email jane.doe@example.com. Phone 555-123-4567."
            ),
            factors=[
                DecisionFactor(
                    name="unsafe factor name",
                    value="Member ID ABC123456",
                    description="Patient Jane Doe provided DOB: 01/02/1960.",
                )
            ],
            policy_ids=["manual_policy"],
            risk_tier="high",
            requires_human_review=True,
            trace_id="trace with spaces",
        )

        safe_view = explanation.to_safe_dict()
        serialized = json.dumps(safe_view, sort_keys=True)

        self.assertEqual(safe_view["result"], 3)
        self.assertTrue(
            GovernanceDecisionExplanation.model_fields["decision_id"].description
        )
        self.assertNotIn("Jane Doe", serialized)
        self.assertNotIn("01/02/1960", serialized)
        self.assertNotIn("jane.doe@example.com", serialized)
        self.assertNotIn("555-123-4567", serialized)
        self.assertNotIn("ABC123456", serialized)

    def test_provider_constants_are_single_source_for_policy_outputs(self):
        _, _, decision = prepare_provider_request(
            provider=APPROVED_PROVIDER_NAMES[0],
            prompt="Summarize a non-sensitive operations request.",
            trace_id="trace-provider-constant",
        )

        self.assertEqual(decision.allowed_providers, list(APPROVED_PROVIDER_NAMES))

    def test_final_output_and_fallback_explanations_attach_to_provider_events(self):
        record_provider_success(
            trace_id="trace-provider-boundary",
            provider="openai",
            tokens=11,
            risk_tier=RiskTier.HIGH,
        )
        record_provider_failure(
            trace_id="trace-provider-boundary",
            provider="openai",
            error=RuntimeError("overloaded while handling Patient Jane Doe"),
            action="fallback",
            risk_tier=RiskTier.HIGH,
        )

        events = get_audit_log("trace-provider-boundary")
        success = next(
            event for event in events if event["event_type"] == "provider_call_succeeded"
        )
        failure = next(
            event for event in events if event["event_type"] == "provider_call_failed"
        )
        serialized = json.dumps(events, sort_keys=True)

        self.assertEqual(
            success["details"]["governance_explanation"]["decision_type"],
            "final_output_boundary",
        )
        self.assertEqual(
            failure["details"]["governance_explanation"]["decision_type"],
            "provider_fallback",
        )
        self.assertIn("DECISION_SUPPORT_ONLY", serialized)
        self.assertIn("FALLBACK_SELECTED", serialized)
        self.assertNotIn("Jane Doe", serialized)

    def test_direct_boundary_explainers_are_deterministic_and_trace_scoped(self):
        first = explain_final_output_boundary(
            trace_id="trace-boundary-a",
            provider="openai",
            risk_tier=RiskTier.HIGH,
        )
        second = explain_final_output_boundary(
            trace_id="trace-boundary-a",
            provider="openai",
            risk_tier=RiskTier.HIGH,
        )
        other_trace = explain_final_output_boundary(
            trace_id="trace-boundary-b",
            provider="openai",
            risk_tier=RiskTier.HIGH,
        )
        fallback = explain_fallback_decision(
            trace_id="trace-boundary-a",
            provider="openai",
            action="fallback",
            risk_tier=RiskTier.HIGH,
        )

        self.assertEqual(first.decision_id, second.decision_id)
        self.assertNotEqual(first.decision_id, other_trace.decision_id)
        self.assertIn("NO_AUTONOMOUS_COVERAGE_DECISION", first.reason_codes)
        self.assertIn("FALLBACK_SELECTED", fallback.reason_codes)

    def test_retry_decision_event_has_structured_explanation(self):
        attempts = {"count": 0}

        def flaky():
            attempts["count"] += 1
            raise RuntimeError("timeout while handling Patient Jane Doe")

        with self.assertRaises(RuntimeError):
            retry_with_backoff(
                flaky,
                trace_id="trace-retry-explanation",
                provider="openai",
                max_retries=1,
                sleeper=lambda _: None,
            )

        events = get_audit_log("trace-retry-explanation")
        details = events[0]["details"]
        serialized = json.dumps(details, sort_keys=True)

        self.assertEqual(details["error_category"], "retryable_provider_error")
        self.assertEqual(
            details["governance_explanation"]["decision_type"],
            "provider_retry",
        )
        self.assertIn("RETRYABLE_PROVIDER_ERROR", serialized)
        self.assertNotIn("Jane Doe", serialized)


if __name__ == "__main__":
    unittest.main()
