import builtins
import json
import sys
import types
import unittest
from unittest.mock import patch

from multi_model_agent.audit import clear_audit_log, get_audit_log, set_audit_store
from multi_model_agent.audit_store import InMemoryAuditStore
from multi_model_agent.governance import prepare_provider_request
from multi_model_agent.observability import record_metric
from multi_model_agent.schemas import RiskTier
from multi_model_agent.telemetry import (
    METRIC_ESTIMATED_COST_USD,
    METRIC_HUMAN_ESCALATIONS_TOTAL,
    METRIC_PROVIDER_LATENCY_MS,
    METRIC_REQUEST_COUNT,
    SPAN_OUTPUT_SCHEMA_VALIDATION,
    SPAN_ESCALATION_HITL,
    SPAN_POLICY_EGRESS_CHECK,
    SPAN_REQUEST,
    SPAN_RISK_CLASSIFICATION,
    clear_recorded_telemetry,
    configure_telemetry,
    get_recorded_metrics,
    get_recorded_spans,
    governance_span,
    record_governance_metric,
    sanitize_telemetry_attributes,
)
from multi_model_agent.tools import _call_litellm_with_retry, call_openai


PHI_PROMPT = (
    "Patient: Jane Doe. DOB: 01/02/1960. Email jane.doe@example.com. "
    "Phone 555-123-4567. Member ID ABC123456. "
    "Summarize this prior authorization request."
)
RAW_VALUES = [
    "Jane Doe",
    "01/02/1960",
    "jane.doe@example.com",
    "555-123-4567",
    "ABC123456",
]


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        set_audit_store(InMemoryAuditStore())
        clear_audit_log()
        configure_telemetry(
            enabled=False,
            capture_spans=False,
            capture_metrics=False,
            use_otel=False,
        )
        clear_recorded_telemetry()

    def tearDown(self):
        configure_telemetry(
            enabled=False,
            capture_spans=False,
            capture_metrics=False,
            use_otel=False,
        )
        clear_recorded_telemetry()
        clear_audit_log()

    def test_noop_mode_records_nothing_and_requires_no_collector(self):
        with governance_span(
            SPAN_REQUEST,
            {
                "governance.trace_id": "trace-noop",
                "governance.provider": "openai",
            },
        ) as span:
            span.set_attribute("governance.risk_tier", "high")

        record_governance_metric(
            METRIC_REQUEST_COUNT,
            1,
            {"governance.trace_id": "trace-noop"},
        )

        self.assertEqual(get_recorded_spans(), [])
        self.assertEqual(get_recorded_metrics(), [])

    def test_spans_and_metrics_can_be_captured_without_otel_exporter(self):
        configure_telemetry(
            enabled=True,
            capture_spans=True,
            capture_metrics=True,
            use_otel=False,
        )

        with governance_span(
            SPAN_REQUEST,
            {
                "governance.trace_id": "trace-capture",
                "governance.provider": "openai",
                "gen_ai.request.model": "gpt-5.5",
            },
        ) as span:
            span.set_attribute("governance.risk_tier", "high")

        record_governance_metric(
            METRIC_REQUEST_COUNT,
            1,
            {"governance.trace_id": "trace-capture", "governance.provider": "openai"},
        )

        spans = get_recorded_spans()
        metrics = get_recorded_metrics()

        self.assertEqual(spans[0]["name"], SPAN_REQUEST)
        self.assertEqual(spans[0]["attributes"]["governance.trace_id"], "trace-capture")
        self.assertEqual(spans[0]["attributes"]["governance.risk_tier"], "high")
        self.assertEqual(metrics[0]["name"], METRIC_REQUEST_COUNT)

    def test_attribute_sanitizer_allowlists_keys_and_redacts_phi(self):
        safe = sanitize_telemetry_attributes(
            {
                "governance.trace_id": "trace-phi",
                "governance.escalation_reason": (
                    "Patient Jane Doe DOB: 01/02/1960 with Member ID ABC123456."
                ),
                "governance.policy_ids": [
                    "high_risk_prior_auth_requires_human_review",
                    "Patient Jane Doe",
                ],
                "prompt": PHI_PROMPT,
                "governance.raw_prompt": PHI_PROMPT,
            }
        )
        serialized = json.dumps(safe, sort_keys=True)

        self.assertIn("governance.trace_id", safe)
        self.assertIn("governance.escalation_reason", safe)
        self.assertNotIn("prompt", safe)
        self.assertNotIn("governance.raw_prompt", safe)
        for raw_value in RAW_VALUES:
            self.assertNotIn(raw_value, serialized)

    def test_attribute_sanitizer_orders_set_values_deterministically(self):
        first = sanitize_telemetry_attributes(
            {"governance.policy_ids": {"z_policy", "a_policy", "m_policy"}}
        )
        second = sanitize_telemetry_attributes(
            {"governance.policy_ids": {"m_policy", "z_policy", "a_policy"}}
        )

        self.assertEqual(
            first["governance.policy_ids"],
            ["a_policy", "m_policy", "z_policy"],
        )
        self.assertEqual(first, second)

    def test_governance_path_emits_safe_control_plane_spans(self):
        configure_telemetry(
            enabled=True,
            capture_spans=True,
            capture_metrics=True,
            use_otel=False,
        )

        prepare_provider_request(
            provider="openai",
            prompt=PHI_PROMPT,
            trace_id="trace-governance-telemetry",
        )

        spans = get_recorded_spans()
        span_names = {span["name"] for span in spans}
        metrics = get_recorded_metrics()
        metric_names = {metric["name"] for metric in metrics}
        serialized = json.dumps({"spans": spans, "metrics": metrics}, sort_keys=True)

        self.assertIn(SPAN_RISK_CLASSIFICATION, span_names)
        self.assertIn(SPAN_POLICY_EGRESS_CHECK, span_names)
        self.assertIn(SPAN_ESCALATION_HITL, span_names)
        self.assertIn(METRIC_REQUEST_COUNT, metric_names)
        self.assertIn(METRIC_HUMAN_ESCALATIONS_TOTAL, metric_names)
        self.assertNotIn("human_escalation_rate", metric_names)
        self.assertNotIn("provider_error_rate", metric_names)
        self.assertTrue(
            any(
                span["attributes"].get("governance.risk_tier") == "high"
                for span in spans
            )
        )
        for raw_value in RAW_VALUES:
            self.assertNotIn(raw_value, serialized)

    def test_full_tool_path_emits_safe_spans_and_derived_rate_primitives(self):
        configure_telemetry(
            enabled=True,
            capture_spans=True,
            capture_metrics=True,
            use_otel=False,
        )
        fake_litellm = types.SimpleNamespace(
            completion=lambda **kwargs: {
                "choices": [{"message": {"content": "provider summary"}}],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                    "total_tokens": 7,
                },
            }
        )

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            output = call_openai(PHI_PROMPT, fallback_allowed=False)

        spans = get_recorded_spans()
        span_names = {span["name"] for span in spans}
        metrics = get_recorded_metrics()
        metric_names = {metric["name"] for metric in metrics}
        serialized = json.dumps({"spans": spans, "metrics": metrics}, sort_keys=True)
        output_span = next(
            span for span in spans if span["name"] == SPAN_OUTPUT_SCHEMA_VALIDATION
        )

        self.assertEqual(output, "provider summary")
        self.assertIn(SPAN_REQUEST, span_names)
        self.assertIn("provider.openai.call", span_names)
        self.assertIn(SPAN_OUTPUT_SCHEMA_VALIDATION, span_names)
        self.assertIn(METRIC_REQUEST_COUNT, metric_names)
        self.assertIn(METRIC_ESTIMATED_COST_USD, metric_names)
        self.assertNotIn("provider_error_rate", metric_names)
        self.assertNotIn("human_escalation_rate", metric_names)
        self.assertEqual(
            output_span["attributes"]["governance.estimated_cost_usd"],
            7 * 0.00001,
        )
        for raw_value in RAW_VALUES:
            self.assertNotIn(raw_value, serialized)

    def test_provider_call_span_records_model_usage_and_latency_without_prompt(self):
        configure_telemetry(
            enabled=True,
            capture_spans=True,
            capture_metrics=True,
            use_otel=False,
        )

        fake_litellm = types.SimpleNamespace(
            completion=lambda **kwargs: {
                "choices": [{"message": {"content": "provider summary"}}],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                    "total_tokens": 7,
                },
            }
        )
        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            content, tokens = _call_litellm_with_retry(
                "openai",
                "gpt-test",
                None,
                PHI_PROMPT,
                "trace-provider-telemetry",
            )

        spans = get_recorded_spans()
        metrics = get_recorded_metrics()
        serialized = json.dumps({"spans": spans, "metrics": metrics}, sort_keys=True)
        provider_span = next(
            span for span in spans if span["name"] == "provider.openai.call"
        )

        self.assertEqual(content, "provider summary")
        self.assertEqual(tokens, 7)
        self.assertEqual(provider_span["attributes"]["gen_ai.request.model"], "gpt-test")
        self.assertEqual(provider_span["attributes"]["gen_ai.usage.input_tokens"], 3)
        self.assertEqual(provider_span["attributes"]["gen_ai.usage.output_tokens"], 4)
        self.assertIn(METRIC_PROVIDER_LATENCY_MS, {metric["name"] for metric in metrics})
        for raw_value in RAW_VALUES:
            self.assertNotIn(raw_value, serialized)

    def test_enabled_telemetry_degrades_when_otel_imports_are_absent(self):
        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name.startswith("opentelemetry"):
                raise ImportError("simulated missing optional dependency")
            return real_import(name, *args, **kwargs)

        configure_telemetry(
            enabled=True,
            capture_spans=True,
            capture_metrics=True,
            use_otel=True,
        )
        with patch("builtins.__import__", side_effect=blocked_import):
            with governance_span(
                SPAN_REQUEST,
                {"governance.trace_id": "trace-missing-otel"},
            ):
                pass

        self.assertEqual(get_recorded_spans()[0]["name"], SPAN_REQUEST)

    def test_metric_events_persisted_to_audit_are_sanitized(self):
        record_metric(
            trace_id="trace-telemetry-audit",
            name="provider_latency_ms",
            value="Patient Jane Doe DOB: 01/02/1960",
            provider="openai",
            risk_tier=RiskTier.HIGH,
        )

        events = get_audit_log("trace-telemetry-audit")
        serialized = json.dumps(events, sort_keys=True)

        self.assertEqual(events[0]["event_type"], "metric")
        self.assertEqual(events[0]["details"], {"name": "provider_latency_ms"})
        for raw_value in RAW_VALUES:
            self.assertNotIn(raw_value, serialized)


if __name__ == "__main__":
    unittest.main()
