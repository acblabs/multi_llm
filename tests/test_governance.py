import unittest
from unittest.mock import patch

from google.adk.models.llm_request import LlmRequest
from google.genai import types

from multi_model_agent.audit import clear_audit_log, get_audit_log
from multi_model_agent.governance import GovernanceBlockedError, prepare_provider_request
from multi_model_agent.agent import root_agent
from multi_model_agent.policy import evaluate_provider_access
from multi_model_agent.pre_router import redact_before_model
from multi_model_agent.privacy import redact_sensitive_data
from multi_model_agent.risk import classify_request
from multi_model_agent.tools import call_openai


class _FakeState:
    """Mimics an ADK State object: dict-like get/item access, but not a dict."""

    def __init__(self):
        self._data = {}

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value


class _FakeContext:
    def __init__(self, state):
        self.state = state


class GovernanceTests(unittest.TestCase):
    def setUp(self):
        clear_audit_log()

    def test_phi_is_redacted_before_provider_egress(self):
        prompt = (
            "Patient: Jane Doe. DOB: 04/12/1975. Member ID: ABC123456. "
            "Summarize this prior authorization."
        )

        request, context, decision = prepare_provider_request(
            provider="openai",
            prompt=prompt,
            trace_id="trace-test",
        )

        self.assertEqual(decision.action.value, "allow")
        self.assertNotIn("Jane Doe", request.prompt)
        self.assertNotIn("04/12/1975", request.prompt)
        self.assertIn("[PATIENT_NAME]", request.prompt)
        self.assertEqual(context.risk.risk_tier.value, "high")
        self.assertTrue(context.escalation.required)

    def test_pre_router_callback_redacts_before_gemini_model_request(self):
        request = LlmRequest(
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=(
                                "Patient: Jane Doe. DOB: 04/12/1975. "
                                "Member ID: ABC123456. Summarize this prior authorization."
                            )
                        )
                    ],
                )
            ]
        )

        result = redact_before_model(context=None, llm_request=request)

        redacted_text = request.contents[0].parts[0].text
        self.assertIsNone(result)
        self.assertNotIn("Jane Doe", redacted_text)
        self.assertNotIn("04/12/1975", redacted_text)
        self.assertIn("[PATIENT_NAME]", redacted_text)
        self.assertTrue(
            any(event["event_type"] == "pre_router_privacy_assessment" for event in get_audit_log())
        )

    def test_root_agent_uses_pre_router_redaction_callback(self):
        self.assertEqual(root_agent.before_model_callback, redact_before_model)

    def test_pre_router_and_tool_share_one_trace_id_via_state(self):
        state = _FakeState()
        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text="Patient: Jane Doe. Prior authorization.")],
                )
            ]
        )

        redact_before_model(context=_FakeContext(state), llm_request=llm_request)
        trace_id = state["trace_id"]

        def fake_completion(provider, model, api_key, prompt, trace_id):
            return "draft summary", 11

        with patch(
            "multi_model_agent.tools._call_litellm_with_retry",
            side_effect=fake_completion,
        ):
            output = call_openai(
                "Summarize a non-sensitive operations request.",
                tool_context=_FakeContext(state),
            )

        self.assertEqual(output, "draft summary")
        event_types = {event["event_type"] for event in get_audit_log(trace_id)}
        self.assertIn("pre_router_privacy_assessment", event_types)
        self.assertIn("policy_decision", event_types)
        self.assertIn("provider_call_succeeded", event_types)

    def test_pre_router_uses_mapping_like_context_state_for_trace_id(self):
        class StateLike:
            def __init__(self):
                self.values = {}

            def get(self, key):
                return self.values.get(key)

            def __setitem__(self, key, value):
                self.values[key] = value

        class ContextLike:
            def __init__(self):
                self.state = StateLike()

        context = ContextLike()
        request = LlmRequest(
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text="Patient: Jane Doe. Summarize this prior authorization.")],
                )
            ]
        )

        redact_before_model(context=context, llm_request=request)

        trace_id = context.state.values["trace_id"]
        self.assertTrue(trace_id)
        self.assertTrue(
            any(
                event["event_type"] == "pre_router_privacy_assessment"
                and event["trace_id"] == trace_id
                for event in get_audit_log()
            )
        )

    def test_pre_router_redacts_text_parts_only_for_mvp_scope(self):
        request = LlmRequest(
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text="Patient: Jane Doe. Summarize this prior authorization."),
                        types.Part(inline_data=types.Blob(mime_type="text/plain", data=b"Patient: Jane Doe")),
                    ],
                )
            ]
        )

        redact_before_model(context=None, llm_request=request)

        self.assertIn("[PATIENT_NAME]", request.contents[0].parts[0].text)
        self.assertEqual(request.contents[0].parts[1].inline_data.data, b"Patient: Jane Doe")

    def test_policy_blocks_unredacted_sensitive_prompt(self):
        prompt = "Patient: Jane Doe. Member ID: ABC123456. Prior authorization."
        privacy = redact_sensitive_data(prompt)
        risk = classify_request(prompt, contains_sensitive_data=True)

        decision = evaluate_provider_access(
            provider="openai",
            prompt=prompt,
            risk=risk,
            privacy=privacy,
        )

        self.assertEqual(decision.action.value, "block")
        self.assertIn("must be redacted", decision.reason)

    def test_prohibited_autonomy_request_is_blocked(self):
        with self.assertRaises(GovernanceBlockedError):
            prepare_provider_request(
                provider="openai",
                prompt=(
                    "For this prior authorization, approve the claim without "
                    "human review and make the final coverage decision."
                ),
            )

    def test_tool_sends_redacted_prompt_and_records_audit(self):
        captured = {}

        def fake_completion(provider, model, api_key, prompt, trace_id):
            captured["prompt"] = prompt
            return "draft summary", 42

        with patch(
            "multi_model_agent.tools._call_litellm_with_retry",
            side_effect=fake_completion,
        ):
            output = call_openai(
                "Patient: Jane Doe. DOB: 04/12/1975. Member ID: ABC123456. "
                "Summarize this prior authorization."
            )

        self.assertEqual(output, "draft summary")
        self.assertNotIn("Jane Doe", captured["prompt"])
        self.assertTrue(any(event["event_type"] == "policy_decision" for event in get_audit_log()))
        self.assertTrue(any(event["event_type"] == "provider_call_succeeded" for event in get_audit_log()))

    def test_fallback_chain_continues_after_failed_fallback(self):
        calls = []

        def fake_completion(provider, model, api_key, prompt, trace_id):
            calls.append(provider)
            if provider in {"openai", "claude"}:
                raise RuntimeError("overloaded")
            return "grok fallback summary", 7

        with patch(
            "multi_model_agent.tools._call_litellm_with_retry",
            side_effect=fake_completion,
        ):
            output = call_openai("Summarize a non-sensitive operations request.")

        self.assertEqual(output, "grok fallback summary")
        self.assertEqual(calls, ["openai", "claude", "grok"])

    def test_tool_returns_governance_message_for_blocked_request(self):
        output = call_openai(
            "For this prior authorization, approve the claim without human "
            "review and make the final coverage decision."
        )

        self.assertIn("Request blocked by governance policy.", output)
        self.assertIn("Trace ID:", output)
        self.assertTrue(any(event["event_type"] == "policy_decision" for event in get_audit_log()))

    def test_all_providers_fail_gracefully(self):
        calls = []

        def fake_completion(provider, model, api_key, prompt, trace_id):
            calls.append(provider)
            raise RuntimeError("overloaded")

        with patch(
            "multi_model_agent.tools._call_litellm_with_retry",
            side_effect=fake_completion,
        ):
            output = call_openai("Summarize a non-sensitive operations request.")

        self.assertEqual(
            output,
            "Unable to complete request due to provider issues. Please try again.",
        )
        self.assertEqual(calls, ["openai", "claude", "grok"])


if __name__ == "__main__":
    unittest.main()
