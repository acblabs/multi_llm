from typing import Any

from google.adk.models.llm_request import LlmRequest

from .audit import append_audit_event
from .config import MODEL_CONFIG
from .explainer import explain_redaction_decision
from .observability import ensure_trace_id, new_trace_id
from .privacy import redact_sensitive_data
from .schemas import PrivacyAssessment, PrivacyFinding
from .telemetry import (
    METRIC_REDACTION_COUNT_DISTRIBUTION,
    SPAN_PRE_ROUTER_REDACTION,
    SPAN_SYNTHESIS_GEMINI,
    governance_span,
    record_governance_metric,
)


def redact_before_model(callback_context: Any, llm_request: LlmRequest):
    """Redact PHI/PII from text parts before the Gemini router model call.

    Parameter names match the ADK before_model_callback contract: ADK invokes
    this as ``callback(callback_context=..., llm_request=...)``.
    """
    trace_id = _trace_id_from_context(callback_context)
    with governance_span(
        SPAN_PRE_ROUTER_REDACTION,
        {
            "governance.trace_id": trace_id,
            "governance.provider": "gemini",
            "gen_ai.system": "google",
            "gen_ai.request.model": MODEL_CONFIG.get("gemini"),
        },
    ) as span:
        redaction_count = 0
        finding_kinds: set[str] = set()
        safe_findings: list[PrivacyFinding] = []

        for content in llm_request.contents:
            if not getattr(content, "parts", None):
                continue

            for part in content.parts:
                text = getattr(part, "text", None)
                if not text:
                    continue

                assessment = redact_sensitive_data(text)
                if assessment.redacted_text != text:
                    part.text = assessment.redacted_text
                    redaction_count += len(assessment.findings)
                    finding_kinds.update(
                        finding.kind for finding in assessment.findings
                    )
                    safe_findings.extend(
                        PrivacyFinding(
                            kind=finding.kind,
                            value="",
                            replacement=finding.replacement,
                        )
                        for finding in assessment.findings
                    )
        span.set_attribute("governance.redaction_count", redaction_count)

    privacy_summary = PrivacyAssessment(
        original_text="",
        redacted_text="",
        findings=safe_findings,
        contains_sensitive_data=redaction_count > 0,
    )
    explanation = explain_redaction_decision(
        trace_id=trace_id,
        privacy=privacy_summary,
        target="gemini_router",
    )

    append_audit_event(
        trace_id=trace_id,
        event_type="pre_router_privacy_assessment",
        action="redact" if redaction_count else "no_redaction",
        details={
            "redaction_count": redaction_count,
            "finding_kinds": sorted(finding_kinds),
            "target": "gemini_router",
            "governance_explanation": explanation.to_safe_dict(),
        },
    )
    record_governance_metric(
        METRIC_REDACTION_COUNT_DISTRIBUTION,
        redaction_count,
        {
            "governance.trace_id": trace_id,
            "governance.provider": "gemini",
            "gen_ai.request.model": MODEL_CONFIG.get("gemini"),
        },
    )
    with governance_span(
        SPAN_SYNTHESIS_GEMINI,
        {
            "governance.trace_id": trace_id,
            "governance.provider": "gemini",
            "gen_ai.system": "google",
            "gen_ai.request.model": MODEL_CONFIG.get("gemini"),
            "governance.telemetry_boundary": "adk_managed_call",
        },
    ):
        # The ADK-owned Gemini call happens after this callback returns; this
        # span marks the safe handoff boundary controlled by this repo.
        pass

    return None


def _trace_id_from_context(context: Any) -> str:
    state = getattr(context, "state", None)
    if state is not None:
        return ensure_trace_id(state)

    session = getattr(context, "session", None)
    session_id = getattr(session, "id", None)
    if session_id:
        return str(session_id)

    return new_trace_id()
