from typing import Any

from google.adk.models.llm_request import LlmRequest

from .audit import append_audit_event
from .observability import ensure_trace_id, new_trace_id
from .privacy import redact_sensitive_data


def redact_before_model(context: Any, llm_request: LlmRequest):
    """Redact PHI/PII from text parts before the Gemini router model call."""
    trace_id = _trace_id_from_context(context)
    redaction_count = 0
    finding_kinds: set[str] = set()

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
                finding_kinds.update(finding.kind for finding in assessment.findings)

    append_audit_event(
        trace_id=trace_id,
        event_type="pre_router_privacy_assessment",
        action="redact" if redaction_count else "no_redaction",
        details={
            "redaction_count": redaction_count,
            "finding_kinds": sorted(finding_kinds),
            "target": "gemini_router",
        },
    )

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
