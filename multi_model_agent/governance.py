from .audit import append_audit_event
from .escalation import assess_human_escalation
from .observability import new_trace_id, record_metric
from .policy import evaluate_provider_access
from .privacy import redact_sensitive_data
from .risk import classify_request
from .schemas import (
    GovernanceContext,
    PolicyAction,
    PolicyDecision,
    ProviderRequest,
    RiskTier,
)


class GovernanceBlockedError(Exception):
    def __init__(self, decision: PolicyDecision, trace_id: str):
        self.decision = decision
        self.trace_id = trace_id
        super().__init__(decision.reason)


def prepare_provider_request(
    *,
    provider: str,
    prompt: str,
    trace_id: str | None = None,
    use_case: str | None = None,
) -> tuple[ProviderRequest, GovernanceContext, PolicyDecision]:
    trace = trace_id or new_trace_id()
    privacy = redact_sensitive_data(prompt)
    risk = classify_request(
        prompt,
        contains_sensitive_data=privacy.contains_sensitive_data,
        use_case=use_case,
    )
    escalation = assess_human_escalation(risk=risk, privacy=privacy)
    governed_prompt = privacy.redacted_text
    decision = evaluate_provider_access(
        provider=provider,
        prompt=governed_prompt,
        risk=risk,
        privacy=privacy,
    )
    context = GovernanceContext(
        trace_id=trace,
        original_prompt=prompt,
        governed_prompt=governed_prompt,
        privacy=privacy,
        risk=risk,
        escalation=escalation,
        policy_decisions=[decision],
    )

    _record_governance_events(context, decision)

    if decision.action == PolicyAction.BLOCK:
        raise GovernanceBlockedError(decision, trace)

    request = ProviderRequest(
        trace_id=trace,
        provider=provider,
        prompt=governed_prompt,
        risk_tier=risk.risk_tier,
        data_sensitivity=risk.data_sensitivity,
        human_review_required=escalation.required,
    )
    return request, context, decision


def record_provider_success(
    *,
    trace_id: str,
    provider: str,
    tokens: int,
    risk_tier: RiskTier | None = None,
) -> None:
    append_audit_event(
        trace_id=trace_id,
        event_type="provider_call_succeeded",
        provider=provider,
        action="allow",
        risk_tier=risk_tier,
        details={"tokens": tokens},
    )
    record_metric(
        trace_id=trace_id,
        name="provider_tokens",
        value=tokens,
        provider=provider,
        risk_tier=risk_tier,
    )


def record_provider_failure(
    *,
    trace_id: str,
    provider: str,
    error: Exception,
    action: str,
    risk_tier: RiskTier | None = None,
) -> None:
    append_audit_event(
        trace_id=trace_id,
        event_type="provider_call_failed",
        provider=provider,
        action=action,
        risk_tier=risk_tier,
        details={"error": str(error)},
    )


def governance_failure_message(error: GovernanceBlockedError) -> str:
    return (
        "Request blocked by governance policy. "
        f"Trace ID: {error.trace_id}. Reason: {error.decision.reason}"
    )


def _record_governance_events(
    context: GovernanceContext, decision: PolicyDecision
) -> None:
    append_audit_event(
        trace_id=context.trace_id,
        event_type="privacy_assessment",
        action="redact" if context.privacy.contains_sensitive_data else "no_redaction",
        risk_tier=context.risk.risk_tier,
        details={
            "findings": [finding.kind for finding in context.privacy.findings],
            "contains_sensitive_data": context.privacy.contains_sensitive_data,
        },
    )
    append_audit_event(
        trace_id=context.trace_id,
        event_type="risk_classification",
        action=context.risk.risk_tier.value,
        risk_tier=context.risk.risk_tier,
        details={
            "use_case": context.risk.use_case,
            "rationale": context.risk.rationale,
            "requires_human_review": context.risk.requires_human_review,
        },
    )
    append_audit_event(
        trace_id=context.trace_id,
        event_type="policy_decision",
        provider=decision.provider,
        action=decision.action.value,
        risk_tier=context.risk.risk_tier,
        details={"reason": decision.reason},
    )
    append_audit_event(
        trace_id=context.trace_id,
        event_type="human_escalation",
        action="required" if context.escalation.required else "not_required",
        risk_tier=context.risk.risk_tier,
        details={
            "reason": context.escalation.reason,
            "target_queue": context.escalation.target_queue,
        },
    )
