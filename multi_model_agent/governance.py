from .audit import append_audit_event
from .escalation import assess_human_escalation
from .evidence_coverage import generate_evidence_coverage_report
from .explainer import (
    collect_governance_explanations,
    explain_fallback_decision,
    explain_final_output_boundary,
)
from .observability import new_trace_id, record_metric
from .policy import evaluate_provider_access
from .privacy import redact_sensitive_data
from .review import record_human_review_assignment
from .risk import classify_request
from .schemas import (
    GovernanceContext,
    GovernanceDecisionExplanation,
    EvidenceCoverageReport,
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
    explanations = collect_governance_explanations(
        trace_id=trace,
        privacy=privacy,
        risk=risk,
        policy_decision=decision,
        escalation=escalation,
    )
    evidence_coverage_report = _build_evidence_coverage_report(
        trace_id=trace,
        governed_prompt=governed_prompt,
        workflow_type=risk.use_case,
        human_review_required=escalation.required,
    )
    context = GovernanceContext(
        trace_id=trace,
        original_prompt=prompt,
        governed_prompt=governed_prompt,
        privacy=privacy,
        risk=risk,
        escalation=escalation,
        policy_decisions=[decision],
        explanations=explanations,
        evidence_coverage_report=evidence_coverage_report,
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
    explanation = explain_final_output_boundary(
        trace_id=trace_id,
        provider=provider,
        risk_tier=risk_tier,
    )
    append_audit_event(
        trace_id=trace_id,
        event_type="provider_call_succeeded",
        provider=provider,
        action="allow",
        risk_tier=risk_tier,
        details={
            "tokens": tokens,
            "governance_explanation": explanation.to_safe_dict(),
        },
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
    explanation = explain_fallback_decision(
        trace_id=trace_id,
        provider=provider,
        action=action,
        risk_tier=risk_tier,
    )
    append_audit_event(
        trace_id=trace_id,
        event_type="provider_call_failed",
        provider=provider,
        action=action,
        risk_tier=risk_tier,
        details={
            "error": str(error),
            "governance_explanation": explanation.to_safe_dict(),
        },
    )


def governance_failure_message(error: GovernanceBlockedError) -> str:
    return (
        "Request blocked by governance policy. "
        f"Trace ID: {error.trace_id}. Reason: {error.decision.reason}"
    )


def _record_governance_events(
    context: GovernanceContext, decision: PolicyDecision
) -> None:
    explanations = {
        explanation.decision_type: explanation for explanation in context.explanations
    }
    redaction_explanation = explanations.get("redaction_decision")
    risk_explanation = explanations.get("risk_classification")
    routing_explanation = explanations.get("provider_routing")
    egress_explanation = explanations.get("provider_egress")
    hitl_explanation = explanations.get("human_review_required")
    safe_redaction_explanation = _safe_explanation_dict(redaction_explanation)
    safe_risk_explanation = _safe_explanation_dict(risk_explanation)
    safe_routing_explanation = _safe_explanation_dict(routing_explanation)
    safe_egress_explanation = _safe_explanation_dict(egress_explanation)
    safe_hitl_explanation = _safe_explanation_dict(hitl_explanation)

    # Keep top-level reason codes and policy IDs for simple audit queries; the
    # nested explanation carries the reviewer-readable rationale and factors.
    append_audit_event(
        trace_id=context.trace_id,
        event_type="privacy_assessment",
        action="redact" if context.privacy.contains_sensitive_data else "no_redaction",
        risk_tier=context.risk.risk_tier,
        details={
            "redaction_summary": context.privacy.redaction_summary(),
            "contains_sensitive_data": context.privacy.contains_sensitive_data,
            "reason_codes": redaction_explanation.reason_codes
            if redaction_explanation
            else [],
            "policy_ids": redaction_explanation.policy_ids
            if redaction_explanation
            else [],
            "governance_explanation": safe_redaction_explanation,
        },
    )
    append_audit_event(
        trace_id=context.trace_id,
        event_type="risk_classification",
        action=context.risk.risk_tier.value,
        risk_tier=context.risk.risk_tier,
        details={
            "use_case": context.risk.use_case,
            "reason_codes": context.risk.reason_codes,
            "policy_ids": context.risk.policy_ids,
            "requires_human_review": context.risk.requires_human_review,
            "governance_explanation": safe_risk_explanation,
        },
    )
    append_audit_event(
        trace_id=context.trace_id,
        event_type="routing_decision",
        provider=decision.provider,
        action="selected",
        risk_tier=context.risk.risk_tier,
        details={
            "allowed_providers": decision.allowed_providers,
            "reason_codes": routing_explanation.reason_codes
            if routing_explanation
            else [],
            "policy_ids": routing_explanation.policy_ids
            if routing_explanation
            else [],
            "governance_explanation": safe_routing_explanation,
        },
    )
    append_audit_event(
        trace_id=context.trace_id,
        event_type="policy_decision",
        provider=decision.provider,
        action=decision.action.value,
        risk_tier=context.risk.risk_tier,
        details={
            "reason_codes": decision.reason_codes,
            "policy_ids": decision.policy_ids,
            "governance_explanation": safe_egress_explanation,
        },
    )
    append_audit_event(
        trace_id=context.trace_id,
        event_type="human_escalation",
        action="required" if context.escalation.required else "not_required",
        risk_tier=context.risk.risk_tier,
        details={
            "target_queue": context.escalation.target_queue,
            "reason_codes": context.escalation.reason_codes,
            "policy_ids": context.escalation.policy_ids,
            "governance_explanation": safe_hitl_explanation,
        },
    )
    if context.escalation.required:
        record_human_review_assignment(
            trace_id=context.trace_id,
            target_queue=context.escalation.target_queue,
            risk_tier=context.risk.risk_tier,
            reason_codes=context.escalation.reason_codes,
            policy_ids=context.escalation.policy_ids,
        )

    if context.evidence_coverage_report is not None:
        append_audit_event(
            trace_id=context.trace_id,
            event_type="evidence_coverage_report",
            action="generated",
            risk_tier=context.risk.risk_tier,
            details={
                "evidence_coverage_report": (
                    context.evidence_coverage_report.to_safe_dict()
                ),
                "human_review_required": context.escalation.required,
                "policy_ids": ["prior_auth_evidence_coverage_support_boundary"],
                "reason_codes": [
                    "PRIOR_AUTH_EVIDENCE_COVERAGE_GENERATED",
                    "DECISION_SUPPORT_ONLY",
                ],
            },
        )


def _safe_explanation_dict(
    explanation: GovernanceDecisionExplanation | None,
) -> dict[str, object] | None:
    return explanation.to_safe_dict() if explanation is not None else None


def _build_evidence_coverage_report(
    *,
    trace_id: str,
    governed_prompt: str,
    workflow_type: str,
    human_review_required: bool,
) -> EvidenceCoverageReport | None:
    if workflow_type != "prior_authorization":
        return None
    return generate_evidence_coverage_report(
        trace_id=trace_id,
        text=governed_prompt,
        workflow_type=workflow_type,
        human_review_required=human_review_required,
    )
