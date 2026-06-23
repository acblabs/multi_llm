from collections.abc import Iterable
from typing import Any

from .audit_hashing import sha256_canonical
from .schemas import (
    APPROVED_PROVIDER_NAMES,
    DecisionFactor,
    EscalationDecision,
    GovernanceDecisionExplanation,
    PolicyDecision,
    PrivacyAssessment,
    RiskAssessment,
    RiskTier,
)


def collect_governance_explanations(
    *,
    trace_id: str,
    privacy: PrivacyAssessment,
    risk: RiskAssessment,
    policy_decision: PolicyDecision,
    escalation: EscalationDecision,
) -> list[GovernanceDecisionExplanation]:
    return [
        explain_redaction_decision(trace_id=trace_id, privacy=privacy),
        explain_risk_classification(trace_id=trace_id, risk=risk),
        explain_routing_decision(
            trace_id=trace_id,
            provider=policy_decision.provider,
            risk=risk,
            allowed_providers=policy_decision.allowed_providers,
        ),
        explain_provider_egress_decision(
            trace_id=trace_id,
            decision=policy_decision,
            risk=risk,
            privacy=privacy,
        ),
        explain_human_escalation_decision(
            trace_id=trace_id,
            escalation=escalation,
            risk=risk,
            privacy=privacy,
        ),
    ]


def explain_redaction_decision(
    *,
    trace_id: str | None,
    privacy: PrivacyAssessment,
    target: str = "provider_prompt",
) -> GovernanceDecisionExplanation:
    summary = privacy.redaction_summary()
    finding_counts = summary["finding_counts_by_kind"]
    return _build_redaction_explanation(
        trace_id=trace_id,
        redaction_count=int(summary["total_findings"]),
        finding_kinds=sorted(finding_counts),
        contains_sensitive_data=bool(summary["contains_sensitive_data"]),
        target=target,
    )


def explain_risk_classification(
    *,
    trace_id: str | None,
    risk: RiskAssessment,
) -> GovernanceDecisionExplanation:
    return _build_explanation(
        trace_id=trace_id,
        decision_type="risk_classification",
        result=risk.risk_tier.value,
        reason_codes=risk.reason_codes or _fallback_risk_reason_codes(risk),
        human_rationale=risk.rationale,
        factors=[
            _factor("use_case", risk.use_case),
            _factor("risk_tier", risk.risk_tier.value),
            _factor("data_sensitivity", risk.data_sensitivity.value),
            _factor("requires_human_review", risk.requires_human_review),
        ],
        policy_ids=risk.policy_ids,
        risk_tier=risk.risk_tier.value,
        requires_human_review=risk.requires_human_review,
    )


def explain_routing_decision(
    *,
    trace_id: str | None,
    provider: str,
    risk: RiskAssessment,
    allowed_providers: Iterable[str],
) -> GovernanceDecisionExplanation:
    approved = provider in APPROVED_PROVIDER_NAMES
    reason_codes = (
        ["APPROVED_PROVIDER_CANDIDATE"] if approved else ["UNAPPROVED_PROVIDER"]
    )
    if risk.risk_tier in {RiskTier.HIGH, RiskTier.PROHIBITED}:
        reason_codes.append("HIGH_RISK_ROUTING_REVIEW")

    return _build_explanation(
        trace_id=trace_id,
        decision_type="provider_routing",
        result=_safe_provider_label(provider),
        reason_codes=reason_codes,
        human_rationale=(
            "The requested provider route is evaluated against approved-provider "
            "and risk controls before egress."
        ),
        factors=[
            _factor("provider", _safe_provider_label(provider)),
            _factor("approved_provider", approved),
            _factor("allowed_provider_count", len(list(allowed_providers))),
            _factor("risk_tier", risk.risk_tier.value),
        ],
        policy_ids=["approved_provider_set", "model_selection_within_governance_boundary"],
        risk_tier=risk.risk_tier.value,
        requires_human_review=risk.requires_human_review,
    )


def explain_provider_egress_decision(
    *,
    trace_id: str | None,
    decision: PolicyDecision,
    risk: RiskAssessment,
    privacy: PrivacyAssessment,
) -> GovernanceDecisionExplanation:
    allowed = decision.action.value == "allow"
    rationale = (
        "Provider egress is allowed only after approved-provider, risk, and redaction controls pass."
        if allowed
        else "Provider egress is blocked because one or more governance controls did not pass."
    )

    return _build_explanation(
        trace_id=trace_id,
        decision_type="provider_egress",
        result=decision.action.value,
        reason_codes=decision.reason_codes or ["PROVIDER_EGRESS_ALLOWED"],
        human_rationale=rationale,
        factors=[
            _factor("provider", _safe_provider_label(decision.provider)),
            _factor("action", decision.action.value),
            _factor("risk_tier", risk.risk_tier.value),
            _factor("sensitive_data_detected", privacy.contains_sensitive_data),
        ],
        policy_ids=decision.policy_ids,
        risk_tier=risk.risk_tier.value,
        requires_human_review=risk.requires_human_review,
    )


def explain_human_escalation_decision(
    *,
    trace_id: str | None,
    escalation: EscalationDecision,
    risk: RiskAssessment,
    privacy: PrivacyAssessment,
) -> GovernanceDecisionExplanation:
    rationale = (
        "Human review is required for this governance context before operational use."
        if escalation.required
        else "No configured escalation criterion was met for this governance context."
    )

    return _build_explanation(
        trace_id=trace_id,
        decision_type="human_review_required",
        result=escalation.required,
        reason_codes=escalation.reason_codes
        or (["HUMAN_REVIEW_REQUIRED"] if escalation.required else ["NO_HITL_TRIGGER"]),
        human_rationale=rationale,
        factors=[
            _factor("risk_tier", risk.risk_tier.value),
            _factor("risk_requires_review", risk.requires_human_review),
            _factor("sensitive_data_detected", privacy.contains_sensitive_data),
            _factor("target_queue", escalation.target_queue or "none"),
        ],
        policy_ids=escalation.policy_ids,
        risk_tier=risk.risk_tier.value,
        requires_human_review=escalation.required,
    )


def explain_retry_decision(
    *,
    trace_id: str | None,
    provider: str,
    action: str,
    attempt: int,
    max_retries: int,
) -> GovernanceDecisionExplanation:
    reason_codes_by_action = {
        "retry": ["RETRYABLE_PROVIDER_ERROR", "RETRY_BUDGET_AVAILABLE"],
        "fallback": ["FALLBACK_PROVIDER_ERROR", "RETRY_NOT_SELECTED"],
        "fail": ["NON_RETRYABLE_PROVIDER_ERROR", "RETRY_NOT_ALLOWED"],
    }
    return _build_explanation(
        trace_id=trace_id,
        decision_type="provider_retry",
        result=action,
        reason_codes=reason_codes_by_action.get(action, ["PROVIDER_ERROR_CLASSIFIED"]),
        human_rationale=(
            "Provider errors are classified into retry, fallback, or fail actions "
            "without storing raw provider error text."
        ),
        factors=[
            _factor("provider", _safe_provider_label(provider)),
            _factor("attempt", max(int(attempt), 0)),
            _factor("max_retries", max(int(max_retries), 0)),
            _factor("error_action", action),
        ],
        policy_ids=["provider_retry_budget", "safe_provider_failure_handling"],
    )


def explain_fallback_decision(
    *,
    trace_id: str | None,
    provider: str,
    action: str,
    risk_tier: RiskTier | None,
) -> GovernanceDecisionExplanation:
    uses_fallback = action == "fallback"
    return _build_explanation(
        trace_id=trace_id,
        decision_type="provider_fallback",
        result=action,
        reason_codes=(
            ["FALLBACK_SELECTED", "PROVIDER_ERROR_CLASSIFIED"]
            if uses_fallback
            else ["FALLBACK_NOT_SELECTED", "PROVIDER_ERROR_CLASSIFIED"]
        ),
        human_rationale=(
            "Fallback routing is selected only for provider failure classes that allow it."
            if uses_fallback
            else "Fallback routing is not selected for this provider failure class."
        ),
        factors=[
            _factor("provider", _safe_provider_label(provider)),
            _factor("failure_action", action),
            _factor("risk_tier", risk_tier.value if risk_tier else "unknown"),
        ],
        policy_ids=["safe_provider_failure_handling", "fallback_with_governance_recheck"],
        risk_tier=risk_tier.value if risk_tier else None,
        requires_human_review=risk_tier in {RiskTier.HIGH, RiskTier.PROHIBITED}
        if risk_tier
        else None,
    )


def explain_final_output_boundary(
    *,
    trace_id: str | None,
    provider: str,
    risk_tier: RiskTier | None,
) -> GovernanceDecisionExplanation:
    requires_review = (
        risk_tier in {RiskTier.HIGH, RiskTier.PROHIBITED} if risk_tier else None
    )
    reason_codes = ["DECISION_SUPPORT_ONLY", "NO_AUTONOMOUS_COVERAGE_DECISION"]
    if requires_review:
        reason_codes.append("HUMAN_REVIEW_REQUIRED")

    return _build_explanation(
        trace_id=trace_id,
        decision_type="final_output_boundary",
        result="decision_support_only",
        reason_codes=reason_codes,
        human_rationale=(
            "Model output remains decision support and must not be treated as an "
            "approval, denial, diagnosis, treatment recommendation, or medical-necessity decision."
        ),
        factors=[
            _factor("provider", _safe_provider_label(provider)),
            _factor("risk_tier", risk_tier.value if risk_tier else "unknown"),
            _factor("requires_human_review", requires_review),
        ],
        policy_ids=["prior_auth_decision_support_boundary"],
        risk_tier=risk_tier.value if risk_tier else None,
        requires_human_review=requires_review,
    )


def _build_explanation(
    *,
    trace_id: str | None,
    decision_type: str,
    result: str | int | float | bool | None,
    reason_codes: Iterable[str],
    human_rationale: str,
    factors: Iterable[DecisionFactor],
    policy_ids: Iterable[str],
    risk_tier: str | None = None,
    requires_human_review: bool | None = None,
) -> GovernanceDecisionExplanation:
    factor_list = list(factors)
    code_list = _unique_strings(reason_codes)
    policy_list = _unique_strings(policy_ids)
    payload: dict[str, Any] = {
        "decision_type": decision_type,
        "result": result,
        "reason_codes": code_list,
        "human_rationale": human_rationale,
        "factors": [factor.model_dump(mode="json") for factor in factor_list],
        "policy_ids": policy_list,
        "risk_tier": risk_tier,
        "requires_human_review": requires_human_review,
        "trace_id": trace_id,
    }
    # This is a per-trace decision instance ID, not a stable rule ID or a
    # production-scale uniqueness guarantee.
    decision_id = "gd_" + sha256_canonical(payload)[:24]

    return GovernanceDecisionExplanation(decision_id=decision_id, **payload)


def _factor(
    name: str,
    value: str | int | float | bool | None,
    *,
    weight: float | None = None,
    description: str | None = None,
) -> DecisionFactor:
    return DecisionFactor(
        name=name,
        value=value,
        weight=weight,
        description=description,
    )


def _build_redaction_explanation(
    *,
    trace_id: str | None,
    redaction_count: int,
    finding_kinds: Iterable[str],
    contains_sensitive_data: bool,
    target: str,
) -> GovernanceDecisionExplanation:
    count = max(int(redaction_count), 0)
    safe_kinds = ",".join(sorted({str(kind) for kind in finding_kinds})) or "none"
    return _build_explanation(
        trace_id=trace_id,
        decision_type="redaction_decision",
        result=contains_sensitive_data,
        reason_codes=_redaction_reason_codes(contains_sensitive_data),
        human_rationale=_redaction_rationale(contains_sensitive_data, target),
        factors=[
            _factor(
                "redaction_count",
                count,
                description="Count of detected sensitive identifier spans.",
            ),
            _factor(
                "finding_kinds",
                safe_kinds,
                description="Identifier classes detected, without raw values.",
            ),
            _factor("target", target),
        ],
        policy_ids=["phi_pii_redaction_before_model_or_provider_egress"],
    )


def _redaction_reason_codes(contains_sensitive_data: bool) -> list[str]:
    if contains_sensitive_data:
        return ["PHI_DETECTED", "REDACTION_APPLIED"]
    return ["NO_SENSITIVE_DATA_DETECTED", "REDACTION_NOT_REQUIRED"]


def _redaction_rationale(contains_sensitive_data: bool, target: str) -> str:
    if contains_sensitive_data and target == "gemini_router":
        return "Sensitive identifiers were redacted from the router request."
    if contains_sensitive_data:
        return "Sensitive identifiers were detected and replaced before model or provider egress."
    if target == "gemini_router":
        return "No supported sensitive identifier pattern was detected in the router request."
    return "No supported sensitive identifier pattern was detected for this text segment."


def _fallback_risk_reason_codes(risk: RiskAssessment) -> list[str]:
    # Risk objects normally come from classify_request(), which fills reason codes.
    # Keep these defaults for manually constructed or deserialized compatibility objects.
    if risk.risk_tier == RiskTier.PROHIBITED:
        return ["PROHIBITED_AUTONOMY_REQUEST", "HUMAN_REVIEW_REQUIRED"]
    if risk.risk_tier == RiskTier.HIGH:
        return ["PRIOR_AUTH_WORKFLOW", "HIGH_RISK_HEALTHCARE_ACCESS_CONTEXT"]
    if risk.risk_tier == RiskTier.LIMITED:
        return ["HEALTHCARE_CONTENT"]
    return ["NO_HEALTHCARE_ACCESS_SIGNAL"]


def _safe_provider_label(provider: str) -> str:
    return provider if provider in APPROVED_PROVIDER_NAMES else "unapproved_provider"


def _unique_strings(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        safe_value = str(value)
        if safe_value and safe_value not in seen:
            result.append(safe_value)
            seen.add(safe_value)
    return result
