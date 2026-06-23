from .schemas import EscalationDecision, PrivacyAssessment, RiskAssessment, RiskTier


def assess_human_escalation(
    *, risk: RiskAssessment, privacy: PrivacyAssessment
) -> EscalationDecision:
    if risk.risk_tier == RiskTier.PROHIBITED:
        return EscalationDecision(
            required=True,
            reason="Prohibited request requires governance review and cannot proceed.",
            target_queue="rai-governance-review",
            reason_codes=["PROHIBITED_AUTONOMY_REQUEST", "HUMAN_REVIEW_REQUIRED"],
            policy_ids=["prohibited_autonomy_requires_governance_review"],
        )

    if risk.requires_human_review:
        return EscalationDecision(
            required=True,
            reason=risk.rationale,
            target_queue="prior-auth-clinical-operations-review",
            reason_codes=risk.reason_codes or ["HUMAN_REVIEW_REQUIRED"],
            policy_ids=risk.policy_ids or ["high_risk_prior_auth_requires_human_review"],
        )

    if privacy.contains_sensitive_data:
        return EscalationDecision(
            required=True,
            reason="Sensitive data was detected; output should be reviewed before use.",
            target_queue="privacy-review",
            reason_codes=["PHI_DETECTED", "HUMAN_REVIEW_REQUIRED"],
            policy_ids=["sensitive_data_requires_human_review"],
        )

    return EscalationDecision(
        required=False,
        reason="No escalation criteria met.",
        reason_codes=["NO_HITL_TRIGGER"],
        policy_ids=["minimal_risk_no_human_review_required"],
    )
