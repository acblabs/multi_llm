from .schemas import EscalationDecision, PrivacyAssessment, RiskAssessment, RiskTier


def assess_human_escalation(
    *, risk: RiskAssessment, privacy: PrivacyAssessment
) -> EscalationDecision:
    if risk.risk_tier == RiskTier.PROHIBITED:
        return EscalationDecision(
            required=True,
            reason="Prohibited request requires governance review and cannot proceed.",
            target_queue="rai-governance-review",
        )

    if risk.requires_human_review:
        return EscalationDecision(
            required=True,
            reason=risk.rationale,
            target_queue="prior-auth-clinical-operations-review",
        )

    if privacy.contains_sensitive_data:
        return EscalationDecision(
            required=True,
            reason="Sensitive data was detected; output should be reviewed before use.",
            target_queue="privacy-review",
        )

    return EscalationDecision(required=False, reason="No escalation criteria met.")
