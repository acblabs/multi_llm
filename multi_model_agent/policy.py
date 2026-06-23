from .schemas import (
    APPROVED_PROVIDER_NAMES,
    PolicyAction,
    PolicyDecision,
    PrivacyAssessment,
    RiskAssessment,
    RiskTier,
)


def evaluate_provider_access(
    *,
    provider: str,
    prompt: str,
    risk: RiskAssessment,
    privacy: PrivacyAssessment,
) -> PolicyDecision:
    blocked: list[str] = []
    reason_codes: list[str] = []
    policy_ids = [
        "approved_provider_egress_only",
        "phi_pii_redaction_before_provider_egress",
        "block_prohibited_autonomy_egress",
    ]

    if provider not in APPROVED_PROVIDER_NAMES:
        blocked.append(f"Provider '{provider}' is not in the approved provider set.")
        reason_codes.append("UNAPPROVED_PROVIDER")

    if risk.risk_tier == RiskTier.PROHIBITED:
        blocked.append("Prohibited autonomy request; external provider call blocked.")
        reason_codes.append("PROHIBITED_RISK_TIER")

    if privacy.contains_sensitive_data and prompt != privacy.redacted_text:
        blocked.append("Sensitive data was detected and must be redacted before egress.")
        reason_codes.append("UNREDACTED_SENSITIVE_DATA")

    if blocked:
        return PolicyDecision(
            provider=provider,
            action=PolicyAction.BLOCK,
            reason="; ".join(blocked),
            allowed_providers=list(APPROVED_PROVIDER_NAMES),
            blocked_reasons=blocked,
            reason_codes=reason_codes,
            policy_ids=policy_ids,
        )

    return PolicyDecision(
        provider=provider,
        action=PolicyAction.ALLOW,
        reason=(
            "Provider call allowed after risk classification, privacy redaction, "
            "and approved-provider validation."
        ),
        allowed_providers=list(APPROVED_PROVIDER_NAMES),
        reason_codes=[
            "APPROVED_PROVIDER",
            "PROMPT_REDACTED_OR_SAFE",
            "RISK_POLICY_PERMITS_EGRESS",
        ],
        policy_ids=policy_ids,
    )
