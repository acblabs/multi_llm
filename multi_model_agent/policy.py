from .schemas import PolicyAction, PolicyDecision, PrivacyAssessment, RiskAssessment, RiskTier


APPROVED_PROVIDERS = ("openai", "claude", "grok")


def evaluate_provider_access(
    *,
    provider: str,
    prompt: str,
    risk: RiskAssessment,
    privacy: PrivacyAssessment,
) -> PolicyDecision:
    blocked: list[str] = []

    if provider not in APPROVED_PROVIDERS:
        blocked.append(f"Provider '{provider}' is not in the approved provider set.")

    if risk.risk_tier == RiskTier.PROHIBITED:
        blocked.append("Prohibited autonomy request; external provider call blocked.")

    if privacy.contains_sensitive_data and prompt != privacy.redacted_text:
        blocked.append("Sensitive data was detected and must be redacted before egress.")

    if blocked:
        return PolicyDecision(
            provider=provider,
            action=PolicyAction.BLOCK,
            reason="; ".join(blocked),
            allowed_providers=list(APPROVED_PROVIDERS),
            blocked_reasons=blocked,
        )

    return PolicyDecision(
        provider=provider,
        action=PolicyAction.ALLOW,
        reason=(
            "Provider call allowed after risk classification, privacy redaction, "
            "and approved-provider validation."
        ),
        allowed_providers=list(APPROVED_PROVIDERS),
    )
