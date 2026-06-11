from .schemas import DataSensitivity, RiskAssessment, RiskTier


PRIOR_AUTH_TERMS = (
    "prior authorization",
    "preauthorization",
    "coverage determination",
    "medical necessity",
    "payer",
    "denial",
    "appeal",
)

PROHIBITED_AUTONOMY_TERMS = (
    "approve the claim",
    "deny the claim",
    "override the clinician",
    "without human review",
    "make the final coverage decision",
)

CLINICAL_TERMS = (
    "diagnosis",
    "treatment",
    "medication",
    "dose",
    "therapy",
    "clinical",
    "patient",
)


def classify_request(
    prompt: str,
    *,
    contains_sensitive_data: bool = False,
    use_case: str | None = None,
) -> RiskAssessment:
    text = prompt.lower()
    detected_use_case = use_case or _detect_use_case(text)
    data_sensitivity = (
        DataSensitivity.PHI_PII if contains_sensitive_data else DataSensitivity.INTERNAL
    )

    if any(term in text for term in PROHIBITED_AUTONOMY_TERMS):
        return RiskAssessment(
            risk_tier=RiskTier.PROHIBITED,
            use_case=detected_use_case,
            rationale=(
                "Request asks the agent to make or bypass a final healthcare, "
                "coverage, or clinical decision without meaningful human review."
            ),
            data_sensitivity=data_sensitivity,
            requires_human_review=True,
        )

    if detected_use_case == "prior_authorization":
        return RiskAssessment(
            risk_tier=RiskTier.HIGH,
            use_case=detected_use_case,
            rationale=(
                "Prior-authorization support can affect access to care and must "
                "remain decision support with human review."
            ),
            data_sensitivity=data_sensitivity,
            requires_human_review=True,
        )

    if any(term in text for term in CLINICAL_TERMS):
        return RiskAssessment(
            risk_tier=RiskTier.LIMITED,
            use_case=detected_use_case,
            rationale=(
                "Healthcare-adjacent content requires safety boundaries even when "
                "the request is not a prior-authorization workflow."
            ),
            data_sensitivity=data_sensitivity,
            requires_human_review=contains_sensitive_data,
        )

    return RiskAssessment(
        risk_tier=RiskTier.MINIMAL,
        use_case=detected_use_case,
        rationale="No healthcare access, clinical, or sensitive-data signal detected.",
        data_sensitivity=data_sensitivity,
        requires_human_review=False,
    )


def _detect_use_case(text: str) -> str:
    if any(term in text for term in PRIOR_AUTH_TERMS):
        return "prior_authorization"
    if any(term in text for term in CLINICAL_TERMS):
        return "healthcare_support"
    return "general"
