from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class RiskTier(str, Enum):
    MINIMAL = "minimal"
    LIMITED = "limited"
    HIGH = "high"
    PROHIBITED = "prohibited"


class DataSensitivity(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    PHI_PII = "phi_pii"


class PolicyAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"


class PrivacyFinding(BaseModel):
    kind: str
    value: str
    replacement: str


class PrivacyAssessment(BaseModel):
    original_text: str
    redacted_text: str
    findings: list[PrivacyFinding] = Field(default_factory=list)
    contains_sensitive_data: bool = False


class RiskAssessment(BaseModel):
    risk_tier: RiskTier
    use_case: str
    rationale: str
    data_sensitivity: DataSensitivity
    requires_human_review: bool = False
    clinical_boundary: str = (
        "Administrative decision support only; not autonomous diagnosis, treatment, "
        "coverage approval, or coverage denial."
    )


class PolicyDecision(BaseModel):
    provider: str
    action: PolicyAction
    reason: str
    allowed_providers: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)


class EscalationDecision(BaseModel):
    required: bool
    reason: str
    target_queue: str | None = None


class AuditEvent(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: str
    event_type: str
    provider: str | None = None
    action: str | None = None
    risk_tier: RiskTier | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class GovernanceContext(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    original_prompt: str
    governed_prompt: str
    privacy: PrivacyAssessment
    risk: RiskAssessment
    escalation: EscalationDecision
    policy_decisions: list[PolicyDecision] = Field(default_factory=list)


class ProviderRequest(BaseModel):
    trace_id: str
    provider: str
    prompt: str
    risk_tier: RiskTier
    data_sensitivity: DataSensitivity
    human_review_required: bool


class ProviderResponse(BaseModel):
    trace_id: str
    provider: str
    content: str
    tokens: int = 0
    fallback_from: str | None = None


class ErrorEnvelope(BaseModel):
    trace_id: str
    category: str
    message: str
    retryable: bool = False


class RedTeamCase(BaseModel):
    id: str
    category: str
    prompt: str
    expected: dict[str, Any]


class RedTeamResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    case_id: str
    passed: bool
    observed: dict[str, Any]
    notes: str
