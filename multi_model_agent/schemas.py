from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
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

    def to_safe_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "replacement": self.replacement,
        }


class PrivacyAssessment(BaseModel):
    original_text: str
    redacted_text: str
    findings: list[PrivacyFinding] = Field(default_factory=list)
    contains_sensitive_data: bool = False

    def redaction_summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.kind] = counts.get(finding.kind, 0) + 1

        return {
            "total_findings": len(self.findings),
            "finding_counts_by_kind": dict(sorted(counts.items())),
            "contains_sensitive_data": self.contains_sensitive_data,
        }

    def to_safe_dict(self) -> dict[str, Any]:
        return {"redaction_summary": self.redaction_summary()}


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


class StoredAuditEvent(BaseModel):
    schema_version: Literal["audit.v1"] = "audit.v1"
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str
    trace_id: str
    event_type: str
    payload: dict[str, Any]
    payload_hash: str
    previous_hash: str | None = None
    event_hash: str


class AuditVerificationResult(BaseModel):
    valid: bool
    event_count: int
    errors: list[str] = Field(default_factory=list)
    final_hash: str | None = None
    path: str | None = None


class GovernanceContext(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    original_prompt: str
    governed_prompt: str
    privacy: PrivacyAssessment
    risk: RiskAssessment
    escalation: EscalationDecision
    policy_decisions: list[PolicyDecision] = Field(default_factory=list)

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "privacy": self.privacy.to_safe_dict(),
            "risk": self.risk.model_dump(mode="json"),
            "escalation": self.escalation.model_dump(mode="json"),
            "policy_decisions": [
                decision.model_dump(mode="json") for decision in self.policy_decisions
            ],
        }


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
