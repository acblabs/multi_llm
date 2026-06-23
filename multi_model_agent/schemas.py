import re
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr


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


APPROVED_PROVIDER_NAMES = ("openai", "claude", "grok")
SAFE_DECISION_SCALAR = StrictStr | StrictInt | StrictFloat | StrictBool | None

_SAFE_SCHEMA_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_SCHEMA_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SCHEMA_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[-.\s])?(?:\(\d{3}\)[-.\s]?|\d{3}[-.\s])\d{3}[-.\s]\d{4}(?!\d)"
)
_SCHEMA_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_SCHEMA_DOB_RE = re.compile(
    r"\b(?:dob|date of birth|birth date)\s*[:#-]?\s*"
    r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Z][a-z]+ \d{1,2}, \d{4})",
    re.IGNORECASE,
)
_SCHEMA_MEMBER_ID_RE = re.compile(
    r"\b(?:member|subscriber|policy|patient|mrn|claim)\s*(?:id|number|#)\s*"
    r"[:#-]?\s*[A-Z0-9-]{5,}\b",
    re.IGNORECASE,
)
_SCHEMA_PATIENT_NAME_RE = re.compile(
    r"\b(?:patient|member)\s*(?:name)?\s*[:#-]?\s*"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b",
    re.IGNORECASE,
)
_SCHEMA_BARE_DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")


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


class DecisionFactor(BaseModel):
    name: str
    value: SAFE_DECISION_SCALAR
    weight: float | None = None
    description: str | None = None

    def to_safe_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": _safe_schema_identifier(self.name, label="factor"),
            "value": _safe_schema_scalar(self.value),
        }
        if self.weight is not None:
            result["weight"] = self.weight
        if self.description:
            result["description"] = _redact_schema_text(self.description)
        return result


class GovernanceDecisionExplanation(BaseModel):
    decision_id: str = Field(
        description=(
            "Per-trace governance decision instance identifier. This is not a "
            "stable rule ID or production-scale uniqueness guarantee."
        )
    )
    decision_type: str
    result: SAFE_DECISION_SCALAR
    reason_codes: list[str]
    human_rationale: str
    factors: list[DecisionFactor] = Field(default_factory=list)
    policy_ids: list[str] = Field(default_factory=list)
    risk_tier: str | None = None
    requires_human_review: bool | None = None
    trace_id: str | None = None

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "decision_id": _safe_schema_identifier(self.decision_id, label="decision"),
            "decision_type": _safe_schema_identifier(
                self.decision_type,
                label="decision_type",
            ),
            "result": _safe_schema_scalar(self.result),
            "reason_codes": [
                _safe_schema_identifier(code, label="reason_code")
                for code in self.reason_codes
            ],
            "human_rationale": _redact_schema_text(self.human_rationale),
            "factors": [factor.to_safe_dict() for factor in self.factors],
            "policy_ids": [
                _safe_schema_identifier(policy_id, label="policy")
                for policy_id in self.policy_ids
            ],
            "risk_tier": _redact_schema_text(self.risk_tier)
            if self.risk_tier
            else None,
            "requires_human_review": self.requires_human_review,
            "trace_id": _safe_schema_identifier(self.trace_id, label="trace")
            if self.trace_id
            else None,
        }


class RiskAssessment(BaseModel):
    risk_tier: RiskTier
    use_case: str
    rationale: str
    data_sensitivity: DataSensitivity
    requires_human_review: bool = False
    reason_codes: list[str] = Field(default_factory=list)
    policy_ids: list[str] = Field(default_factory=list)
    clinical_boundary: str = (
        "Administrative decision support only; not autonomous diagnosis, treatment, "
        "coverage approval, or coverage denial."
    )

    def to_safe_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class PolicyDecision(BaseModel):
    provider: str
    action: PolicyAction
    reason: str
    allowed_providers: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    policy_ids: list[str] = Field(default_factory=list)

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "provider": (
                self.provider
                if self.provider in APPROVED_PROVIDER_NAMES
                else "unapproved_provider"
            ),
            "action": self.action.value,
            "allowed_providers": list(self.allowed_providers),
            "reason_codes": list(self.reason_codes),
            "policy_ids": list(self.policy_ids),
        }


class EscalationDecision(BaseModel):
    required: bool
    reason: str
    target_queue: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    policy_ids: list[str] = Field(default_factory=list)

    def to_safe_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


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
    explanations: list[GovernanceDecisionExplanation] = Field(default_factory=list)

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "privacy": self.privacy.to_safe_dict(),
            "risk": self.risk.to_safe_dict(),
            "escalation": self.escalation.to_safe_dict(),
            "policy_decisions": [
                decision.to_safe_dict() for decision in self.policy_decisions
            ],
            "governance_explanations": [
                explanation.to_safe_dict() for explanation in self.explanations
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


def _safe_schema_identifier(value: str, *, label: str) -> str:
    if _SAFE_SCHEMA_IDENTIFIER_RE.fullmatch(value):
        return value
    digest = sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{label}:{digest}"


def _safe_schema_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact_schema_text(value)
    return None


def _redact_schema_text(value: str) -> str:
    redacted = _SCHEMA_EMAIL_RE.sub("[EMAIL]", value)
    redacted = _SCHEMA_PHONE_RE.sub("[PHONE]", redacted)
    redacted = _SCHEMA_SSN_RE.sub("[SSN]", redacted)
    redacted = _SCHEMA_DOB_RE.sub("[DOB]", redacted)
    redacted = _SCHEMA_MEMBER_ID_RE.sub("[MEMBER_ID]", redacted)
    redacted = _SCHEMA_PATIENT_NAME_RE.sub("[PATIENT_NAME]", redacted)
    return _SCHEMA_BARE_DATE_RE.sub("[DATE]", redacted)
