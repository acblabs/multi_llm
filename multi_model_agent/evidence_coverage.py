import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from .privacy import redact_sensitive_data
from .schemas import EvidenceCoverageItem, EvidenceCoverageReport, SourceReference


PROHIBITED_DECISION_BOUNDARY = [
    "coverage_approval",
    "coverage_denial",
    "medical_necessity_determination",
    "diagnosis_decision",
    "treatment_recommendation",
]

PROHIBITED_DECISION_LANGUAGE: dict[str, re.Pattern[str]] = {
    "coverage_approval_language": re.compile(r"\bapprove(?:d|s)?\b", re.IGNORECASE),
    "coverage_denial_language": re.compile(r"\bden(?:y|ied|ies)\b", re.IGNORECASE),
    "medical_necessity_language": re.compile(
        r"\b(?:not\s+)?medically\s+necessary\b",
        re.IGNORECASE,
    ),
    "diagnosis_decision_language": re.compile(
        r"\bdiagnosis\s+is\b",
        re.IGNORECASE,
    ),
    "treatment_recommendation_language": re.compile(
        r"\brecommend(?:s|ed|ing)?\s+treatment\b",
        re.IGNORECASE,
    ),
}

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")


@dataclass(frozen=True)
class EvidenceRequirement:
    requirement_id: str
    requirement_name: str
    present_patterns: tuple[str, ...]
    insufficient_patterns: tuple[str, ...] = ()
    required_by_default: bool = True
    relevance_patterns: tuple[str, ...] = ()


_REQUIREMENTS: tuple[EvidenceRequirement, ...] = (
    EvidenceRequirement(
        requirement_id="diagnosis_condition",
        requirement_name="Diagnosis or condition",
        present_patterns=(
            r"\bdiagnos(?:is|ed)\b",
            r"\bcondition\b",
            r"\bICD-?10\b",
            r"\bDx\b",
        ),
        insufficient_patterns=(
            r"\bdiagnos(?:is)?\s*[:#-]?\s*(?:pending|unknown|tbd|not\s+provided|not\s+documented|n/?a)\b",
            r"\bdiagnosis\s*[:#-]?\s*pain\b",
        ),
    ),
    EvidenceRequirement(
        requirement_id="requested_service",
        requirement_name="Requested service or procedure",
        present_patterns=(
            r"\brequest(?:ed|ing)?\s+(?:service|procedure|medication|drug|therapy|imaging|MRI|CT|DME)\b",
            r"\bprior\s+authorization\s+request\s+for\b",
            r"\bCPT\b",
            r"\bHCPCS\b",
            r"\bprocedure\b",
            r"\bMRI\b",
            r"\bCT\b",
        ),
        insufficient_patterns=(
            r"\brequested\s+(?:service|procedure)\s*[:#-]?\s*(?:tbd|unknown|not\s+provided|not\s+documented)\b",
        ),
    ),
    EvidenceRequirement(
        requirement_id="clinical_rationale",
        requirement_name="Clinical rationale",
        present_patterns=(
            r"\bclinical\s+rationale\b",
            r"\bindication\b",
            r"\bbecause\b",
            r"\bpersistent\b",
            r"\bsymptoms?\b",
            r"\bfunctional\s+impairment\b",
        ),
        insufficient_patterns=(
            r"\bclinical\s+rationale\s*[:#-]?\s*(?:tbd|unknown|not\s+provided|not\s+documented|none)\b",
            r"\bno\s+clinical\s+rationale\b",
        ),
    ),
    EvidenceRequirement(
        requirement_id="relevant_history",
        requirement_name="Relevant history",
        present_patterns=(
            r"\bhistory\b",
            r"\bprior\s+episodes?\b",
            r"\bduration\b",
            r"\bcomorbidit(?:y|ies)\b",
            r"\bprevious\b",
        ),
        insufficient_patterns=(
            r"\bhistory\s*[:#-]?\s*(?:tbd|unknown|not\s+provided|not\s+documented|none)\b",
            r"\bno\s+relevant\s+history\b",
        ),
    ),
    EvidenceRequirement(
        requirement_id="prior_conservative_therapy",
        requirement_name="Prior conservative therapy",
        present_patterns=(
            r"\bconservative\s+(?:therapy|treatment|management)\b",
            r"\bphysical\s+therapy\b",
            r"\bPT\b",
            r"\bNSAIDs?\b",
            r"\bhome\s+exercise\b",
            r"\btrial\s+of\b",
        ),
        insufficient_patterns=(
            r"\b(?:prior\s+)?conservative\s+(?:therapy|treatment|management)\s*[:#-]?\s*(?:tbd|unknown|not\s+provided|not\s+documented|none)\b",
            r"\bphysical\s+therapy\s*[:#-]?\s*(?:tbd|unknown|not\s+provided|not\s+documented|none)\b",
        ),
    ),
    EvidenceRequirement(
        requirement_id="imaging_or_lab_documentation",
        requirement_name="Imaging or lab documentation",
        present_patterns=(
            r"\b(?:imaging|radiology|MRI|CT|x-?ray)\s+(?:report|result(?:s)?|finding(?:s)?|documentation)\b",
            r"\b(?:MRI|CT|x-?ray)\s+(?:show(?:s|ed)?|demonstrat(?:e|es|ed)|reveal(?:s|ed)|note(?:s|d))\b",
            r"\b(?:lab|labs|laboratory)\s+(?:report|result(?:s)?|value(?:s)?|documentation)\b",
            r"\b(?:A1c|HbA1c|CBC|CMP|creatinine|hemoglobin)\s*[:=]?\s*\d",
        ),
        insufficient_patterns=(
            r"\b(?:imaging|labs?|results?)\s*[:#-]?\s*(?:tbd|unknown|not\s+provided|not\s+documented|unavailable|none)\b",
            r"\bno\s+(?:imaging|lab)\s+(?:documentation|results?)\b",
        ),
    ),
    EvidenceRequirement(
        requirement_id="medication_history",
        requirement_name="Medication history",
        present_patterns=(
            r"\bmedication\s+history\b",
            r"\bcurrent\s+medications?\b",
            r"\bprior\s+medications?\b",
            r"\b(?:tried|failed|trial\s+of|taking|currently\s+taking)\s+[A-Za-z0-9][A-Za-z0-9 ./-]*(?:medication|NSAIDs?|mg|tablet|injection|weekly|daily)\b",
            r"\b(?:dose|dosage)\s*[:#-]?\s*\d",
        ),
        insufficient_patterns=(
            r"\bmedication\s+history\s*[:#-]?\s*(?:tbd|unknown|not\s+provided|not\s+documented|none)\b",
            r"\bcurrent\s+medications?\s*[:#-]?\s*(?:tbd|unknown|not\s+provided|not\s+documented|none)\b",
        ),
        required_by_default=False,
        relevance_patterns=(
            r"\bmedication\b",
            r"\bdrug\b",
            r"\bpharmacy\b",
            r"\bRx\b",
            r"\bdose\b",
            r"\bmg\b",
            r"\binjection\b",
            r"\bbiologic\b",
        ),
    ),
    EvidenceRequirement(
        requirement_id="provider_notes",
        requirement_name="Provider notes",
        present_patterns=(
            r"\bprovider\s+notes?\b",
            r"\boffice\s+visit\b",
            r"\bprogress\s+notes?\b",
            r"\bclinical\s+notes?\b",
            r"\bvisit\s+notes?\b",
        ),
        insufficient_patterns=(
            r"\bprovider\s+notes?\s*[:#-]?\s*(?:tbd|unknown|not\s+provided|not\s+documented|none)\b",
            r"\bno\s+provider\s+notes?\b",
        ),
    ),
    EvidenceRequirement(
        requirement_id="payer_policy_references",
        requirement_name="Payer policy references",
        present_patterns=(
            r"\bpayer\s+policy\b",
            r"\bpolicy\s+reference\b",
            r"\bcoverage\s+criteria\b",
            r"\bplan\s+criteria\b",
            r"\bguideline\b",
        ),
        insufficient_patterns=(
            r"\bpayer\s+policy\s*[:#-]?\s*(?:tbd|unknown|not\s+provided|not\s+documented|none)\b",
            r"\bpolicy\s+reference\s*[:#-]?\s*(?:tbd|unknown|not\s+provided|not\s+documented|none)\b",
        ),
        required_by_default=False,
        relevance_patterns=(
            r"\bpayer\b",
            r"\bpolicy\b",
            r"\bcriteria\b",
            r"\bguideline\b",
            r"\bplan\s+requirement\b",
        ),
    ),
)


def generate_evidence_coverage_report(
    *,
    trace_id: str,
    text: str,
    source_locations: Mapping[str, SourceReference | Mapping[str, Any]] | None = None,
    workflow_type: str = "prior_authorization",
    human_review_required: bool = True,
) -> EvidenceCoverageReport:
    """Build a deterministic prior-auth evidence coverage report.

    The report is documentation triage for human review. It never stores source
    excerpts, and it never makes coverage, medical-necessity, diagnosis, or
    treatment decisions.
    """
    redacted_text = redact_sensitive_data(text).redacted_text
    items = [
        _build_item(requirement, redacted_text, source_locations or {})
        for requirement in _REQUIREMENTS
    ]
    summary = _overall_summary(items, human_review_required=human_review_required)
    report = EvidenceCoverageReport(
        trace_id=_safe_identifier(trace_id, label="trace"),
        workflow_type=workflow_type,
        items=items,
        overall_summary=summary,
        human_review_required=human_review_required,
        prohibited_decision_boundary=list(PROHIBITED_DECISION_BOUNDARY),
    )
    assert_evidence_report_boundary(report)
    return report


def find_prohibited_decision_language(text: str) -> list[str]:
    violations: list[str] = []
    for label, pattern in PROHIBITED_DECISION_LANGUAGE.items():
        if pattern.search(text):
            violations.append(label)
    return violations


def validate_evidence_report_boundaries(
    report: EvidenceCoverageReport,
    *,
    reviewer_summary: str | None = None,
) -> list[str]:
    violations: list[str] = []
    for index, item in enumerate(report.items):
        for label in find_prohibited_decision_language(item.rationale):
            violations.append(f"items[{index}].rationale:{label}")

    for label in find_prohibited_decision_language(report.overall_summary):
        violations.append(f"overall_summary:{label}")

    if reviewer_summary:
        for label in find_prohibited_decision_language(reviewer_summary):
            violations.append(f"reviewer_summary:{label}")

    return violations


def assert_evidence_report_boundary(
    report: EvidenceCoverageReport,
    *,
    reviewer_summary: str | None = None,
) -> None:
    violations = validate_evidence_report_boundaries(
        report,
        reviewer_summary=reviewer_summary,
    )
    if violations:
        raise ValueError(
            "Evidence coverage report contains prohibited decision language: "
            + ", ".join(violations)
        )


def _build_item(
    requirement: EvidenceRequirement,
    redacted_text: str,
    source_locations: Mapping[str, SourceReference | Mapping[str, Any]],
) -> EvidenceCoverageItem:
    relevant = requirement.required_by_default or _matches_any(
        redacted_text,
        requirement.relevance_patterns,
    )[0]
    insufficient, insufficient_match = _matches_any(
        redacted_text,
        requirement.insufficient_patterns,
    )
    present, present_match = _matches_any(redacted_text, requirement.present_patterns)

    if not relevant:
        status = "not_applicable"
        match = None
    elif insufficient:
        status = "insufficient"
        match = insufficient_match
    elif present:
        status = "present"
        match = present_match
    else:
        status = "missing"
        match = None

    return EvidenceCoverageItem(
        requirement_id=requirement.requirement_id,
        requirement_name=requirement.requirement_name,
        status=status,
        source_reference=_source_reference_for(
            requirement.requirement_id,
            source_locations,
        ),
        source_excerpt_hash=_source_excerpt_hash(redacted_text, match),
        rationale=_rationale_for(requirement, status),
    )


def _matches_any(
    text: str,
    patterns: tuple[str, ...],
) -> tuple[bool, re.Match[str] | None]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return True, match
    return False, None


def _source_reference_for(
    requirement_id: str,
    source_locations: Mapping[str, SourceReference | Mapping[str, Any]],
) -> SourceReference | None:
    value = source_locations.get(requirement_id) or source_locations.get("default")
    if value is None:
        return None
    if isinstance(value, SourceReference):
        reference = value
    elif isinstance(value, Mapping):
        reference = SourceReference.model_validate(dict(value))
    else:
        return None

    safe_reference = reference.to_safe_dict()
    return SourceReference.model_validate(safe_reference) if safe_reference else None


def _source_excerpt_hash(
    redacted_text: str,
    match: re.Match[str] | None,
) -> str | None:
    if match is None:
        return None

    start = max(match.start() - 80, 0)
    end = min(match.end() + 80, len(redacted_text))
    excerpt = redacted_text[start:end].strip()
    if not excerpt:
        return None
    return sha256(excerpt.encode("utf-8")).hexdigest()


def _rationale_for(requirement: EvidenceRequirement, status: str) -> str:
    lower_name = requirement.requirement_name.lower()
    if status == "present":
        return f"Documentation contains a {lower_name} signal for reviewer verification."
    if status == "insufficient":
        return (
            f"Documentation mentions {lower_name} but leaves detail incomplete "
            "for reviewer verification."
        )
    if status == "not_applicable":
        return (
            f"{requirement.requirement_name} was not indicated as relevant by "
            "the supplied documentation."
        )
    return f"Documentation does not include a clear {lower_name} element."


def _overall_summary(
    items: list[EvidenceCoverageItem],
    *,
    human_review_required: bool,
) -> str:
    counts = {status: 0 for status in ("present", "insufficient", "missing", "not_applicable")}
    for item in items:
        counts[item.status] += 1
    review_sentence = (
        "Human review remains required."
        if human_review_required
        else "Human review status follows the governing workflow."
    )
    return (
        "Evidence coverage found "
        f"{counts['present']} present, {counts['insufficient']} insufficient, "
        f"{counts['missing']} missing, and {counts['not_applicable']} not applicable "
        f"documentation elements. {review_sentence} This report is documentation "
        "support only."
    )


def _safe_identifier(value: str, *, label: str) -> str:
    if _SAFE_IDENTIFIER_RE.fullmatch(value):
        return value
    digest = sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{label}:{digest}"
