import re

from .schemas import PrivacyAssessment, PrivacyFinding


SENSITIVE_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "email",
        "[EMAIL]",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "phone",
        "[PHONE]",
        re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    ),
    ("ssn", "[SSN]", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (
        "date_of_birth",
        "[DOB]",
        re.compile(
            r"\b(?:dob|date of birth|birth date)\s*[:#-]?\s*"
            r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Z][a-z]+ \d{1,2}, \d{4})",
            re.IGNORECASE,
        ),
    ),
    (
        "member_id",
        "[MEMBER_ID]",
        re.compile(
            r"\b(?:member|subscriber|policy|patient|mrn|claim)\s*(?:id|number|#)\s*"
            r"[:#-]?\s*[A-Z0-9-]{5,}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "patient_name",
        "[PATIENT_NAME]",
        re.compile(
            r"\b(?:patient|member)\s*(?:name)?\s*[:#-]\s*"
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b",
            re.IGNORECASE,
        ),
    ),
]


def redact_sensitive_data(text: str) -> PrivacyAssessment:
    redacted = text
    findings: list[PrivacyFinding] = []

    for kind, replacement, pattern in SENSITIVE_PATTERNS:
        def _replace(match: re.Match[str]) -> str:
            value = match.group(0)
            findings.append(
                PrivacyFinding(kind=kind, value=value, replacement=replacement)
            )
            return replacement

        redacted = pattern.sub(_replace, redacted)

    return PrivacyAssessment(
        original_text=text,
        redacted_text=redacted,
        findings=findings,
        contains_sensitive_data=bool(findings),
    )
