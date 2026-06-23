import json
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel

from .audit_hashing import (
    AUDIT_SCHEMA_VERSION,
    canonical_json,
    compute_event_hash,
    compute_payload_hash,
    format_datetime_utc,
)
from .privacy import redact_sensitive_data
from .schemas import (
    AuditEvent,
    AuditVerificationResult,
    GovernanceContext,
    PrivacyAssessment,
    PrivacyFinding,
    RiskTier,
    StoredAuditEvent,
)


SAFE_DETAIL_KEYS = {
    "action",
    "allowed_providers",
    "attempt",
    "contains_sensitive_data",
    "cost_usd",
    "error_category",
    "estimated_cost_usd",
    "fallback_from",
    "fallback_to",
    "finding_counts_by_kind",
    "finding_kinds",
    "human_review_required",
    "input_tokens",
    "max_retries",
    "model",
    "model_provenance",
    "output_tokens",
    "policy_ids",
    "provider",
    "reason_codes",
    "redaction_count",
    "redaction_summary",
    "requires_human_review",
    "retryable",
    "risk_tier",
    "target",
    "target_queue",
    "token_counts",
    "tokens",
    "total_findings",
    "use_case",
}

DROP_DETAIL_KEYS = {
    "content",
    "date_of_birth",
    "dob",
    "email",
    "exception",
    "excerpt",
    "full_model_prompt",
    "full_model_response",
    "full_prompt",
    "governed_prompt",
    "member_id",
    "model_prompt",
    "model_response",
    "original_prompt",
    "original_text",
    "patient_name",
    "phone",
    "prompt",
    "raw_payload",
    "raw_prompt",
    "raw_response",
    "raw_reviewer_id",
    "response",
    "reviewer_id",
    "source_excerpt",
    "ssn",
    "system_prompt",
    "text",
    "unredacted_excerpt",
    "user_prompt",
}

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_SAFE_PROVENANCE_TEXT_RE = re.compile(r"^[A-Za-z0-9_.:/@+-]{1,200}$")
_BARE_DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_PERSON_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b")
_JSONL_LOCKS: dict[Path, RLock] = {}
_JSONL_LOCKS_LOCK = RLock()

SAFE_MODEL_PROVENANCE_KEYS = {
    "deployment",
    "fallback_from",
    "fallback_to",
    "model",
    "model_id",
    "model_version",
    "provider",
    "region",
    "request_id",
    "response_id",
    "revision",
    "service",
    "snapshot",
    "snapshot_id",
    "system",
    "vendor",
    "version",
}

SAFE_TOKEN_COUNT_KEYS = {
    "cached_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "reasoning_tokens",
    "total_tokens",
    "tokens",
}


class AuditStore(Protocol):
    def append(self, event: AuditEvent | dict[str, Any]) -> StoredAuditEvent:
        ...

    def query_by_trace_id(self, trace_id: str) -> list[StoredAuditEvent]:
        ...

    def list_events(self) -> list[StoredAuditEvent]:
        ...

    def verify_chain(self) -> AuditVerificationResult:
        ...

    def clear(self) -> None:
        ...


def sanitize_for_persistence(event: AuditEvent | dict[str, Any]) -> dict[str, Any]:
    """Return the allowlisted payload used for persistence and hashing."""
    data = _event_to_dict(event)
    payload: dict[str, Any] = {}

    for key in ("provider", "action", "risk_tier"):
        value = data.get(key)
        if value is not None:
            payload[key] = _sanitize_scalar(value)

    details = data.get("details") or {}
    if data.get("event_type") == "metric":
        safe_details = _sanitize_metric_details(details)
    else:
        safe_details = safe_audit_details(details)
    if safe_details:
        payload["details"] = safe_details

    return payload


def safe_audit_details(details: Any) -> dict[str, Any]:
    if isinstance(details, PrivacyAssessment):
        return details.to_safe_dict()
    if isinstance(details, GovernanceContext):
        return _sanitize_mapping(details.to_safe_dict())
    safe = _sanitize_mapping(details if isinstance(details, dict) else {"value": details})
    return safe if isinstance(safe, dict) else {}


def stored_event_to_audit_dict(event: StoredAuditEvent) -> dict[str, Any]:
    payload = event.payload
    return {
        "timestamp": event.timestamp,
        "trace_id": event.trace_id,
        "event_type": event.event_type,
        "provider": payload.get("provider"),
        "action": payload.get("action"),
        "risk_tier": payload.get("risk_tier"),
        "details": payload.get("details", {}),
        "schema_version": event.schema_version,
        "event_id": event.event_id,
        "payload_hash": event.payload_hash,
        "previous_hash": event.previous_hash,
        "event_hash": event.event_hash,
        "payload": payload,
    }


class InMemoryAuditStore:
    def __init__(self) -> None:
        self._events: list[StoredAuditEvent] = []
        self._lock = RLock()

    def append(self, event: AuditEvent | dict[str, Any]) -> StoredAuditEvent:
        with self._lock:
            previous_hash = self._events[-1].event_hash if self._events else None
            stored = _build_stored_event(event, previous_hash=previous_hash)
            self._events.append(stored)
            return stored

    def query_by_trace_id(self, trace_id: str) -> list[StoredAuditEvent]:
        safe_trace_id = _safe_identifier(trace_id, label="trace")
        with self._lock:
            return [event for event in self._events if event.trace_id == safe_trace_id]

    def list_events(self) -> list[StoredAuditEvent]:
        with self._lock:
            return list(self._events)

    def verify_chain(self) -> AuditVerificationResult:
        with self._lock:
            return verify_stored_events(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class JsonlAuditStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = _jsonl_lock_for_path(self.path)

    def append(self, event: AuditEvent | dict[str, Any]) -> StoredAuditEvent:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            existing_events = self._load_events()
            verification = verify_stored_events(existing_events, path=str(self.path))
            if not verification.valid:
                raise ValueError(
                    "Refusing to append to invalid audit chain: "
                    + "; ".join(verification.errors)
                )

            previous_hash = existing_events[-1].event_hash if existing_events else None
            stored = _build_stored_event(event, previous_hash=previous_hash)
            with self.path.open("a", encoding="utf-8", newline="\n") as audit_file:
                audit_file.write(canonical_json(stored.model_dump(mode="json")))
                audit_file.write("\n")
            return stored

    def query_by_trace_id(self, trace_id: str) -> list[StoredAuditEvent]:
        safe_trace_id = _safe_identifier(trace_id, label="trace")
        with self._lock:
            return [
                event for event in self._load_events() if event.trace_id == safe_trace_id
            ]

    def list_events(self) -> list[StoredAuditEvent]:
        with self._lock:
            return self._load_events()

    def verify_chain(self) -> AuditVerificationResult:
        with self._lock:
            return verify_stored_events(self._load_events_for_verification(), path=str(self.path))

    def clear(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def _load_events(self) -> list[StoredAuditEvent]:
        if not self.path.exists():
            return []

        events: list[StoredAuditEvent] = []
        with self.path.open("r", encoding="utf-8") as audit_file:
            for line_number, line in enumerate(audit_file, start=1):
                if not line.strip():
                    continue
                try:
                    events.append(StoredAuditEvent.model_validate_json(line))
                except Exception as error:
                    raise ValueError(
                        f"Invalid audit event at line {line_number}: {error}"
                    ) from error
        return events

    def _load_events_for_verification(self) -> list[StoredAuditEvent | str]:
        if not self.path.exists():
            return []

        events: list[StoredAuditEvent | str] = []
        with self.path.open("r", encoding="utf-8") as audit_file:
            for line in audit_file:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    events.append(StoredAuditEvent.model_validate_json(stripped))
                except Exception:
                    events.append(stripped)
        return events


def verify_stored_events(
    events: Iterable[StoredAuditEvent | str],
    *,
    path: str | None = None,
) -> AuditVerificationResult:
    errors: list[str] = []
    previous_hash: str | None = None
    event_count = 0

    for index, event in enumerate(events, start=1):
        event_count += 1
        if isinstance(event, str):
            errors.append(f"Line {index}: invalid stored audit event JSON/schema")
            previous_hash = None
            continue

        if event.schema_version != AUDIT_SCHEMA_VERSION:
            errors.append(
                f"Line {index}: unsupported schema_version {event.schema_version!r}"
            )

        if event.previous_hash != previous_hash:
            errors.append(
                f"Line {index}: previous_hash mismatch; expected {previous_hash!r}"
            )

        try:
            expected_payload_hash = compute_payload_hash(event.payload)
            if event.payload_hash != expected_payload_hash:
                errors.append(f"Line {index}: payload_hash mismatch")

            expected_event_hash = compute_event_hash(
                schema_version=event.schema_version,
                event_id=event.event_id,
                timestamp=event.timestamp,
                trace_id=event.trace_id,
                event_type=event.event_type,
                payload_hash=event.payload_hash,
                previous_hash=event.previous_hash,
            )
            if event.event_hash != expected_event_hash:
                errors.append(f"Line {index}: event_hash mismatch")
        except Exception as error:
            errors.append(f"Line {index}: hash recomputation failed: {error}")

        previous_hash = event.event_hash

    return AuditVerificationResult(
        valid=not errors,
        event_count=event_count,
        errors=errors,
        final_hash=previous_hash,
        path=path,
    )


def _jsonl_lock_for_path(path: Path) -> RLock:
    resolved_path = path.expanduser().resolve()
    with _JSONL_LOCKS_LOCK:
        lock = _JSONL_LOCKS.get(resolved_path)
        if lock is None:
            lock = RLock()
            _JSONL_LOCKS[resolved_path] = lock
        return lock


def _build_stored_event(
    event: AuditEvent | dict[str, Any],
    *,
    previous_hash: str | None,
) -> StoredAuditEvent:
    data = _event_to_dict(event)
    trace_id = _safe_identifier(str(data.get("trace_id", "")), label="trace")
    event_type = _safe_identifier(str(data.get("event_type", "")), label="event_type")
    if not trace_id:
        raise ValueError("Audit events require a trace_id")
    if not event_type:
        raise ValueError("Audit events require an event_type")

    timestamp = _coerce_timestamp(data.get("timestamp"))
    event_id = _safe_identifier(str(data.get("event_id") or uuid4()), label="event")
    payload = sanitize_for_persistence(event)
    payload_hash = compute_payload_hash(payload)
    event_hash = compute_event_hash(
        schema_version=AUDIT_SCHEMA_VERSION,
        event_id=event_id,
        timestamp=timestamp,
        trace_id=trace_id,
        event_type=event_type,
        payload_hash=payload_hash,
        previous_hash=previous_hash,
    )

    return StoredAuditEvent(
        schema_version=AUDIT_SCHEMA_VERSION,
        event_id=event_id,
        timestamp=timestamp,
        trace_id=trace_id,
        event_type=event_type,
        payload=payload,
        payload_hash=payload_hash,
        previous_hash=previous_hash,
        event_hash=event_hash,
    )


def _event_to_dict(event: AuditEvent | dict[str, Any]) -> dict[str, Any]:
    if isinstance(event, AuditEvent):
        return event.model_dump(mode="python")
    if isinstance(event, BaseModel):
        return event.model_dump(mode="python")
    return dict(event)


def _coerce_timestamp(value: Any) -> str:
    if value is None:
        return format_datetime_utc(datetime.now(timezone.utc))
    if isinstance(value, datetime):
        return format_datetime_utc(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return format_datetime_utc(parsed)
        except ValueError:
            return format_datetime_utc(datetime.now(timezone.utc))
    raise TypeError(f"Unsupported audit timestamp type: {type(value).__name__}")


def _sanitize_metric_details(details: Any) -> dict[str, Any]:
    if not isinstance(details, dict):
        return {}

    result: dict[str, Any] = {}
    name = details.get("name")
    if isinstance(name, str) and _SAFE_IDENTIFIER_RE.fullmatch(name):
        result["name"] = name

    value = details.get("value")
    if isinstance(value, (bool, int, float)):
        result["value"] = value

    for key in ("provider", "risk_tier", "tokens", "input_tokens", "output_tokens"):
        if key in details:
            safe_value = _sanitize_value(details[key])
            if safe_value is not None:
                result[key] = safe_value

    return result


def _sanitize_model_provenance(value: Any) -> Any:
    if isinstance(value, list):
        sanitized = [_sanitize_model_provenance(item) for item in value]
        return [item for item in sanitized if item]

    if not isinstance(value, dict):
        return None

    result: dict[str, Any] = {}
    for key, item in value.items():
        normalized_key = _normalize_key(key)
        if normalized_key == "token_counts":
            safe_token_counts = _sanitize_token_counts(item)
            if safe_token_counts:
                result["token_counts"] = safe_token_counts
            continue

        if normalized_key not in SAFE_MODEL_PROVENANCE_KEYS:
            continue

        if isinstance(item, str):
            if _SAFE_PROVENANCE_TEXT_RE.fullmatch(item):
                result[normalized_key] = item
            continue

        if isinstance(item, (bool, int, float)):
            result[normalized_key] = item

    return result


def _sanitize_token_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}

    result: dict[str, int] = {}
    for key, item in value.items():
        normalized_key = _normalize_key(key)
        if normalized_key in SAFE_TOKEN_COUNT_KEYS and isinstance(item, int) and item >= 0:
            result[normalized_key] = item
    return result


def _safe_identifier(value: str, *, label: str) -> str:
    if _SAFE_IDENTIFIER_RE.fullmatch(value):
        return value
    digest = sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{label}:{digest}"


def _sanitize_mapping(value: dict[str, Any]) -> dict[str, Any]:
    if _looks_like_privacy_finding(value):
        return _safe_privacy_finding_dict(value)
    if _looks_like_privacy_assessment(value):
        return _safe_privacy_assessment_dict(value)

    result: dict[str, Any] = {}
    contains_sensitive_data = bool(value.get("contains_sensitive_data", False))

    for key, item in value.items():
        normalized_key = _normalize_key(key)
        if normalized_key in DROP_DETAIL_KEYS:
            continue

        if normalized_key == "error":
            result["error_category"] = _safe_error_category(item)
            continue

        if normalized_key == "findings":
            summary = _redaction_summary_from_findings(
                item,
                contains_sensitive_data=contains_sensitive_data,
            )
            result["redaction_summary"] = summary
            continue

        if isinstance(item, PrivacyAssessment):
            result["redaction_summary"] = item.redaction_summary()
            continue

        if normalized_key == "model_provenance":
            safe_provenance = _sanitize_model_provenance(item)
            if safe_provenance:
                result["model_provenance"] = safe_provenance
            continue

        if normalized_key == "token_counts":
            safe_token_counts = _sanitize_token_counts(item)
            if safe_token_counts:
                result["token_counts"] = safe_token_counts
            continue

        if normalized_key == "privacy":
            safe_value = _sanitize_value(item)
            if isinstance(safe_value, dict) and "redaction_summary" in safe_value:
                result["redaction_summary"] = safe_value["redaction_summary"]
            continue

        if normalized_key not in SAFE_DETAIL_KEYS:
            continue

        safe_value = _sanitize_value(item)
        if safe_value is not None:
            result[normalized_key] = safe_value

    return result


def _sanitize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float, RiskTier)):
        return _sanitize_scalar(value)
    if isinstance(value, PrivacyAssessment):
        return value.to_safe_dict()
    if isinstance(value, PrivacyFinding):
        return value.to_safe_dict()
    if isinstance(value, GovernanceContext):
        return _sanitize_mapping(value.to_safe_dict())
    if isinstance(value, BaseModel):
        return _sanitize_value(value.model_dump(mode="json"))
    if isinstance(value, list):
        sanitized = [_sanitize_value(item) for item in value]
        return [item for item in sanitized if item is not None]
    if isinstance(value, tuple):
        sanitized = [_sanitize_value(item) for item in value]
        return [item for item in sanitized if item is not None]
    if isinstance(value, dict):
        return _sanitize_mapping(value)
    return None


def _sanitize_scalar(value: Any) -> Any:
    if isinstance(value, RiskTier):
        return value.value
    if isinstance(value, str):
        return _redact_text_for_persistence(value)
    return value


def _redact_text_for_persistence(value: str) -> str:
    redacted = redact_sensitive_data(value).redacted_text
    redacted = _BARE_DATE_RE.sub("[DATE]", redacted)
    redacted = _PERSON_NAME_RE.sub(_replace_person_name, redacted)
    return redacted


def _replace_person_name(match: re.Match[str]) -> str:
    phrase = match.group(0)
    if phrase.startswith("["):
        return phrase
    safe_phrases = {
        "Human Review",
        "Prior Authorization",
        "Provider Access",
    }
    if phrase in safe_phrases:
        return phrase
    return "[PERSON]"


def _safe_error_category(value: Any) -> str:
    message = str(value).lower()
    if any(token in message for token in ("timeout", "connection", "rate_limit", "429")):
        return "retryable_provider_error"
    if any(token in message for token in ("overloaded", "internal_server_error", "500")):
        return "provider_fallback_error"
    if any(token in message for token in ("authentication", "api key", "invalid_request", "400")):
        return "provider_configuration_error"
    return "provider_error"


def _normalize_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(key).strip().lower()).strip("_")


def _looks_like_privacy_finding(value: dict[str, Any]) -> bool:
    return {"kind", "value", "replacement"}.issubset(value.keys())


def _looks_like_privacy_assessment(value: dict[str, Any]) -> bool:
    assessment_keys = {
        "contains_sensitive_data",
        "findings",
        "original_text",
        "redacted_text",
    }
    return (
        set(value.keys()).issubset(assessment_keys)
        and "findings" in value
        and ("original_text" in value or "redacted_text" in value)
    )


def _safe_privacy_finding_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": _redact_text_for_persistence(str(value.get("kind", ""))),
        "replacement": _redact_text_for_persistence(str(value.get("replacement", ""))),
    }


def _safe_privacy_assessment_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "redaction_summary": _redaction_summary_from_findings(
            value.get("findings", []),
            contains_sensitive_data=bool(value.get("contains_sensitive_data", False)),
        )
    }


def _redaction_summary_from_findings(
    findings: Any,
    *,
    contains_sensitive_data: bool,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    total = 0

    if isinstance(findings, list):
        for finding in findings:
            kind = _finding_kind(finding)
            if not kind:
                continue
            total += 1
            counts[kind] = counts.get(kind, 0) + 1

    return {
        "total_findings": total,
        "finding_counts_by_kind": dict(sorted(counts.items())),
        "contains_sensitive_data": contains_sensitive_data or total > 0,
    }


def _finding_kind(finding: Any) -> str | None:
    if isinstance(finding, PrivacyFinding):
        return finding.kind
    if isinstance(finding, str):
        return _redact_text_for_persistence(finding)
    if isinstance(finding, dict):
        kind = finding.get("kind")
        return _redact_text_for_persistence(str(kind)) if kind else None
    return None
