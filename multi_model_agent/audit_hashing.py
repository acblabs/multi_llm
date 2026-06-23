import json
from datetime import date, datetime, timezone
from enum import Enum
from hashlib import sha256
from math import isfinite
from typing import Any

from pydantic import BaseModel


AUDIT_SCHEMA_VERSION = "audit.v1"


def format_datetime_utc(value: datetime) -> str:
    """Return a UTC ISO-8601 timestamp with fixed microsecond precision."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    utc_value = value.astimezone(timezone.utc)
    return utc_value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def normalize_for_json(value: Any) -> Any:
    """Normalize values before canonical JSON serialization.

    Hashing should fail loudly for unexpected runtime objects instead of
    quietly stringifying values that may be non-deterministic or unsafe.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not isfinite(value):
            raise TypeError("Non-finite floats cannot be canonicalized")
        return value

    if isinstance(value, datetime):
        return format_datetime_utc(value)

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, BaseModel):
        return normalize_for_json(value.model_dump(mode="json"))

    if isinstance(value, tuple):
        return [normalize_for_json(item) for item in value]

    if isinstance(value, list):
        return [normalize_for_json(item) for item in value]

    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Canonical JSON object keys must be strings: {key!r}")
            normalized[key] = normalize_for_json(item)
        return normalized

    raise TypeError(f"Object of type {type(value).__name__} cannot be canonicalized")


def canonical_json(value: Any) -> str:
    return json.dumps(
        normalize_for_json(value),
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_canonical(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def compute_payload_hash(payload: dict[str, Any]) -> str:
    return sha256_canonical(payload)


def compute_event_hash(
    *,
    schema_version: str,
    event_id: str,
    timestamp: str,
    trace_id: str,
    event_type: str,
    payload_hash: str,
    previous_hash: str | None,
) -> str:
    return sha256_canonical(
        {
            "schema_version": schema_version,
            "event_id": event_id,
            "timestamp": timestamp,
            "trace_id": trace_id,
            "event_type": event_type,
            "payload_hash": payload_hash,
            "previous_hash": previous_hash,
        }
    )
