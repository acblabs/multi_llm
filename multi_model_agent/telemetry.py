import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any

from .privacy import redact_sensitive_data


TELEMETRY_ENABLED_ENV = "MULTI_LLM_OTEL_ENABLED"
TELEMETRY_CAPTURE_SPANS_ENV = "MULTI_LLM_TELEMETRY_CAPTURE_SPANS"
TELEMETRY_CAPTURE_METRICS_ENV = "MULTI_LLM_TELEMETRY_CAPTURE_METRICS"
TELEMETRY_AUTO_CONFIGURE_ENV = "MULTI_LLM_OTEL_AUTO_CONFIGURE"
TELEMETRY_SERVICE_NAME_ENV = "OTEL_SERVICE_NAME"
DEFAULT_SERVICE_NAME = "multi-llm-governance"

SPAN_REQUEST = "request"
SPAN_PRE_ROUTER_REDACTION = "pre_router.redaction"
SPAN_RISK_CLASSIFICATION = "risk.classification"
SPAN_POLICY_EGRESS_CHECK = "policy.egress_check"
SPAN_PROVIDER_CALL_TEMPLATE = "provider.{provider}.call"
SPAN_SYNTHESIS_GEMINI = "synthesis.gemini"
SPAN_OUTPUT_SCHEMA_VALIDATION = "output.schema_validation"
SPAN_ESCALATION_HITL = "escalation.hitl"

METRIC_REQUEST_COUNT = "request_count"
METRIC_ERROR_COUNT = "error_count"
METRIC_PROVIDER_LATENCY_MS = "provider_latency_ms"
METRIC_RETRY_COUNT = "retry_count"
METRIC_FALLBACK_COUNT = "fallback_count"
METRIC_ESTIMATED_COST_USD = "estimated_cost_usd"
METRIC_RISK_TIER_DISTRIBUTION = "risk_tier_distribution"
METRIC_REDACTION_COUNT_DISTRIBUTION = "redaction_count_distribution"
METRIC_POLICY_VIOLATION_ATTEMPTS = "policy_violation_attempts"
METRIC_HUMAN_ESCALATIONS_TOTAL = "human_escalations_total"

ALLOWED_ATTRIBUTE_KEYS = {
    "gen_ai.system",
    "gen_ai.request.model",
    "gen_ai.response.model",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "governance.error_category",
    "governance.escalation_reason",
    "governance.estimated_cost_usd",
    "governance.fallback_used",
    "governance.human_review_required",
    "governance.policy_action",
    "governance.policy_ids",
    "governance.provider",
    "governance.redaction_count",
    "governance.retry_count",
    "governance.risk_tier",
    "governance.telemetry_boundary",
    "governance.trace_id",
    "governance.workflow_type",
}

METRIC_NAMES = {
    METRIC_REQUEST_COUNT,
    METRIC_ERROR_COUNT,
    METRIC_PROVIDER_LATENCY_MS,
    METRIC_RETRY_COUNT,
    METRIC_FALLBACK_COUNT,
    METRIC_ESTIMATED_COST_USD,
    METRIC_RISK_TIER_DISTRIBUTION,
    METRIC_REDACTION_COUNT_DISTRIBUTION,
    METRIC_POLICY_VIOLATION_ATTEMPTS,
    METRIC_HUMAN_ESCALATIONS_TOTAL,
}

_ENABLED_VALUES = {"1", "true", "yes", "on"}
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_SAFE_METRIC_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_BARE_DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_PERSON_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b")
_MAX_ATTRIBUTE_STRING_LENGTH = 200
# This is a best-effort telemetry leak guard, not a name detector. The allowlist
# keeps common TitleCase governance phrases readable while the docs retain the
# MVP limitation that bare-name handling is heuristic.
_SAFE_PERSON_PHRASES = {
    "Clinical Operations",
    "Human Review",
    "Open Telemetry",
    "Prior Authorization",
    "Provider Access",
    "Responsible AI",
}


@dataclass(frozen=True)
class TelemetrySettings:
    enabled: bool = False
    capture_spans: bool = False
    capture_metrics: bool = False
    service_name: str = DEFAULT_SERVICE_NAME
    use_otel: bool = True


@dataclass
class RecordedSpan:
    name: str
    attributes: dict[str, Any]
    duration_ms: float
    status: str = "ok"
    error_category: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "attributes": dict(self.attributes),
            "duration_ms": self.duration_ms,
            "status": self.status,
        }
        if self.error_category:
            result["error_category"] = self.error_category
        return result


@dataclass
class RecordedMetric:
    name: str
    value: int | float
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "attributes": dict(self.attributes),
        }


class _LocalTelemetryState:
    def __init__(self) -> None:
        self._lock = RLock()
        self._spans: list[RecordedSpan] = []
        self._metrics: list[RecordedMetric] = []

    def record_span(self, span: RecordedSpan) -> None:
        with self._lock:
            self._spans.append(span)

    def record_metric(self, metric: RecordedMetric) -> None:
        with self._lock:
            self._metrics.append(metric)

    def spans(self) -> list[dict[str, Any]]:
        with self._lock:
            return [span.to_dict() for span in self._spans]

    def metrics(self) -> list[dict[str, Any]]:
        with self._lock:
            return [metric.to_dict() for metric in self._metrics]

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()
            self._metrics.clear()


class _OpenTelemetryRuntime:
    def __init__(self, settings: TelemetrySettings):
        self.available = False
        self.tracer = None
        self.meter = None
        self._metric_instruments: dict[str, Any] = {}

        try:
            from opentelemetry import metrics, trace
        except Exception:
            return

        self.available = True
        _configure_otel_provider_if_requested(settings)
        self.tracer = trace.get_tracer("multi_model_agent.telemetry")
        self.meter = metrics.get_meter("multi_model_agent.telemetry")

    def start_span(self, name: str, attributes: dict[str, Any]):
        if self.tracer is None:
            return None
        try:
            return self.tracer.start_as_current_span(name, attributes=attributes)
        except Exception:
            return None

    def record_metric(
        self,
        name: str,
        value: int | float,
        attributes: dict[str, Any],
    ) -> None:
        if self.meter is None:
            return

        try:
            instrument = self._metric_instruments.get(name)
            if instrument is None:
                if name == METRIC_PROVIDER_LATENCY_MS:
                    instrument = self.meter.create_histogram(name, unit="ms")
                elif name == METRIC_ESTIMATED_COST_USD:
                    instrument = self.meter.create_histogram(name, unit="USD")
                else:
                    instrument = self.meter.create_counter(name)
                self._metric_instruments[name] = instrument

            if hasattr(instrument, "record"):
                instrument.record(value, attributes=attributes)
            elif hasattr(instrument, "add"):
                instrument.add(value, attributes=attributes)
        except Exception:
            return


class GovernanceSpan:
    def __init__(self, name: str, attributes: dict[str, Any] | None = None):
        self.name = name
        self.attributes = (
            sanitize_telemetry_attributes(attributes or {})
            if _span_export_active()
            else {}
        )
        self._start = 0.0
        self._otel_context_manager = None
        self._otel_span = None

    def __enter__(self) -> "GovernanceSpan":
        self._start = time.perf_counter()
        if _settings.enabled and _settings.use_otel:
            runtime = _otel_runtime()
            if runtime is not None:
                self._otel_context_manager = runtime.start_span(
                    self.name,
                    self.attributes,
                )
                if self._otel_context_manager is not None:
                    try:
                        self._otel_span = self._otel_context_manager.__enter__()
                    except Exception:
                        self._otel_context_manager = None
                        self._otel_span = None
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        status = "ok"
        error_category = None
        if exc is not None:
            status = "error"
            error_category = _safe_error_category(exc)
            self.set_attribute("governance.error_category", error_category)

        if self._otel_context_manager is not None:
            try:
                self._otel_context_manager.__exit__(exc_type, exc, traceback)
            except Exception:
                pass

        if _settings.capture_spans:
            duration_ms = max((time.perf_counter() - self._start) * 1000, 0.0)
            _local_state.record_span(
                RecordedSpan(
                    name=self.name,
                    attributes=dict(self.attributes),
                    duration_ms=duration_ms,
                    status=status,
                    error_category=error_category,
                )
            )
        return False

    def set_attribute(self, key: str, value: Any) -> None:
        if not _span_export_active():
            return
        safe = sanitize_telemetry_attributes({key: value})
        if not safe:
            return
        for safe_key, safe_value in safe.items():
            self.attributes[safe_key] = safe_value
            if self._otel_span is not None:
                try:
                    self._otel_span.set_attribute(safe_key, safe_value)
                except Exception:
                    pass


_local_state = _LocalTelemetryState()
_settings = TelemetrySettings()
_settings_lock = RLock()
_runtime: _OpenTelemetryRuntime | None = None


def configure_telemetry(
    *,
    enabled: bool | None = None,
    capture_spans: bool | None = None,
    capture_metrics: bool | None = None,
    service_name: str | None = None,
    use_otel: bool | None = None,
) -> TelemetrySettings:
    """Configure optional telemetry without making OTel a hard dependency."""
    global _settings, _runtime
    with _settings_lock:
        _settings = TelemetrySettings(
            enabled=_env_flag(TELEMETRY_ENABLED_ENV)
            if enabled is None
            else enabled,
            capture_spans=_env_flag(TELEMETRY_CAPTURE_SPANS_ENV)
            if capture_spans is None
            else capture_spans,
            capture_metrics=_env_flag(TELEMETRY_CAPTURE_METRICS_ENV)
            if capture_metrics is None
            else capture_metrics,
            service_name=service_name
            or os.getenv(TELEMETRY_SERVICE_NAME_ENV)
            or DEFAULT_SERVICE_NAME,
            use_otel=True if use_otel is None else use_otel,
        )
        _runtime = None
        return _settings


def reset_telemetry() -> TelemetrySettings:
    _local_state.clear()
    return configure_telemetry()


def clear_recorded_telemetry() -> None:
    _local_state.clear()


def get_recorded_spans() -> list[dict[str, Any]]:
    return _local_state.spans()


def get_recorded_metrics() -> list[dict[str, Any]]:
    return _local_state.metrics()


def governance_span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> GovernanceSpan:
    return GovernanceSpan(name=name, attributes=attributes)


def provider_call_span_name(provider: str) -> str:
    safe_provider = _safe_identifier(str(provider), label="provider")
    return SPAN_PROVIDER_CALL_TEMPLATE.format(provider=safe_provider)


def record_governance_metric(
    name: str,
    value: int | float = 1,
    attributes: dict[str, Any] | None = None,
) -> None:
    if name not in METRIC_NAMES or not _SAFE_METRIC_NAME_RE.fullmatch(name):
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return
    if not _metric_export_active():
        return

    safe_attributes = sanitize_telemetry_attributes(attributes or {})
    if _settings.capture_metrics:
        _local_state.record_metric(
            RecordedMetric(
                name=name,
                value=value,
                attributes=safe_attributes,
            )
        )

    if _settings.enabled and _settings.use_otel:
        runtime = _otel_runtime()
        if runtime is not None:
            runtime.record_metric(name, value, safe_attributes)


def sanitize_telemetry_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in attributes.items():
        safe_key = str(key)
        if safe_key not in ALLOWED_ATTRIBUTE_KEYS:
            continue

        safe_value = _sanitize_attribute_value(safe_key, value)
        if safe_value is not None:
            result[safe_key] = safe_value
    return result


def _sanitize_attribute_value(key: str, value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, Enum):
        value = value.value

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value == value and value not in {float("inf"), float("-inf")}:
            return value
        return None

    if isinstance(value, str):
        if key == "governance.trace_id":
            return _safe_identifier(value, label="trace")
        return _sanitize_attribute_text(value)

    if isinstance(value, set):
        iterable = sorted(value, key=lambda item: str(item))
    elif isinstance(value, (list, tuple)):
        iterable = value
    else:
        return None

    sanitized: list[Any] = []
    for item in iterable:
        safe_item = _sanitize_attribute_value(key, item)
        if isinstance(safe_item, (str, bool, int, float)):
            sanitized.append(safe_item)
    return sanitized if sanitized else None


def _sanitize_attribute_text(value: str) -> str:
    redacted = redact_sensitive_data(value).redacted_text
    redacted = _BARE_DATE_RE.sub("[DATE]", redacted)
    redacted = _PERSON_NAME_RE.sub(_replace_person_name, redacted)
    if len(redacted) > _MAX_ATTRIBUTE_STRING_LENGTH:
        return redacted[: _MAX_ATTRIBUTE_STRING_LENGTH - 3] + "..."
    return redacted


def _replace_person_name(match: re.Match[str]) -> str:
    phrase = match.group(0)
    if phrase in _SAFE_PERSON_PHRASES:
        return phrase
    if phrase.startswith("["):
        return phrase
    return "[PERSON]"


def _safe_identifier(value: str, *, label: str) -> str:
    if _SAFE_IDENTIFIER_RE.fullmatch(value):
        return value
    import hashlib

    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{label}:{digest}"


def _safe_error_category(error: BaseException) -> str:
    try:
        from .reliability import classify_error

        category = classify_error(error)
    except Exception:
        return "provider_error"

    return {
        "retry": "retryable_provider_error",
        "fallback": "provider_fallback_error",
        "fail": "provider_configuration_error",
    }.get(category, "provider_error")


def _span_export_active() -> bool:
    return _settings.capture_spans or (_settings.enabled and _settings.use_otel)


def _metric_export_active() -> bool:
    return _settings.capture_metrics or (_settings.enabled and _settings.use_otel)


def _otel_runtime() -> _OpenTelemetryRuntime | None:
    global _runtime
    with _settings_lock:
        if _runtime is None:
            _runtime = _OpenTelemetryRuntime(_settings)
        return _runtime if _runtime.available else None


def _configure_otel_provider_if_requested(settings: TelemetrySettings) -> None:
    if not _env_flag(TELEMETRY_AUTO_CONFIGURE_ENV):
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception:
        return

    try:
        current_provider = trace.get_tracer_provider()
        if current_provider.__class__.__name__ != "ProxyTracerProvider":
            return

        provider = TracerProvider(
            resource=Resource.create({"service.name": settings.service_name})
        )
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
    except Exception:
        return


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _ENABLED_VALUES


configure_telemetry()
