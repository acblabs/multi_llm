import json
import subprocess
import sys


AUDIT_LOG_PATH_PREFIX = "audit_logs/"
AUDIT_LOG_KEEPER = "audit_logs/.gitkeep"
AUDIT_SIGNATURE_KEYS = {"schema_version", "payload_hash", "event_hash"}


def main() -> int:
    staged_paths = _staged_paths()
    if staged_paths is None:
        return 1

    blocked = blocked_staged_audit_artifacts(staged_paths)
    if not blocked:
        return 0

    sys.stderr.write(
        "Refusing to commit generated audit log output. "
        "Audit logs may contain regulated metadata and should stay local.\n"
    )
    for path, reason in blocked:
        sys.stderr.write(f"  - {path}: {reason}\n")
    return 1


def blocked_staged_audit_artifacts(paths: list[str]) -> list[tuple[str, str]]:
    blocked: list[tuple[str, str]] = []
    for path in paths:
        if path.startswith(AUDIT_LOG_PATH_PREFIX) and path != AUDIT_LOG_KEEPER:
            blocked.append((path, "default audit log path"))
            continue

        content = _staged_file_content(path)
        if content is None:
            continue

        if looks_like_audit_log_content(content):
            blocked.append((path, "audit.v1 JSONL content signature"))

    return blocked


def looks_like_audit_log_content(content: bytes) -> bool:
    if not content or b"\x00" in content:
        return False

    text = content.decode("utf-8", errors="ignore")
    stripped_text = text.strip()
    if not stripped_text:
        return False

    try:
        candidate = json.loads(stripped_text)
    except json.JSONDecodeError:
        candidate = None
    if _contains_audit_event_object(candidate):
        return True

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            candidate = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if _is_audit_event_object(candidate):
            return True

    return False


def _contains_audit_event_object(value: object) -> bool:
    if _is_audit_event_object(value):
        return True
    if isinstance(value, list):
        return any(_is_audit_event_object(item) for item in value)
    return False


def _is_audit_event_object(value: object) -> bool:
    return (
        isinstance(value, dict)
        and value.get("schema_version") == "audit.v1"
        and AUDIT_SIGNATURE_KEYS.issubset(value.keys())
    )


def _staged_paths() -> list[str] | None:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return None
    return [path for path in result.stdout.splitlines() if path]


def _staged_file_content(path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f":{path}"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


if __name__ == "__main__":
    raise SystemExit(main())
