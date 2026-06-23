import subprocess
import sys


REQUIRED_GOVERNANCE_DOCS = (
    "README.md",
    "docs/architecture.md",
    "examples/prior_authorization/governance_walkthrough.md",
    "governance/system_card.md",
    "governance/model_risk_tiering.md",
    "governance/ai_impact_assessment.md",
)

DOC_UPDATE_TRIGGER_PREFIXES = (
    ".githooks/",
    "deployment/",
    "evals/",
    "multi_model_agent/",
    "scripts/",
)
DOC_UPDATE_TRIGGER_FILES = (
    "cloudbuild.yaml",
)


def main() -> int:
    staged_paths = _staged_paths()
    if staged_paths is None:
        return 1

    missing = missing_required_docs_for_staged_changes(staged_paths)
    if not missing:
        return 0

    sys.stderr.write(
        "Governance or implementation changes require documentation updates.\n"
        "Stage at least one of these reviewer-facing docs, or split the commit "
        "so code and docs land together:\n"
    )
    for path in missing:
        sys.stderr.write(f"  - {path}\n")
    return 1


def missing_required_docs_for_staged_changes(paths: list[str]) -> list[str]:
    normalized_paths = {_normalize_path(path) for path in paths if path}
    if not _has_documentation_trigger(normalized_paths):
        return []
    if normalized_paths.intersection(REQUIRED_GOVERNANCE_DOCS):
        return []

    return list(REQUIRED_GOVERNANCE_DOCS)


def _has_documentation_trigger(paths: set[str]) -> bool:
    return any(_is_documentation_trigger(path) for path in paths)


def _is_documentation_trigger(path: str) -> bool:
    if path in REQUIRED_GOVERNANCE_DOCS:
        return False
    if path in DOC_UPDATE_TRIGGER_FILES:
        return True
    return any(path.startswith(prefix) for prefix in DOC_UPDATE_TRIGGER_PREFIXES)


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip()


def _staged_paths() -> list[str] | None:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return None
    return [path for path in result.stdout.splitlines() if path]


if __name__ == "__main__":
    raise SystemExit(main())
