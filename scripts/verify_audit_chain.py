import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_model_agent.audit_store import JsonlAuditStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a PHI-safe JSONL audit hash chain."
    )
    parser.add_argument("path", help="Path to the JSONL audit log to verify")
    args = parser.parse_args()

    result = JsonlAuditStore(args.path).verify_chain()
    print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
