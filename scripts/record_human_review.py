import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_model_agent.audit import set_audit_store  # noqa: E402
from multi_model_agent.audit_store import JsonlAuditStore  # noqa: E402
from multi_model_agent.review import (  # noqa: E402
    ReviewConfigurationError,
    record_human_review_decision,
    resolve_trace_state,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record a sanitized human-review decision in the audit log."
    )
    parser.add_argument("--trace-id", required=True, help="Trace ID to close")
    parser.add_argument(
        "--reviewer-id",
        required=True,
        help="Raw reviewer ID; HMACed before persistence and never printed",
    )
    parser.add_argument(
        "--reviewer-role",
        default="clinical_operations",
        help="Reviewer role, not raw identity",
    )
    parser.add_argument(
        "--decision",
        required=True,
        choices=["accepted", "modified", "rejected", "escalated"],
        help="Final human review decision",
    )
    parser.add_argument("--rationale", required=True, help="Reviewer rationale")
    parser.add_argument(
        "--audit-log",
        help="Optional JSONL audit log path. Defaults to the configured audit sink.",
    )
    args = parser.parse_args()

    if args.audit_log:
        set_audit_store(JsonlAuditStore(args.audit_log))

    try:
        decision = record_human_review_decision(
            trace_id=args.trace_id,
            reviewer_id=args.reviewer_id,
            reviewer_role=args.reviewer_role,
            decision=args.decision,
            rationale=args.rationale,
        )
        trace_state = resolve_trace_state(args.trace_id)
    except ReviewConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "human_review_decision": decision.to_safe_dict(),
                "trace_state": trace_state.model_dump(mode="json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
