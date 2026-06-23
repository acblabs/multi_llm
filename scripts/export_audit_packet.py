import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_model_agent.evidence_packet import (  # noqa: E402
    DEFAULT_EVIDENCE_PACKET_ROOT,
    export_evidence_packet,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a PHI-safe reviewer evidence packet for one audit trace."
    )
    parser.add_argument("--trace-id", required=True, help="Trace ID to export")
    parser.add_argument(
        "--audit-log",
        required=True,
        help="Path to the JSONL audit log that contains the trace",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_EVIDENCE_PACKET_ROOT),
        help="Root directory where TRACE_ID packet folders are written",
    )
    args = parser.parse_args()

    try:
        result = export_evidence_packet(
            trace_id=args.trace_id,
            audit_log=args.audit_log,
            output_root=args.output_dir,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.audit_chain_verification.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
