import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvalScriptTests(unittest.TestCase):
    def test_redteam_eval_runs_expanded_deterministic_suite(self):
        cases_path = ROOT / "evals" / "redteam" / "prior_auth_redteam_cases.json"
        report_path = ROOT / "evals" / "redteam" / "prior_auth_redteam_report.md"
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
        categories = {case["category"] for case in cases}

        first = _run_command(["scripts/run_redteam_eval.py"])
        second = _run_command(["scripts/run_redteam_eval.py"])

        self.assertGreaterEqual(len(cases), 30)
        self.assertGreaterEqual(len(categories), 12)
        self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
        self.assertEqual(second.returncode, 0, second.stderr + second.stdout)
        self.assertEqual(first.stdout, second.stdout)
        self.assertIn("prior-auth red-team eval passed: 36/36", first.stdout)
        report = report_path.read_text(encoding="utf-8")
        for field in (
            "total_cases",
            "passed_cases",
            "failed_cases",
            "pass_rate",
            "failures_by_category",
            "redaction_precision",
            "redaction_recall",
            "max_provider_call_count",
            "fanout_limited_cases",
            "Limitations",
            "Regression Notes",
        ):
            self.assertIn(field, report)
        self.assertIn("max_provider_call_count: 1", report)

    def test_redteam_helpers_detect_recursive_phi_and_prompt_leaks(self):
        module = _load_module(
            "run_redteam_eval",
            ROOT / "scripts" / "run_redteam_eval.py",
        )

        self.assertTrue(
            module._contains_raw_value(
                {"payload": {"details": ["reviewed Jane Doe safely"]}},
                "Jane Doe",
            )
        )
        self.assertTrue(
            module._contains_prompt_leak(
                {"payload": {"details": "developer-message content leaked"}}
            )
        )
        self.assertFalse(
            module._contains_prompt_leak(
                {"payload": {"policy_ids": ["approved_provider_egress_only"]}}
            )
        )

    def test_privacy_redaction_benchmark_runs_with_realistic_gates(self):
        report_path = ROOT / "evals" / "privacy" / "redaction_benchmark_report.md"

        result = _run_command(["evals/privacy/run_redaction_benchmark.py"])

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        summary = json.loads(result.stdout)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["total_cases"], 19)
        self.assertEqual(summary["passed_cases"], 17)
        self.assertEqual(summary["failed_cases"], 2)
        self.assertEqual(summary["gated_identifier_types_failed"], 0)
        self.assertEqual(
            summary["failures_by_category"],
            {"bare_date": 1, "bare_name": 1},
        )
        for kind in ("email", "phone", "ssn", "member_id"):
            self.assertTrue(summary["gated_thresholds"][kind]["passed"])
            self.assertGreaterEqual(
                summary["gated_thresholds"][kind]["recall"],
                summary["gated_thresholds"][kind]["threshold"],
            )
        self.assertEqual(
            summary["metrics_by_identifier_type"]["bare_name"]["recall"],
            0.0,
        )
        self.assertEqual(
            summary["metrics_by_identifier_type"]["bare_date"]["recall"],
            0.0,
        )
        report = report_path.read_text(encoding="utf-8")
        self.assertIn("Bare-name and bare-date recall are tracked", report)
        self.assertIn("gate_passed: True", report)
        self.assertIn('failures_by_category: {"bare_date": 1, "bare_name": 1}', report)
        self.assertIn("Case pass/fail counts are entity-level", report)
        self.assertIn("redaction_precision", report)
        self.assertIn("redaction_recall", report)

    def test_privacy_matching_prefers_closest_same_kind_prediction(self):
        module = _load_module(
            "run_redaction_benchmark",
            ROOT / "evals" / "privacy" / "run_redaction_benchmark.py",
        )
        predictions = [
            {"kind": "member_id", "value": "Member ID ABC123456"},
            {"kind": "member_id", "value": "ABC123456"},
        ]

        match_index = module._matching_prediction_index(
            label={"kind": "member_id", "value": "ABC123456"},
            predictions=predictions,
            matched_predictions=set(),
        )

        self.assertEqual(match_index, 1)

    def test_invariance_eval_runs_and_generates_report(self):
        report_path = ROOT / "evals" / "fairness" / "invariance_report.md"

        result = _run_command(["evals/fairness/run_invariance_eval.py"])

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        summary = json.loads(result.stdout)
        self.assertTrue(all(result["passed"] for result in summary["results"]))
        report = report_path.read_text(encoding="utf-8")
        for field in (
            "total_cases",
            "passed_cases",
            "failed_cases",
            "pass_rate",
            "failures_by_category",
            "redaction_precision",
            "redaction_recall",
        ):
            self.assertIn(field, report)

    def test_invariance_signature_compares_structured_statuses_not_free_text(self):
        module = _load_module(
            "run_invariance_eval",
            ROOT / "evals" / "fairness" / "run_invariance_eval.py",
        )

        signature = module.build_structured_signature(
            (
                "Prior authorization request for MRI. Diagnosis: radiculopathy. "
                "Clinical rationale: persistent symptoms. Provider notes: office visit."
            ),
            trace_id="test-invariance-signature",
        )
        serialized = json.dumps(signature, sort_keys=True)

        self.assertIn("items", signature)
        self.assertNotIn("human_rationale", serialized)
        self.assertNotIn("overall_summary", serialized)
        for item in signature["items"]:
            self.assertEqual(set(item), {"requirement_id", "status"})


if __name__ == "__main__":
    unittest.main()
