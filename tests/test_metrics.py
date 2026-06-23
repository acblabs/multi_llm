import unittest
from concurrent.futures import ThreadPoolExecutor

from multi_model_agent.metrics import clear_usage_log, get_usage_summary, log_usage


class MetricsTests(unittest.TestCase):
    def setUp(self):
        clear_usage_log()

    def tearDown(self):
        clear_usage_log()

    def test_usage_summary_is_ephemeral_and_returns_copy(self):
        log_usage("openai", 10)

        summary = get_usage_summary()
        summary["calls"].append({"provider": "openai", "tokens": 999, "cost": 999})

        fresh_summary = get_usage_summary()

        self.assertTrue(fresh_summary["ephemeral"])
        self.assertFalse(fresh_summary["audit_evidence"])
        self.assertEqual(fresh_summary["total_tokens"], 10)
        self.assertEqual(len(fresh_summary["calls"]), 1)

    def test_usage_log_sanitizes_provider_and_token_count(self):
        message = log_usage("patient-jane-doe", -7)

        summary = get_usage_summary()

        self.assertIn("[unknown: 0 tokens", message)
        self.assertEqual(summary["calls"][0]["provider"], "unknown")
        self.assertEqual(summary["calls"][0]["tokens"], 0)

    def test_usage_log_is_thread_safe_for_local_summary(self):
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(lambda _: log_usage("openai", 1), range(50)))

        summary = get_usage_summary()

        self.assertEqual(summary["total_tokens"], 50)
        self.assertEqual(len(summary["calls"]), 50)


if __name__ == "__main__":
    unittest.main()
