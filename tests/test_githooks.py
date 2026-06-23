import unittest

from scripts.check_staged_audit_logs import (
    blocked_staged_audit_artifacts,
    looks_like_audit_log_content,
)
from scripts.check_staged_docs import (
    REQUIRED_GOVERNANCE_DOCS,
    missing_required_docs_for_staged_changes,
)


class GitHookGuardTests(unittest.TestCase):
    def test_content_signature_detects_audit_log_outside_default_path(self):
        audit_line = (
            b'{"event_hash":"abc","payload_hash":"def",'
            b'"schema_version":"audit.v1","payload":{}}\n'
        )

        self.assertTrue(looks_like_audit_log_content(audit_line))

    def test_content_signature_detects_pretty_printed_audit_json(self):
        audit_json = b"""
        {
          "schema_version": "audit.v1",
          "payload_hash": "abc",
          "event_hash": "def"
        }
        """

        self.assertTrue(looks_like_audit_log_content(audit_json))

    def test_content_signature_detects_audit_json_array(self):
        audit_json = b"""
        [
          {
            "schema_version": "audit.v1",
            "payload_hash": "abc",
            "event_hash": "def"
          }
        ]
        """

        self.assertTrue(looks_like_audit_log_content(audit_json))

    def test_markdown_schema_documentation_is_allowed(self):
        markdown = b"""
        # Stored Event Schema

        Each persisted event should include fields named:

        ```json
        {
          "schema_version": "audit.v1",
          "payload_hash": "sha256",
          "event_hash": "sha256"
        }
        ```

        This is documentation, not an audit log artifact.
        """

        self.assertFalse(looks_like_audit_log_content(markdown))

    def test_non_audit_json_is_allowed(self):
        self.assertFalse(looks_like_audit_log_content(b'{"schema_version":"other"}'))

    def test_default_audit_path_is_blocked_even_without_content(self):
        blocked = blocked_staged_audit_artifacts(["audit_logs/custom.jsonl"])

        self.assertEqual(blocked, [("audit_logs/custom.jsonl", "default audit log path")])

    def test_governance_code_changes_require_a_documentation_update(self):
        missing = missing_required_docs_for_staged_changes(
            ["multi_model_agent/evidence_coverage.py"]
        )

        self.assertEqual(missing, list(REQUIRED_GOVERNANCE_DOCS))

    def test_governance_code_changes_pass_when_one_reviewer_doc_is_staged(self):
        missing = missing_required_docs_for_staged_changes(
            ["multi_model_agent/evidence_coverage.py", "docs/architecture.md"]
        )

        self.assertEqual(missing, [])

    def test_test_only_commits_do_not_require_documentation_updates(self):
        missing = missing_required_docs_for_staged_changes(["tests/test_githooks.py"])

        self.assertEqual(missing, [])

    def test_docs_only_commits_do_not_require_every_governance_doc(self):
        missing = missing_required_docs_for_staged_changes(["README.md"])

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
