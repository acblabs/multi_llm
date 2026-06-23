import unittest

from scripts.check_staged_audit_logs import (
    blocked_staged_audit_artifacts,
    looks_like_audit_log_content,
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


if __name__ == "__main__":
    unittest.main()
