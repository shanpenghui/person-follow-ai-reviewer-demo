from pathlib import Path
import unittest

from ai_review_bot.analyzer import analyze
from ai_review_bot.diff_parser import read_diff


class AnalyzerTest(unittest.TestCase):
    def test_analyzer_flags_robotics_review_domains(self) -> None:
        report = analyze(Path("sample_repo"), read_diff(Path("examples/risky_person_follow.patch")))

        rule_ids = {finding.rule.rule_id for finding in report.findings}
        self.assertIn("motion-safety", rule_ids)
        self.assertIn("action-contract", rule_ids)
        self.assertEqual(report.risk_level, "高")
        self.assertFalse(report.tests_changed)


if __name__ == "__main__":
    unittest.main()
