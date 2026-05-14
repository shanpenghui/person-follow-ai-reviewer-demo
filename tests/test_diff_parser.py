from pathlib import Path
import unittest

from ai_review_bot.diff_parser import read_diff


class DiffParserTest(unittest.TestCase):
    def test_parse_changed_lines_from_fixture(self) -> None:
        changed = read_diff(Path("examples/risky_person_follow.patch"))

        self.assertEqual(
            [item.path for item in changed],
            [
                "scene_servo/person_yolo_servo_node.py",
                "person_follow/follow_action_server_node.py",
            ],
        )
        self.assertTrue(changed[0].changed_lines)
        self.assertTrue(changed[1].changed_lines)


if __name__ == "__main__":
    unittest.main()
