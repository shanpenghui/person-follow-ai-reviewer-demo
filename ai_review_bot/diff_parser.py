from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


HUNK_RE = re.compile(r"@@ -(?P<old_start>\d+)(?:,\d+)? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@")


@dataclass(frozen=True)
class Hunk:
    old_start: int
    new_start: int
    new_lines: tuple[int, ...]


@dataclass
class ChangedFile:
    path: str
    status: str = "modified"
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def changed_lines(self) -> set[int]:
        lines: set[int] = set()
        for hunk in self.hunks:
            lines.update(hunk.new_lines)
        return lines


def _normalize_path(raw: str) -> str:
    path = raw.strip()
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return path.replace("\\", "/")


def parse_unified_diff(diff_text: str) -> list[ChangedFile]:
    """Parse enough unified diff metadata for PR risk review.

    The parser intentionally focuses on the new-side file path and changed line
    numbers. It does not try to model every git patch edge case because the demo
    only needs reliable context retrieval and rule matching.
    """
    files: list[ChangedFile] = []
    current_file: ChangedFile | None = None
    current_new_line = 0
    current_old_line = 0
    active_new_lines: list[int] | None = None
    active_old_start = 0
    active_new_start = 0

    def close_hunk() -> None:
        nonlocal active_new_lines, active_old_start, active_new_start
        if current_file is not None and active_new_lines is not None:
            current_file.hunks.append(
                Hunk(
                    old_start=active_old_start,
                    new_start=active_new_start,
                    new_lines=tuple(active_new_lines),
                )
            )
        active_new_lines = None

    for raw_line in diff_text.splitlines():
        if raw_line.startswith("diff --git "):
            close_hunk()
            parts = raw_line.split()
            path = _normalize_path(parts[-1]) if len(parts) >= 4 else "unknown"
            current_file = ChangedFile(path=path)
            files.append(current_file)
            continue

        if current_file is None:
            continue

        if raw_line.startswith("new file mode"):
            current_file.status = "added"
            continue
        if raw_line.startswith("deleted file mode"):
            current_file.status = "deleted"
            continue
        if raw_line.startswith("+++ "):
            new_path = raw_line[4:].strip()
            if new_path != "/dev/null":
                current_file.path = _normalize_path(new_path)
            continue

        match = HUNK_RE.match(raw_line)
        if match:
            close_hunk()
            active_old_start = int(match.group("old_start"))
            active_new_start = int(match.group("new_start"))
            current_old_line = active_old_start
            current_new_line = active_new_start
            active_new_lines = []
            continue

        if active_new_lines is None:
            continue

        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            active_new_lines.append(current_new_line)
            current_new_line += 1
        elif raw_line.startswith("-") and not raw_line.startswith("---"):
            current_old_line += 1
        else:
            current_old_line += 1
            current_new_line += 1

    close_hunk()
    return files


def read_diff(path: Path) -> list[ChangedFile]:
    return parse_unified_diff(path.read_text(encoding="utf-8"))

