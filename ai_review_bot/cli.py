from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from .analyzer import analyze
from .diff_parser import parse_unified_diff
from .reporter import render_markdown


def _git_diff(repo_root: Path, base: str | None) -> str:
    cmd = ["git", "-C", str(repo_root), "diff", "--unified=80"]
    if base:
        cmd.extend([base, "--"])
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    return result.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI-assisted PR review demo for person_follow.")
    parser.add_argument("--repo", type=Path, default=Path("sample_repo"), help="Repository root to analyze.")
    parser.add_argument("--diff", type=Path, help="Unified diff file. If omitted, git diff is used.")
    parser.add_argument("--base", help="Base ref for git diff, for example origin/main.")
    parser.add_argument("--output", type=Path, help="Write markdown report to this path.")
    parser.add_argument("--include-prompt", action="store_true", help="Append the LLM prompt used for final reasoning.")
    args = parser.parse_args(argv)

    repo_root = args.repo.resolve()
    if args.diff:
        diff_text = args.diff.read_text(encoding="utf-8")
    else:
        diff_text = _git_diff(repo_root, args.base)

    changed_files = parse_unified_diff(diff_text)
    if not changed_files:
        print("No changed files found in diff.", file=sys.stderr)
        return 1

    report = analyze(repo_root, changed_files)
    markdown = render_markdown(report, include_prompt=args.include_prompt)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote report to {args.output}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

