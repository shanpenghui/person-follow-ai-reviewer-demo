from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .context import FunctionContext, locate_changed_functions
from .diff_parser import ChangedFile
from .rules import RULES, ReviewRule, is_test_path


@dataclass
class Finding:
    rule: ReviewRule
    files: set[str] = field(default_factory=set)
    functions: list[FunctionContext] = field(default_factory=list)

    @property
    def severity(self) -> str:
        return self.rule.severity


@dataclass
class ReviewReport:
    repo_root: Path
    changed_files: list[ChangedFile]
    findings: list[Finding]
    tests_changed: bool

    @property
    def risk_level(self) -> str:
        if any(f.severity == "high" for f in self.findings):
            return "高"
        if any(f.severity == "medium" for f in self.findings):
            return "中"
        return "低"


def _rule_matches(rule: ReviewRule, changed_file: ChangedFile, functions: list[FunctionContext]) -> bool:
    path = changed_file.path.lower()
    path_hit = any(keyword.lower() in path for keyword in rule.path_keywords)
    code_blob = "\n".join(fn.source for fn in functions).lower()
    code_hit = any(keyword.lower() in code_blob for keyword in rule.code_keywords)
    return path_hit or code_hit


def analyze(repo_root: Path, changed_files: list[ChangedFile]) -> ReviewReport:
    findings_by_rule: dict[str, Finding] = {}
    tests_changed = any(is_test_path(file.path) for file in changed_files)

    for changed_file in changed_files:
        if changed_file.status == "deleted":
            continue
        functions = locate_changed_functions(repo_root, changed_file)
        for rule in RULES:
            if _rule_matches(rule, changed_file, functions):
                finding = findings_by_rule.setdefault(rule.rule_id, Finding(rule=rule))
                finding.files.add(changed_file.path)
                finding.functions.extend(functions)

    return ReviewReport(
        repo_root=repo_root,
        changed_files=changed_files,
        findings=sorted(
            findings_by_rule.values(),
            key=lambda item: {"high": 0, "medium": 1, "low": 2}.get(item.severity, 3),
        ),
        tests_changed=tests_changed,
    )


def build_llm_prompt(report: ReviewReport) -> str:
    context_lines: list[str] = []
    for finding in report.findings:
        context_lines.append(f"## {finding.rule.title}")
        for fn in finding.functions[:3]:
            snippet = "\n".join(fn.source.splitlines()[:80])
            context_lines.append(
                f"文件: {fn.file_path}:{fn.start_line}\n函数: {fn.qualified_name}\n```python\n{snippet}\n```"
            )

    return "\n\n".join(
        [
            "你是机器人研发团队的 AI PR Reviewer。只基于下面的代码上下文给出风险、依据和测试建议。",
            f"整体风险等级: {report.risk_level}",
            "关注领域: 人体/物体跟随、D455 深度、YOLO 目标选择、手势 FSM、Action Server、底盘和头部运动安全。",
            "\n".join(context_lines) or "没有定位到函数上下文，请只做保守建议。",
            "输出格式: 风险摘要、逐条发现、建议测试、需要人工确认的问题。",
        ]
    )

