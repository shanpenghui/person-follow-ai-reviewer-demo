from __future__ import annotations

from .analyzer import ReviewReport, build_llm_prompt


def render_markdown(report: ReviewReport, include_prompt: bool = False) -> str:
    changed = ", ".join(file.path for file in report.changed_files) or "无"
    lines: list[str] = [
        "# AI PR Review Report",
        "",
        f"- 被分析仓库: `{report.repo_root}`",
        f"- 变更文件: {changed}",
        f"- 综合风险等级: **{report.risk_level}**",
        f"- 是否包含测试变更: {'是' if report.tests_changed else '否'}",
        "",
    ]

    if not report.findings:
        lines.extend(
            [
                "## 结论",
                "",
                "没有命中机器人跟随场景的高风险规则。建议仍由 reviewer 检查业务语义和实机验证范围。",
            ]
        )
    else:
        lines.extend(["## 风险发现", ""])
        for index, finding in enumerate(report.findings, start=1):
            functions = sorted({f"{fn.file_path}:{fn.start_line} `{fn.qualified_name}`" for fn in finding.functions})
            lines.extend(
                [
                    f"### {index}. [{finding.severity.upper()}] {finding.rule.title}",
                    "",
                    f"影响文件: {', '.join(sorted(finding.files))}",
                    "",
                    f"为什么重要: {finding.rule.why_it_matters}",
                    "",
                    f"Review 重点: {finding.rule.review_prompt}",
                    "",
                ]
            )
            if functions:
                lines.append("定位到的函数:")
                lines.extend(f"- {item}" for item in functions[:8])
                lines.append("")
            lines.append("建议补充测试:")
            lines.extend(f"- {item}" for item in finding.rule.test_suggestions)
            lines.append("")

    if not report.tests_changed and report.findings:
        lines.extend(
            [
                "## 缺失测试提醒",
                "",
                "本次 patch 未包含测试文件变更。对运动控制、深度距离、手势 FSM 或 Action 契约的改动，建议至少补一组离线单测或回放用例。",
                "",
            ]
        )

    lines.extend(
        [
            "## 面试讲法",
            "",
            "这个 demo 展示的是 AI 不是直接替代 reviewer，而是在 PR 阶段先用静态分析和项目规则圈定高风险上下文，再把有限上下文交给 LLM 生成可执行的 review 建议和测试清单。",
            "",
        ]
    )

    if include_prompt:
        lines.extend(["## LLM Prompt", "", "```text", build_llm_prompt(report), "```", ""])

    return "\n".join(lines)

