# 面试讲解手卡

## 一句话项目

我基于 `person_follow` 机器人跟随模块做了一个 AI PR Review 助手。它会在 PR 阶段自动读取 diff，定位变更函数，结合机器人运动安全、D455 深度、手势 FSM、Action 契约等项目规则，生成风险报告、review 建议和测试补全清单。

## 为什么这个案例可信

- 不是泛泛的聊天机器人，而是嵌入研发流程的工具。
- 输入是真实工程代码和 PR diff。
- 输出不是“建议注意边界条件”，而是绑定到文件、函数、风险类型和测试场景。
- 对机器人项目很贴合：运动控制、目标锁定、深度异常、手势抖动、Action 兼容性都有明确工程风险。

## Demo 流程

```powershell
cd D:\person-follow-ai-reviewer-demo
python -m unittest discover -s tests
python -m ai_review_bot.cli --repo sample_repo --diff examples\risky_person_follow.patch --include-prompt --output reports\demo_review.md
```

讲解时打开 `reports\demo_review.md`，重点展示三件事：

1. 它识别出 `_handle_lost` 删除 0 速度发布属于高风险运动安全问题。
2. 它识别出 `start:<target>` 解析方式变化会破坏 `start:person`、`start:fridge` 这类外部契约。
3. 它发现 patch 没有测试变更，并给出了具体测试场景。

## STAR 话术

S：机器人跟随模块改动频繁，review 时很难靠人工稳定覆盖运动安全、深度异常、手势状态机和行为树 Action 兼容性。

T：我希望在 PR 阶段做一个 AI 助手，先自动发现高风险变更并给 reviewer 提供上下文和测试建议。

A：我做了 diff 解析、AST 函数定位、项目风险规则匹配，并把定位到的函数片段和规则说明整理成 LLM prompt。LLM 只基于这些上下文输出结论，避免泛泛而谈。

R：在 demo patch 中，它能发现“目标丢失后不再发布 0 速度”和“Action 参数解析破坏兼容性”两个典型问题，并自动补出离线测试建议。

## 面试官追问

Q：为什么叫 AI 助力，而不只是规则扫描？

A：规则扫描负责召回高风险上下文，LLM 负责把代码、项目知识和测试策略组织成 reviewer 能直接使用的建议。真实落地时还可以接 GitHub App，在 PR 里自动评论。

Q：怎么减少幻觉？

A：只把 diff 命中的函数、文件路径、规则依据和项目约束交给模型，并要求每条建议必须能指向文件或函数。没有依据的内容只能标成需要人工确认。

Q：怎么控制成本？

A：先用规则和 AST 做过滤，只对机器人运动、安全、接口契约这类高风险改动调用模型。普通文档或低风险配置改动不调用模型。
