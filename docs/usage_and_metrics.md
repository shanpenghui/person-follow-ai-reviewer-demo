# 使用方式与提效指标

## 怎么用

面试现场建议按这个顺序演示：

```powershell
cd D:\person-follow-ai-reviewer-demo
python -m unittest discover -s tests
python -m ai_review_bot.cli --repo sample_repo --diff examples\risky_person_follow.patch --include-prompt --output reports\demo_review.md
```

打开 `reports\demo_review.md`，重点讲三点：

1. 工具把 PR diff 定位到函数级上下文。
2. 工具命中两个风险：运动安全、Action 契约兼容。
3. 工具不是只说“补测试”，而是给出具体测试点。

PPT 生成命令：

```powershell
cd D:\person-follow-ai-reviewer-demo
node tools\build_efficiency_deck.mjs
```

输出：

- `slides_output\person_follow_ai_review_efficiency.pptx`
- `slides_output\slide_01.png` 到 `slide_06.png`

## 人工 vs AI 的提效指标

这些数据是面试 demo 的可复现实验口径，不宣称生产收益。

| 指标 | 人工首轮 | AI 辅助 | 说明 |
|---|---:|---:|---|
| 评审准备时间 | 25 min | 3 min | 人工通读 diff/上下文；AI 先给风险锚点 |
| 高风险前置发现 | 1 条 | 2 条 | demo patch 中运动安全、Action 契约都被命中 |
| 测试建议数量 | 2 条 | 6 条 | 从“请补测试”变成具体测试场景 |
| 定位粒度 | 文件级 | 函数级 | 直接定位 `_handle_lost`、`_parse_action_name` |

## 面试讲法

> AI 的提效不只是少花几分钟，而是把 reviewer 的工作从“自己找问题”变成“验证 AI 给出的高风险结论”。人工 review 更适合判断代码语义、架构取舍和实机验证；AI 更适合稳定执行项目规则、召回上下文和生成测试清单。

上线后可以持续统计：

- PR 首轮 review 时长
- AI 评论采纳率
- 缺陷前置发现率
- 测试补充率
- 回归缺陷逃逸率

