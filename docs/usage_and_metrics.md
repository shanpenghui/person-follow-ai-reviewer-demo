# 面向研发部门的使用方式与提效指标

## 怎么用

建议把它讲成研发部门的 PR 质量前置工具，而不是个人效率工具。面试或部门汇报时按这个顺序演示：

```powershell
cd D:\person-follow-ai-reviewer-demo
python -m unittest discover -s tests
python -m ai_review_bot.cli --repo sample_repo --diff examples\risky_person_follow.patch --include-prompt --output reports\demo_review.md
```

打开 `reports\demo_review.md`，重点讲三点：

1. 工具把 PR diff 定位到函数级上下文。
2. 工具命中两个风险：运动安全、Action 契约兼容。
3. 工具不是只说“补测试”，而是给出具体测试点。

部门试点接入方式：

```text
PR opened / updated
  -> Webhook 触发 AI Review
  -> 自动评论到 PR
  -> Reviewer 采纳/忽略/补充
  -> 采纳率、误报率、节省时间进入研发效能看板
```

PPT 生成命令：

```powershell
cd D:\person-follow-ai-reviewer-demo
node tools\build_efficiency_deck.mjs
```

输出：

- `slides_output\person_follow_ai_review_efficiency.pptx`
- `slides_output\slide_01.png` 到 `slide_06.png`

## 研发部门视角的提效指标

这些数据是 demo 的可复现实验口径，不宣称生产收益。部门落地时建议用 2 周试点数据替换。

| 指标 | 人工首轮 | AI 辅助 | 说明 |
|---|---:|---:|---|
| 单 PR 首轮准备时间 | 25 min | 3 min | 人工通读 diff/上下文；AI 先给风险锚点 |
| 每周节省 reviewer 时间 | 0 h | 7.3 h | 按每周 20 个类似 PR 估算：22 min × 20 |
| 高风险前置发现 | 1 条 | 2 条 | demo patch 中运动安全、Action 契约都被命中 |
| 测试建议数量 | 2 条 | 6 条 | 从“请补测试”变成具体测试场景 |
| 定位粒度 | 文件级 | 函数级 | 直接定位 `_handle_lost`、`_parse_action_name` |

## 针对研发部门的讲法

> 对研发部门来说，AI 的价值不是替代 reviewer，而是把 PR 阶段的重复性认知负担自动化。人工 review 更适合判断代码语义、架构取舍和实机验证；AI 更适合稳定执行项目规则、召回上下文、生成测试清单，并把团队经验沉淀成可复用的质量门槛。

上线后建议持续统计：

- PR 首轮 review 时长
- AI 评论采纳率
- AI 评论误报率
- 缺陷前置发现率
- 测试补充率
- 回归缺陷逃逸率

## 试点建议

选择 1 个机器人/感知控制仓库，连续 2 周对所有 PR 生成 AI 初筛报告。每周汇总：

- 生成了多少条 AI review 建议
- 被 reviewer 采纳多少条
- 哪些规则误报较多，需要调优
- 哪些问题被前置发现，避免进入联调或实机阶段
