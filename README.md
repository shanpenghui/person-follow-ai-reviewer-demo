# Person Follow AI Reviewer Demo

这是一个面向研发部门的 AI 研发效能展示项目：以 `D:\midea\person_follow` 的机器人跟随模块为样例，演示“AI 助力研发”如何落到 PR 评审、风险识别、测试补全和研发效能度量。

## 项目结构

```text
ai_review_bot/     # PR 风险分析 CLI
sample_repo/       # 从 person_follow 复制出来的演示样例代码
examples/          # 面试演示用 patch
tests/             # 工具自身单测
docs/              # 面试讲解手卡
```

## 这个案例讲什么

面试/研发部门汇报主线：

> 我做了一个 AI PR Review 助手，接入机器人跟随模块的研发流程。它读取 PR diff，定位变更函数，结合项目风险规则检索上下文，再让 LLM 生成 review 建议和测试清单，帮助研发部门把运动安全、深度异常、手势 FSM 和 Action 契约等风险前置到 PR 阶段。

这个 demo 当前用离线 patch 演示，不依赖真实 ROS 环境，也不需要 OpenAI API Key。真实落地时可以接 GitHub App / GitLab Webhook，把报告自动评论到 PR。

## 快速运行

```powershell
cd D:\person-follow-ai-reviewer-demo
python -m unittest discover -s tests
python -m ai_review_bot.cli --repo sample_repo --diff examples\risky_person_follow.patch --include-prompt --output reports\demo_review.md
```

报告会生成到：

```text
D:\person-follow-ai-reviewer-demo\reports\demo_review.md
```

## Demo 里故意放的两个风险

1. `scene_servo/person_yolo_servo_node.py` 的目标丢失逻辑删除了 `_publish_cmd(0.0, 0.0)`，机器人可能在目标丢失后继续沿用上一帧底盘速度。
2. `person_follow/follow_action_server_node.py` 把 `start:<target>` 全部改成 `int(target_name)`，会破坏 `start:person`、`start:fridge` 等兼容入口。

工具会输出：

- 综合风险等级。
- 命中的风险类型。
- 影响文件和函数。
- Review 重点。
- 建议补充的测试用例。
- 可选 LLM Prompt。

## GitHub 仓库

本地 remote 按你的账号规划为：

```text
https://github.com/shanpenghui/person-follow-ai-reviewer-demo.git
```

如果远端仓库还不存在，可以在 GitHub 新建同名仓库后执行：

```powershell
cd D:\person-follow-ai-reviewer-demo
git push -u origin main
```
