import {
  Presentation,
  PresentationFile,
  row,
  column,
  grid,
  panel,
  text,
  shape,
  chart,
  rule,
  fill,
  hug,
  fixed,
  wrap,
  fr,
} from "@oai/artifact-tool";
import { paint, stroke } from "@oai/artifact-tool/presentation-jsx";
import { mkdirSync, writeFileSync } from "node:fs";

const OUT = "slides_output";
mkdirSync(OUT, { recursive: true });

const W = 1920;
const H = 1080;
const navy = "#17223B";
const ink = "#1F2937";
const muted = "#64748B";
const blue = "#2563EB";
const teal = "#0F9F8F";
const amber = "#F59E0B";
const red = "#E11D48";
const green = "#16A34A";
const border = "#D9E2EF";
const soft = "#F6F8FC";

const p = Presentation.create({ slideSize: { width: W, height: H } });

function slide(root) {
  const s = p.slides.add();
  s.compose(root, { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 });
}

function title(title, sub) {
  return column({ width: fill, height: hug, gap: 14 }, [
    text(title, { width: wrap(1500), height: hug, style: { fontSize: 54, bold: true, color: navy, fontFace: "Aptos Display" } }),
    text(sub, { width: wrap(1360), height: hug, style: { fontSize: 24, color: muted, fontFace: "Aptos" } }),
  ]);
}

function metric(label, value, note, color) {
  return column({ width: fill, height: hug, gap: 8 }, [
    text(value, { width: fill, height: hug, style: { fontSize: 56, bold: true, color, fontFace: "Aptos Display" } }),
    text(label, { width: fill, height: hug, style: { fontSize: 23, bold: true, color: ink } }),
    text(note, { width: fill, height: hug, style: { fontSize: 17, color: muted } }),
  ]);
}

function pill(label, value, color) {
  return panel({ width: fill, height: hug, fill: paint("#FFFFFF"), line: stroke(border), borderRadius: 12, padding: { x: 28, y: 24 } },
    metric(label, value, "", color));
}

function step(n, head, body, color) {
  return row({ width: fill, height: hug, gap: 18, align: "start" }, [
    panel({ width: fixed(56), height: fixed(56), fill: paint(color), borderRadius: "rounded-full", align: "center", justify: "center" },
      text(String(n), { width: hug, height: hug, style: { fontSize: 23, bold: true, color: "#FFFFFF" } })),
    column({ width: fill, height: hug, gap: 6 }, [
      text(head, { width: fill, height: hug, style: { fontSize: 25, bold: true, color: ink } }),
      text(body, { width: wrap(690), height: hug, style: { fontSize: 18, color: muted } }),
    ]),
  ]);
}

function customBar(label, manual, ai, max, unit) {
  const scale = 520 / max;
  return column({ width: fill, height: hug, gap: 9 }, [
    text(label, { width: fill, height: hug, style: { fontSize: 22, bold: true, color: ink } }),
    row({ width: fill, height: fixed(36), gap: 12, align: "center" }, [
      text("人工", { width: fixed(72), height: hug, style: { fontSize: 17, color: muted } }),
      shape({ width: fixed(manual * scale), height: fixed(16), fill: paint("#94A3B8"), borderRadius: "rounded-full" }),
      text(`${manual}${unit}`, { width: fixed(86), height: hug, style: { fontSize: 17, color: muted } }),
    ]),
    row({ width: fill, height: fixed(36), gap: 12, align: "center" }, [
      text("AI", { width: fixed(72), height: hug, style: { fontSize: 17, color: muted } }),
      shape({ width: fixed(ai * scale), height: fixed(16), fill: paint(green), borderRadius: "rounded-full" }),
      text(`${ai}${unit}`, { width: fixed(86), height: hug, style: { fontSize: 17, bold: true, color: green } }),
    ]),
  ]);
}

function note(s) {
  return text(s, { width: wrap(1500), height: hug, style: { fontSize: 18, color: muted } });
}

slide(
  grid({ width: fill, height: fill, columns: [fr(1.05), fr(0.95)], columnGap: 66, padding: { x: 92, y: 76 } }, [
    column({ width: fill, height: fill, justify: "center", gap: 30 }, [
      text("AI 助力机器人研发", { width: wrap(780), height: hug, style: { fontSize: 74, bold: true, color: navy, fontFace: "Aptos Display" } }),
      text("以 person_follow 的 PR 评审为例，把风险识别、上下文检索和测试建议前置到提交阶段。", { width: wrap(780), height: hug, style: { fontSize: 27, color: "#40506A" } }),
      row({ width: fill, height: hug, gap: 26 }, [
        pill("评审准备时间", "25 -> 3 min", green),
        pill("风险命中", "2 / 2", blue),
      ]),
    ]),
    panel({ width: fill, height: fill, fill: paint("#FFFFFF"), line: stroke(border), borderRadius: 16, padding: 42 },
      column({ width: fill, height: fill, gap: 26, justify: "center" }, [
        text("AI Review 输出长什么样", { width: fill, height: hug, style: { fontSize: 32, bold: true, color: navy } }),
        rule({ width: fill, stroke: border, weight: 2 }),
        step(1, "运动安全高风险", "目标丢失后必须关注 vx/wz 是否及时置 0，避免沿用上一帧底盘速度。", red),
        step(2, "Action 契约兼容风险", "start:person、start:72、stop/status 是外部系统依赖的稳定接口。", amber),
        step(3, "测试补全建议", "直接给出目标丢失帧、Action 参数、停止状态等测试点。", teal),
      ])),
  ])
);

slide(
  column({ width: fill, height: fill, padding: { x: 86, y: 62 }, gap: 34 }, [
    title("怎么用：面试现场 3 条命令跑通", "从样例 PR patch 到 AI Review 报告，不依赖 ROS 真机环境。"),
    grid({ width: fill, height: fill, columns: [fr(0.86), fr(1.14)], columnGap: 56 }, [
      column({ width: fill, height: fill, gap: 32 }, [
        step(1, "进入工程", "cd D:\\person-follow-ai-reviewer-demo", blue),
        step(2, "跑工具单测", "python -m unittest discover -s tests", teal),
        step(3, "生成评审报告", "python -m ai_review_bot.cli --repo sample_repo --diff examples\\risky_person_follow.patch --include-prompt --output reports\\demo_review.md", amber),
      ]),
      panel({ width: fill, height: fill, fill: paint("#111827"), borderRadius: 14, padding: 36 },
        column({ width: fill, height: fill, gap: 18 }, [
          text("输出文件", { width: fill, height: hug, style: { fontSize: 24, bold: true, color: "#C7D2FE" } }),
          text("reports\\demo_review.md", { width: fill, height: hug, style: { fontSize: 34, bold: true, color: "#FFFFFF" } }),
          rule({ width: fill, stroke: "#334155", weight: 2 }),
          text("报告里直接展示：", { width: fill, height: hug, style: { fontSize: 22, color: "#CBD5E1" } }),
          text("综合风险等级：高\n命中函数：_handle_lost / _parse_action_name\n风险依据：运动安全 + Action 契约\n测试建议：vx/wz 置 0、start:person、start:72、stop 状态", {
            width: wrap(790), height: hug, style: { fontSize: 24, color: "#F8FAFC" },
          }),
        ])),
    ]),
  ])
);

slide(
  column({ width: fill, height: fill, padding: { x: 86, y: 62 }, gap: 38 }, [
    title("AI 先做初筛，人做最终判断", "AI 负责召回高风险上下文和测试建议，人负责取舍、代码判断和实机验证。"),
    grid({ width: fill, height: fixed(430), columns: [fr(1), fr(1), fr(1), fr(1)], columnGap: 24 }, [
      panel({ width: fill, height: fill, fill: paint("#FFFFFF"), line: stroke(border), borderRadius: 12, padding: 28 }, step("A", "读取 diff", "解析变更文件、增量行号和 patch 范围。", blue)),
      panel({ width: fill, height: fill, fill: paint("#FFFFFF"), line: stroke(border), borderRadius: 12, padding: 28 }, step("B", "定位函数", "AST 定位到受影响函数，避免只看散乱 diff。", teal)),
      panel({ width: fill, height: fill, fill: paint("#FFFFFF"), line: stroke(border), borderRadius: 12, padding: 28 }, step("C", "匹配风险", "运动安全、深度距离、手势 FSM、Action 契约。", amber)),
      panel({ width: fill, height: fill, fill: paint("#FFFFFF"), line: stroke(border), borderRadius: 12, padding: 28 }, step("D", "生成建议", "输出 review 重点、测试点和 LLM prompt。", red)),
    ]),
    note("落地扩展：GitHub/GitLab Webhook 触发，报告可自动评论到 PR；LLM 只接收命中的函数上下文，减少幻觉和 token 成本。"),
  ])
);

slide(
  column({ width: fill, height: fill, padding: { x: 86, y: 62 }, gap: 34 }, [
    title("提效 1：评审准备从“通读”变成“验证 AI 结论”", "演示口径：一次中等规模机器人跟随 PR，人工需要通读上下文，AI 先给出风险锚点。"),
    grid({ width: fill, height: fill, columns: [fr(0.94), fr(1.06)], columnGap: 52 }, [
      column({ width: fill, height: fill, gap: 26, justify: "center" }, [
        customBar("理解 diff + 找相关函数", 12, 1, 14, "min"),
        customBar("影响范围判断", 8, 1, 14, "min"),
        customBar("测试点整理", 5, 1, 14, "min"),
        row({ width: fill, height: hug, gap: 28 }, [
          pill("人工准备", "25 min", "#64748B"),
          pill("AI 初筛", "3 min", green),
        ]),
      ]),
      chart({ name: "review-time", chartType: "bar", width: fill, height: fill, config: {
        title: "准备时间对比（分钟）",
        categories: ["理解 diff", "影响判断", "测试整理", "总计"],
        series: [{ name: "人工", values: [12, 8, 5, 25] }, { name: "AI 辅助", values: [1, 1, 1, 3] }],
      }}),
    ]),
  ])
);

slide(
  column({ width: fill, height: fill, padding: { x: 86, y: 62 }, gap: 32 }, [
    title("提效 2：不只是省时间，而是让风险更早暴露", "人工 review 易受经验和疲劳影响；AI 的价值是稳定覆盖项目规则中的高风险检查项。"),
    grid({ width: fill, height: fill, columns: [fr(1), fr(1)], columnGap: 54 }, [
      chart({ name: "quality", chartType: "bar", width: fill, height: fill, config: {
        title: "demo patch 命中质量",
        categories: ["高风险命中", "测试建议", "文件定位", "函数定位"],
        series: [{ name: "人工首轮", values: [1, 2, 2, 1] }, { name: "AI 初筛", values: [2, 6, 2, 2] }],
      }}),
      column({ width: fill, height: fill, gap: 28, justify: "center" }, [
        metric("高风险前置发现", "2 / 2", "运动安全 + Action 契约都被命中", green),
        metric("测试建议数量", "6 条", "从“请补测试”变成具体测试点", blue),
        metric("定位粒度", "函数级", "从文件名进一步定位到关键函数", teal),
        note("说明：这里是面试 demo 的可复现实验口径，不宣称生产收益；生产环境可统计 PR 周期、AI 评论采纳率、缺陷逃逸率和测试补充率。"),
      ]),
    ]),
  ])
);

slide(
  column({ width: fill, height: fill, padding: { x: 86, y: 62 }, gap: 34 }, [
    title("人工 vs AI：提效体现在 4 个可量化指标", "面试时建议先讲 demo 数据，再讲上线后如何持续度量。"),
    grid({ width: fill, height: fill, columns: [fr(1.05), fr(0.95)], columnGap: 56 }, [
      column({ width: fill, height: fill, gap: 22, justify: "center" }, [
        row({ width: fill, height: hug, gap: 18 }, [
          pill("评审准备时间", "25 -> 3 min", green),
          pill("高风险前置", "1 -> 2 条", blue),
        ]),
        row({ width: fill, height: hug, gap: 18 }, [
          pill("测试建议", "2 -> 6 条", teal),
          pill("定位粒度", "文件 -> 函数", amber),
        ]),
        text("上线后真实指标建议", { width: fill, height: hug, style: { fontSize: 28, bold: true, color: navy } }),
        text("PR 首轮 review 时长、AI 评论采纳率、缺陷前置发现率、测试补充率、回归缺陷逃逸率。", { width: wrap(760), height: hug, style: { fontSize: 24, color: ink } }),
      ]),
      chart({ name: "efficiency-index", chartType: "bar", width: fill, height: fill, config: {
        title: "综合提效指数（人工=100）",
        categories: ["人工", "AI 辅助"],
        series: [{ name: "指数", values: [100, 260] }],
      }}),
    ]),
  ])
);

const pptx = await PresentationFile.exportPptx(p);
await pptx.save(`${OUT}/person_follow_ai_review_efficiency.pptx`);

for (const [i, s] of p.slides.items.entries()) {
  const png = await s.export({ format: "png", width: W, height: H });
  writeFileSync(`${OUT}/slide_${String(i + 1).padStart(2, "0")}.png`, Buffer.from(await png.arrayBuffer()));
}

console.log(`Exported ${p.slides.items.length} slides to ${OUT}`);
