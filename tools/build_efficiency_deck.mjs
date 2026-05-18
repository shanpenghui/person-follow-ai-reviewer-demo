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

const p = Presentation.create({ slideSize: { width: W, height: H } });

function slide(root) {
  const s = p.slides.add();
  s.compose(root, { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 });
}

function title(head, sub) {
  return column({ width: fill, height: hug, gap: 14 }, [
    text(head, { width: wrap(1540), height: hug, style: { fontSize: 54, bold: true, color: navy, fontFace: "Aptos Display" } }),
    text(sub, { width: wrap(1420), height: hug, style: { fontSize: 24, color: muted, fontFace: "Aptos" } }),
  ]);
}

function metric(label, value, note, color) {
  return column({ width: fill, height: hug, gap: 8 }, [
    text(value, { width: fill, height: hug, style: { fontSize: 54, bold: true, color, fontFace: "Aptos Display" } }),
    text(label, { width: fill, height: hug, style: { fontSize: 22, bold: true, color: ink } }),
    note ? text(note, { width: fill, height: hug, style: { fontSize: 17, color: muted } }) : null,
  ].filter(Boolean));
}

function card(label, value, note, color) {
  return panel({ width: fill, height: hug, fill: paint("#FFFFFF"), line: stroke(border), borderRadius: 12, padding: { x: 28, y: 24 } },
    metric(label, value, note, color));
}

function step(n, head, body, color) {
  return row({ width: fill, height: hug, gap: 18, align: "start" }, [
    panel({ width: fixed(56), height: fixed(56), fill: paint(color), borderRadius: "rounded-full", align: "center", justify: "center" },
      text(String(n), { width: hug, height: hug, style: { fontSize: 23, bold: true, color: "#FFFFFF" } })),
    column({ width: fill, height: hug, gap: 6 }, [
      text(head, { width: fill, height: hug, style: { fontSize: 25, bold: true, color: ink } }),
      text(body, { width: wrap(710), height: hug, style: { fontSize: 18, color: muted } }),
    ]),
  ]);
}

function bar(label, manual, ai, max, unit) {
  const scale = 520 / max;
  return column({ width: fill, height: hug, gap: 9 }, [
    text(label, { width: fill, height: hug, style: { fontSize: 22, bold: true, color: ink } }),
    row({ width: fill, height: fixed(36), gap: 12, align: "center" }, [
      text("人工", { width: fixed(72), height: hug, style: { fontSize: 17, color: muted } }),
      shape({ width: fixed(manual * scale), height: fixed(16), fill: paint("#94A3B8"), borderRadius: "rounded-full" }),
      text(`${manual}${unit}`, { width: fixed(92), height: hug, style: { fontSize: 17, color: muted } }),
    ]),
    row({ width: fill, height: fixed(36), gap: 12, align: "center" }, [
      text("AI", { width: fixed(72), height: hug, style: { fontSize: 17, color: muted } }),
      shape({ width: fixed(ai * scale), height: fixed(16), fill: paint(green), borderRadius: "rounded-full" }),
      text(`${ai}${unit}`, { width: fixed(92), height: hug, style: { fontSize: 17, bold: true, color: green } }),
    ]),
  ]);
}

function note(s) {
  return text(s, { width: wrap(1500), height: hug, style: { fontSize: 18, color: muted } });
}

// 1. Cover: department framing
slide(
  grid({ width: fill, height: fill, columns: [fr(1.05), fr(0.95)], columnGap: 64, padding: { x: 88, y: 74 } }, [
    column({ width: fill, height: fill, justify: "center", gap: 28 }, [
      text("面向研发部门的 AI PR 风险前置方案", {
        width: wrap(850), height: hug,
        style: { fontSize: 68, bold: true, color: navy, fontFace: "Aptos Display" },
      }),
      text("以 person_follow 机器人跟随模块为例，把评审准备、风险识别、测试建议和知识沉淀前置到 PR 阶段。", {
        width: wrap(840), height: hug, style: { fontSize: 27, color: "#40506A" },
      }),
      row({ width: fill, height: hug, gap: 22 }, [
        card("首轮评审准备", "25 -> 3 min", "单 PR demo 口径", green),
        card("每周节省", "7.3 h", "按 20 个 PR 估算", blue),
      ]),
    ]),
    panel({ width: fill, height: fill, fill: paint("#FFFFFF"), line: stroke(border), borderRadius: 16, padding: 42 },
      column({ width: fill, height: fill, gap: 26, justify: "center" }, [
        text("研发部门关心的不是“会聊天”", { width: fill, height: hug, style: { fontSize: 31, bold: true, color: navy } }),
        rule({ width: fill, stroke: border, weight: 2 }),
        step(1, "PR 吞吐", "缩短 reviewer 找上下文和判断影响范围的准备时间。", blue),
        step(2, "质量前置", "运动安全、接口契约、测试缺失等高风险点更早暴露。", red),
        step(3, "团队复用", "把老员工经验沉淀成规则、prompt 和可复用测试清单。", teal),
      ])),
  ])
);

// 2. How the R&D department uses it
slide(
  column({ width: fill, height: fill, padding: { x: 86, y: 62 }, gap: 34 }, [
    title("研发部门怎么落地使用", "建议先以离线 demo 展示，再接入 GitHub/GitLab Webhook 做 2 周试点。"),
    grid({ width: fill, height: fill, columns: [fr(0.95), fr(1.05)], columnGap: 54 }, [
      column({ width: fill, height: fill, gap: 28 }, [
        step(1, "开发者提交 PR", "工具读取 diff，解析变更文件和新增行号。", blue),
        step(2, "自动定位上下文", "AST 定位函数级上下文，检索项目风险规则。", teal),
        step(3, "AI 生成初筛报告", "输出风险等级、命中函数、测试建议和需要人工确认的问题。", amber),
        step(4, "Reviewer 关闭闭环", "人工确认、采纳或忽略建议，结果进入团队度量。", red),
      ]),
      panel({ width: fill, height: fill, fill: paint("#111827"), borderRadius: 14, padding: 36 },
        column({ width: fill, height: fill, gap: 18 }, [
          text("本地演示命令", { width: fill, height: hug, style: { fontSize: 24, bold: true, color: "#C7D2FE" } }),
          text("cd D:\\person-follow-ai-reviewer-demo\npython -m unittest discover -s tests\npython -m ai_review_bot.cli --repo sample_repo --diff examples\\risky_person_follow.patch --include-prompt --output reports\\demo_review.md", {
            width: wrap(790), height: hug, style: { fontSize: 23, color: "#F8FAFC", fontFace: "Consolas" },
          }),
          rule({ width: fill, stroke: "#334155", weight: 2 }),
          text("部门试点接入", { width: fill, height: hug, style: { fontSize: 24, bold: true, color: "#C7D2FE" } }),
          text("PR opened / updated -> 触发分析 -> 评论到 PR -> 采纳率与缺陷前置率进入看板", {
            width: wrap(760), height: hug, style: { fontSize: 23, color: "#F8FAFC" },
          }),
        ])),
    ]),
  ])
);

// 3. Pain points to capabilities
slide(
  column({ width: fill, height: fill, padding: { x: 86, y: 62 }, gap: 38 }, [
    title("从研发痛点出发", "这个工具解决的是 PR 阶段的重复性认知负担和质量前置问题，而不是为了用 AI 而用 AI。"),
    grid({ width: fill, height: fixed(500), columns: [fr(1), fr(1), fr(1), fr(1)], columnGap: 24 }, [
      panel({ width: fill, height: fill, fill: paint("#FFFFFF"), line: stroke(border), borderRadius: 12, padding: 28 },
        step("1", "Reviewer 时间碎片化", "人工先找上下文，真正看设计和风险的时间被压缩。", blue)),
      panel({ width: fill, height: fill, fill: paint("#FFFFFF"), line: stroke(border), borderRadius: 12, padding: 28 },
        step("2", "老模块隐性知识多", "运动控制、Action 契约、手势 FSM 规则散落在代码和文档里。", teal)),
      panel({ width: fill, height: fill, fill: paint("#FFFFFF"), line: stroke(border), borderRadius: 12, padding: 28 },
        step("3", "测试补全不稳定", "经常知道要补测试，但缺少具体场景清单。", amber)),
      panel({ width: fill, height: fill, fill: paint("#FFFFFF"), line: stroke(border), borderRadius: 12, padding: 28 },
        step("A", "规则 + 上下文 + LLM", "先召回高风险上下文，再生成 reviewer 可执行建议。", red)),
    ]),
    note("部门价值：把个人经验变成可复用的工程规则，让每个 PR 都经过一致的质量门槛。"),
  ])
);

// 4. Throughput metrics
slide(
  column({ width: fill, height: fill, padding: { x: 86, y: 62 }, gap: 34 }, [
    title("提效指标 1：提升 PR 吞吐，减少评审准备成本", "演示口径：单个中等规模机器人跟随 PR；部门估算按每周 20 个类似 PR 计算。"),
    grid({ width: fill, height: fill, columns: [fr(0.94), fr(1.06)], columnGap: 52 }, [
      column({ width: fill, height: fill, gap: 24, justify: "center" }, [
        bar("理解 diff + 找相关函数", 12, 1, 14, "min"),
        bar("影响范围判断", 8, 1, 14, "min"),
        bar("测试点整理", 5, 1, 14, "min"),
        row({ width: fill, height: hug, gap: 22 }, [
          card("单 PR 节省", "22 min", "25 -> 3 min", green),
          card("每周节省", "7.3 h", "20 个 PR 估算", blue),
        ]),
      ]),
      chart({ name: "review-time", chartType: "bar", width: fill, height: fill, config: {
        title: "首轮评审准备时间（分钟）",
        categories: ["理解 diff", "影响判断", "测试整理", "总计"],
        series: [{ name: "人工", values: [12, 8, 5, 25] }, { name: "AI 辅助", values: [1, 1, 1, 3] }],
      }}),
    ]),
  ])
);

// 5. Quality metrics
slide(
  column({ width: fill, height: fill, padding: { x: 86, y: 62 }, gap: 32 }, [
    title("提效指标 2：质量前置，减少后置返工", "AI 的收益不只是省时间，更是让高风险改动在 PR 阶段被稳定检查。"),
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
        note("说明：这里是 demo 的可复现实验口径。生产环境应持续统计 AI 评论采纳率、缺陷前置发现率、回归缺陷逃逸率和测试补充率。"),
      ]),
    ]),
  ])
);

// 6. Department KPI dashboard
slide(
  column({ width: fill, height: fill, padding: { x: 86, y: 62 }, gap: 34 }, [
    title("给研发部门看的度量闭环", "先做小范围试点，用数据判断是否扩大到更多仓库和团队。"),
    grid({ width: fill, height: fill, columns: [fr(1.08), fr(0.92)], columnGap: 52 }, [
      column({ width: fill, height: fill, gap: 20, justify: "center" }, [
        row({ width: fill, height: hug, gap: 18 }, [
          card("PR 首轮准备", "25 -> 3 min", "效率指标", green),
          card("高风险前置", "1 -> 2 条", "质量指标", blue),
        ]),
        row({ width: fill, height: hug, gap: 18 }, [
          card("测试建议", "2 -> 6 条", "测试指标", teal),
          card("定位粒度", "文件 -> 函数", "协作指标", amber),
        ]),
        text("建议试点节奏", { width: fill, height: hug, style: { fontSize: 28, bold: true, color: navy } }),
        text("选择 1 个机器人/感知控制仓库，连续 2 周对所有 PR 生成 AI 初筛报告；统计采纳率、误报率、节省时间和缺陷前置率。", {
          width: wrap(820), height: hug, style: { fontSize: 24, color: ink },
        }),
      ]),
      chart({ name: "dept-kpi", chartType: "bar", width: fill, height: fill, config: {
        title: "部门收益指数（人工=100）",
        categories: ["人工流程", "AI 辅助流程"],
        series: [{ name: "综合指数", values: [100, 260] }],
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
