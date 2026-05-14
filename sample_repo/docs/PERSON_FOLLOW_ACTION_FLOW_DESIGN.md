# person_follow 行为协议与流程

> **版本**：v0.6 — 2026-04-15  
> **状态**：设计草案，待实现  
> **范围**：第一版仅包含 YOLO 跟随、已有模板场景跟随、拍照  
> **说明**：当前线上行为仍以现网代码为准，本文档仅定义第一版最小闭环协议。

---

## 1. 第一版目标

1. 用同一个 Action Server（`/person_follow/skill_behavior_tree`）承载 3 类能力：YOLO 跟随、场景跟随（已有模板）、场景拍照。  
2. `action_name` 驱动模式，避免通过改默认配置切模式。  
3. 保持向后兼容：旧流程 `action_name=start` 继续可用。  
4. 先保证最小可用闭环，不引入复杂抢占/恢复策略。

---

## 2. Action 接口定义（兼容扩展）

基于现有 `PersonFollow.action` 扩展，不删除旧字段：

```text
# Goal
string action_name
string json_params          # 新增，默认 ""
---
# Result
bool success
uint16 status
string result_msg           # 新增
string error_code           # 新增，成功时建议为空
---
# Feedback
float32 action_percentage
uint16 status
string status_msg           # 新增
```

### 2.1 `json_params` 最小约定（第一版）

建议固定最小 schema（其余字段可扩展）：

```json
{
  "request_id": "uuid-or-bt-node-id",
  "timeout_sec": 30,
  "template_policy": "overwrite",
  "template_path": "/abs/path/reference_frame.json"
}
```

目的：减少 BT 与 skill_lib 两侧自由拼字符串带来的不一致。

**关于 `follow_mode`**：第一版模式由 `action_name` 决定（`start`/`start_yolo` → YOLO，`start_scene` → 场景），`json_params` 不含 `follow_mode` 字段。后续若需要更灵活的模式切换再扩展。

---

## 3. 状态机与语义（第一版）

### 3.1 状态码

| 状态 | 值 | 说明 |
|------|----|------|
| `IDLE` | 0 | 空闲 |
| `FOLLOWING_YOLO` | 1 | YOLO 跟随中 |
| `FOLLOWING_SCENE` | 2 | 场景跟随中 |
| `CAPTURING` | 3 | 场景采集中 |
| `ERROR` | 4 | 异常态 |

### 3.2 转换规则

| 当前状态 | action_name | 结果 |
|----------|-------------|------|
| `IDLE` | `start` / `start_yolo` | 前置检查通过 → `FOLLOWING_YOLO` |
| `IDLE` | `start_scene` | 有模板 → `FOLLOWING_SCENE`；无模板 → 失败 `TEMPLATE_NOT_FOUND` |
| `IDLE` | `capture_scene` | → `CAPTURING` → 成功后回 `IDLE` |
| `IDLE` | `stop` | 直接成功返回（幂等） |
| `FOLLOWING_YOLO` | `stop` | 停车归位 → `IDLE` |
| `FOLLOWING_SCENE` | `stop` | 停车归位 → `IDLE` |
| `FOLLOWING_SCENE` | `capture_scene` | 失败 `BUSY_STATE`（第一版不支持跟随中更新） |
| `FOLLOWING_YOLO` | `capture_scene` | 失败 `BUSY_STATE` |
| `FOLLOWING_*` | `start*`（同 action_name） | 幂等：返回当前 feedback，不重置状态 |
| `FOLLOWING_*` | `start*`（不同 action_name） | 失败 `BUSY_STATE`（第一版不做抢占重启） |
| 任意 | `status` | 立即返回当前状态（不受 BUSY 限制） |
| `CAPTURING` | `stop` | 中止并回 `IDLE` |
| `CAPTURING` | 其他 | 失败 `BUSY_STATE` |
| `ERROR` | `stop` | 回 `IDLE` |
| `ERROR` | `start*` | 失败 `BUSY_STATE`（先显式 stop） |

### 3.3 stop 与 cancel 语义

- **`stop`**：停车归位并回 `IDLE`，立即 succeed 返回。
- **`cancel`**（ROS2 action cancel）：第一版等价于 `stop`——客户端发 cancel 后，服务端停车归位，goal 以 `CANCELED` 终态返回。当前实现（`follow_action_server_node.py`）已处理 `is_cancel_requested`，行为与 stop 一致。  
- 后续版本可能区分 cancel（暂停/可恢复）和 stop（终止/不可恢复），但第一版不区分。

---

## 4. action_name 定义

| action_name | 行为 | 长运行 | 说明 |
|-------------|------|--------|------|
| `start` | 启动 YOLO 跟随 | 是 | 兼容旧 XML，等价 `start_yolo` |
| `start_yolo` | 启动 YOLO 跟随 | 是 | 显式模式 |
| `start_scene` | 启动场景跟随 | 是 | 无模板直接失败 |
| `capture_scene` | 采集场景模板 | 否 | 仅 `IDLE` 可执行 |
| `stop` | 停止跟随 | 否 | 任意状态幂等 |
| `status` | 查询状态 | 否 | 任意状态可用，立即返回当前状态、模式、模板可用性。不受 BUSY_STATE 限制 |

---

## 5. 错误码（第一版最小集）

| error_code | 触发场景 | 建议处理 |
|------------|----------|----------|
| `BUSY_STATE` | 当前状态不允许该动作 | 先 stop 或等待 |
| `TEMPLATE_NOT_FOUND` | `start_scene` 无模板 | 先 `capture_scene` |
| `CAMERA_NOT_READY` | 相机无可用帧 | 等待重试 |
| `TF_NOT_READY` | TF 不可用 | 等待重试 |
| `INTERNAL_ERROR` | 未预期异常 | 报警 |

---

## 6. 模板采集与覆盖策略

### 6.1 保存策略（避免先删后拍导致模板丢失）

1. 采集并生成临时文件：`reference_frame.tmp.json`  
2. 做质量校验（特征点、深度有效率、关键字段完整）  
3. 校验通过后原子替换为 `reference_frame.json`  
4. 失败时保留旧模板，不做覆盖

### 6.2 第一版约束

第一版不支持跟随中更新模板。  
`capture_scene` 仅允许在 `IDLE` 执行。

---

## 7. 标准调用流程

### 7.1 YOLO 跟随

```text
BT -> PersonFollow(action_name=start|start_yolo)
   -> FOLLOWING_YOLO
   -> feedback
   -> stop/结束条件
BT <- result(success/status/error_code)
```

### 7.2 场景拍照 + 场景跟随

```text
BT -> PersonFollow(action_name=capture_scene)
BT <- result(success=true)
BT -> PersonFollow(action_name=start_scene)
BT <- result(...)
```

### 7.3 直接场景跟随（已有模板）

```text
BT -> PersonFollow(action_name=start_scene)
  ├─ 有模板 → FOLLOWING_SCENE → 到位/stop → IDLE
  └─ 无模板 → result(success=false, error_code=TEMPLATE_NOT_FOUND)
```

> **BT 处理建议**：收到 `TEMPLATE_NOT_FOUND` 时，应先走 §7.2 的 `capture_scene` 流程采集模板，再重试 `start_scene`。

## 8. 与 m_behavior_tree 对接说明（同事配合）

### 8.1 现状差异（基于 `feature-adapt_to_person_follow`）

当前 `PersonFollowAction` 插件实际仅稳定使用：

- 输入：`action_name`
- 输出：`success`, `status`

而 `m_tree_nodes.xml` 中 `PersonFollowAction` 描述仍含 `type/task_index/object/value/language` 等历史端口，和当前插件不一致。

### 8.2 行为树侧建议修改（仅文档定义）

1. 更新 `interfaces/action/PersonFollow.action` 依赖并重新编译。  
2. 在 `person_follow_action.hpp/.cpp` 对齐新端口：  
   - 输入：`action_name`, `json_params`  
   - 输出：`success`, `status`, `result_msg`, `error_code`  
3. 清理 `m_tree_nodes.xml` 中 `PersonFollowAction` 的历史无效端口，改为与插件一致。  
4. 新增/调整 BT XML：  
   - YOLO：`start` 或 `start_yolo`  
   - 场景：`capture_scene -> start_scene`

### 8.3 person_follow 侧建议修改（仅文档定义）

1. 支持新 action_name 与状态机。  
2. 支持 `json_params` 解析与默认值。  
3. 对齐 result/feedback 新字段。  
4. 按 §6 实现模板安全覆盖策略。

---

## 9. 协同交付顺序（建议）

1. **先接口**：冻结 `PersonFollow.action`（字段名、含义、错误码）。  
2. **再两边并行**：person_follow 与 BT 分头改。  
3. **最后联调**：按 §7 三条流程逐条验收。

最小兼容路径：若 BT 暂不改 C++，`action_name="start"` 仍可运行旧流程，不阻塞线上。

---

## 10. 验收清单（文档级）

1. `start_yolo` 可启动并可 `stop` 结束。  
2. `start_scene` 在无模板时返回 `TEMPLATE_NOT_FOUND`。  
3. `capture_scene` 失败时不覆盖旧模板。  
4. `FOLLOWING_*` 状态下调用 `capture_scene` 返回 `BUSY_STATE`。  
5. BT 能拿到 `error_code/result_msg` 并做分支处理。
6. `status` 在 `FOLLOWING_*` 状态下可正常查询。
7. 重复发送相同 `action_name` 时幂等（不重置状态）。
8. `cancel` 等价于 `stop`，goal 以 `CANCELED` 终态返回。

---

## 11. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2026-04-13 | 初版草案（动作与状态机雏形） |
| v0.2 | 2026-04-14 | 结合行为树调用方式重写流程 |
| v0.3 | 2026-04-14 | 聚焦接口定义、错误码、双方改动清单 |
| v0.4 | 2026-04-14 | 对齐 `m_behavior_tree/feature-adapt_to_person_follow` 现状，补齐协同交付与验收规范 |
| v0.5 | 2026-04-14 | 收敛第一版范围：仅 YOLO 跟随、已有模板场景跟随、拍照；移除抢占/跟随中更新等复杂策略 |
| v0.6 | 2026-04-15 | 补齐 status 任意状态可用、幂等重入、cancel=stop 语义；明确 json_params 不含 follow_mode；补场景跟随失败路径文档 |
