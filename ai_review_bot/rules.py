from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewRule:
    rule_id: str
    title: str
    severity: str
    path_keywords: tuple[str, ...]
    code_keywords: tuple[str, ...]
    why_it_matters: str
    review_prompt: str
    test_suggestions: tuple[str, ...]


RULES: tuple[ReviewRule, ...] = (
    ReviewRule(
        rule_id="motion-safety",
        title="底盘/头部运动控制改动需要安全约束",
        severity="high",
        path_keywords=("scene_servo", "person_yolo_servo_node.py", "scene_template_servo_node.py"),
        code_keywords=("Twist", "cmd_vx", "cmd_wz", "_publish_cmd", "clamp", "ramp", "head", "yaw", "pitch"),
        why_it_matters="该模块会直接影响机器人底盘和头部运动，速度限幅、停止指令和状态复位遗漏都可能导致实机风险。",
        review_prompt="检查是否保留速度限幅、加速度斜坡、丢失目标后的 0 速度发布，以及关闭跟随时的 head home 行为。",
        test_suggestions=(
            "构造目标丢失帧，断言 vx/wz 在下一 tick 被置为 0。",
            "构造远距离目标，断言 vx 不超过配置的最大速度。",
            "构造横向偏差，断言 wz 经过限幅且符号正确。",
        ),
    ),
    ReviewRule(
        rule_id="depth-distance",
        title="深度/距离逻辑需要覆盖无效值和边界值",
        severity="high",
        path_keywords=("depth", "d455", "servo_estimator.py"),
        code_keywords=("depth", "distance", "desired_distance", "object_desired_distance", "nan", "isfinite"),
        why_it_matters="跟随距离依赖 D455 深度或估计距离，NaN、0、过近、过远都容易让控制量异常。",
        review_prompt="检查深度值是否过滤 NaN/Inf/0，目标距离误差是否使用 deadband，并确认最终速度不会反向冲撞。",
        test_suggestions=(
            "深度为 NaN/Inf/0 时不发布前进速度。",
            "距离落在 deadband 内时 vx 为 0。",
            "物体目标和 person 目标分别使用正确的期望距离。",
        ),
    ),
    ReviewRule(
        rule_id="gesture-fsm",
        title="手势状态机改动需要防重复触发",
        severity="medium",
        path_keywords=("gesture", "follow_fsm", "gesture_detector_node.py", "follow_fsm_langgraph_node.py"),
        code_keywords=("gesture", "_latched_gesture", "gesture_confirm_count"),
        why_it_matters="手势识别会有连续帧抖动，缺少确认帧或 latch 会造成 start/stop/switch 重复触发。",
        review_prompt="检查 gesture-1/2/3 是否仍有确认帧、冷却时间、连续同手势 latch，以及 master disable 时是否清状态。",
        test_suggestions=(
            "连续 gesture-1 只触发一次 START。",
            "gesture-2 在 FOLLOWING 时停止，并在 IDLE 时不重复发 stop。",
            "切换目标 gesture-3 后需要等待下一次 START 才继续。",
        ),
    ),
    ReviewRule(
        rule_id="action-contract",
        title="Action 接口改动需要保护外部调用契约",
        severity="medium",
        path_keywords=("follow_action_server_node.py", "launch", "README.md"),
        code_keywords=("action_name", "start:", "status", "success", "target_class", "PersonFollow"),
        why_it_matters="行为树和外部调度系统依赖固定的 action_name/status 语义，兼容性破坏会导致上层无法控制跟随任务。",
        review_prompt="检查 start/stop/status、start:person、start:<COCO_ID> 是否向后兼容，status code 是否保持文档一致。",
        test_suggestions=(
            "start:person 发布 target_class_id=0 且等待手势启动。",
            "start:72 直接进入物体跟随并使用 object_desired_distance。",
            "stop 后 action_master_enabled 和 follow_enabled 都被置 false。",
        ),
    ),
    ReviewRule(
        rule_id="target-association",
        title="目标关联/锁定改动需要回放验证",
        severity="medium",
        path_keywords=("person_association_node.py", "target_lock", "handover", "association"),
        code_keywords=("track_id", "iou", "handover", "lock", "score", "bbox"),
        why_it_matters="目标关联会影响机器人是否跟错人，单帧逻辑正确不代表跨帧稳定。",
        review_prompt="检查目标锁定是否有跨帧稳定性、遮挡恢复、多人切换和低置信度过滤策略。",
        test_suggestions=(
            "两人交叉场景下 target_id 不应频繁切换。",
            "目标短暂遮挡后应优先恢复原锁定对象。",
            "低置信度检测不应覆盖高置信度锁定目标。",
        ),
    ),
)


def is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return "/test" in normalized or normalized.startswith("tests/") or normalized.endswith("_test.py") or normalized.startswith("tools/")
