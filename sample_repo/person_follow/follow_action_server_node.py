#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PersonFollow.action adapter for person follow (global enable switch).

语义（参考 dex_hand 模式）：
- action_name="start": 长时间运行的 goal。
  scene_template 模式：
    1. 发布 master_enabled=True, follow_enabled=True
    2. scene_servo_node 收到后直接开始场景跟随
    3. 到位后 scene_completed → gesture_stop_event → succeed 返回
  person_yolo 跟人（start/start:person）：
    1. 发布 master_enabled=True, follow_enabled=True
    2. scene_servo_node 等待手势1触发底盘跟随
    3. 手势2停止后发布 gesture_stop_event → succeed 返回
  person_yolo 非人目标（start:<COCO_CLASS_ID>）：
    1. 发布 target_class_id 和 follow_enabled=True
    2. scene_servo_node 直接开始跟随目标
    3. 到达 object_desired_distance_m 后发布 gesture_stop_event → succeed 返回
  共同：支持 cancel → canceled 返回
- action_name="stop":  立即强制停止，succeed 返回
- action_name="status": 查询状态，succeed 返回
"""

import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Int32, String

from interfaces.action import PersonFollow


class FollowActionServerNode(Node):
    def __init__(self) -> None:
        super().__init__('follow_action_server_node')

        # command mapping by action_name string
        self.CMD_START = 'start'
        self.CMD_STOP = 'stop'
        self.CMD_STATUS = 'status'
        self.TARGET_NAME_TO_ID = {
            'person': 0,
            'people': 0,
            'human': 0,
            '人': 0,
            'refrigerator': 72,
            'fridge': 72,
            '冰箱': 72,
            'couch': 57,
            'sofa': 57,
            '沙发': 57,
            'chair': 56,
            '椅子': 56,
        }

        # status codes for feedback/result
        self.STATUS_IDLE = 0
        self.STATUS_FOLLOWING = 1
        self.STATUS_STOPPED_BY_GESTURE = 2
        self.STATUS_CANCELLED = 3
        self.STATUS_ERROR = 4

        self.declare_parameter('action_server_name', '/person_follow/skill_behavior_tree')
        self.declare_parameter('master_enable_topic', '/person_follow/action_master_enabled')
        self.declare_parameter('action_follow_enabled_topic', '/person_follow/action_follow_enabled')
        self.declare_parameter('follow_enabled_topic', '/person_follow/follow_enabled')
        self.declare_parameter('fsm_force_stop_topic', '/person_follow/fsm_force_stop')
        self.declare_parameter('fsm_state_topic', '/person_follow/fsm_state')
        self.declare_parameter('gesture_stop_event_topic', '/person_follow/gesture_stop_event')
        self.declare_parameter('target_class_id_topic', '/person_follow/target_class_id')
        self.declare_parameter('feedback_hz', 5.0)

        self.action_server_name = str(self.get_parameter('action_server_name').value)
        self.master_enable_topic = str(self.get_parameter('master_enable_topic').value)
        self.action_follow_enabled_topic = str(self.get_parameter('action_follow_enabled_topic').value)
        self.follow_enabled_topic = str(self.get_parameter('follow_enabled_topic').value)
        self.fsm_force_stop_topic = str(self.get_parameter('fsm_force_stop_topic').value)
        self.fsm_state_topic = str(self.get_parameter('fsm_state_topic').value)
        self.gesture_stop_event_topic = str(self.get_parameter('gesture_stop_event_topic').value)
        self.target_class_id_topic = str(self.get_parameter('target_class_id_topic').value)
        self.feedback_hz = max(1.0, float(self.get_parameter('feedback_hz').value))

        self.master_enabled = True
        self.current_fsm_state = 'IDLE'
        self._gesture_stop_event = False

        # track active start goal
        self._active_start_goal = None
        self._start_time = 0.0
        self._shutdown_requested = False

        self.cb_group = ReentrantCallbackGroup()

        self.master_enable_pub = self.create_publisher(Bool, self.master_enable_topic, 10)
        self.action_follow_enabled_pub = self.create_publisher(Bool, self.action_follow_enabled_topic, 10)
        self.follow_enabled_pub = self.create_publisher(Bool, self.follow_enabled_topic, 10)
        self.fsm_force_stop_pub = self.create_publisher(Bool, self.fsm_force_stop_topic, 10)
        self.target_class_pub = self.create_publisher(Int32, self.target_class_id_topic, 10)

        self.create_subscription(String, self.fsm_state_topic, self._on_fsm_state, 10, callback_group=self.cb_group)
        self.create_subscription(Bool, self.gesture_stop_event_topic, self._on_gesture_stop_event, 10, callback_group=self.cb_group)

        self.action_server = ActionServer(
            self,
            PersonFollow,
            self.action_server_name,
            execute_callback=self._execute_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=self.cb_group,
        )

        self._publish_master_enabled(self.master_enabled)

        self.get_logger().info(
            f'follow_action_server started. action={self.action_server_name} '
            f'master_topic={self.master_enable_topic} feedback_hz={self.feedback_hz}'
        )

    def _on_fsm_state(self, msg: String) -> None:
        self.current_fsm_state = str(msg.data).upper()

    def _on_gesture_stop_event(self, msg: Bool) -> None:
        if msg.data:
            self._gesture_stop_event = True
            self.get_logger().info('gesture_stop_event received')

    def _publish_master_enabled(self, enabled: bool) -> None:
        m = Bool()
        m.data = bool(enabled)
        self.master_enable_pub.publish(m)

    def _publish_follow_enabled(self, enabled: bool) -> None:
        b = Bool()
        b.data = bool(enabled)
        self.action_follow_enabled_pub.publish(b)

    def _force_stop_follow(self) -> None:
        self._publish_follow_enabled(False)

        s = Bool()
        s.data = True
        self.fsm_force_stop_pub.publish(s)

    def _publish_target_class(self, class_id: int) -> None:
        msg = Int32()
        msg.data = int(class_id)
        self.target_class_pub.publish(msg)
        self.get_logger().info(f'target_class_id set -> {int(class_id)}')

    def _parse_action_name(self, raw: str) -> tuple[str, int | None]:
        text = raw.strip()
        if not text:
            return '', None
        parts = text.replace('=', ':').replace(',', ':').split(':', 1)
        cmd = parts[0].strip().lower()
        if len(parts) == 1:
            return cmd, None

        target = parts[1].strip().lower()
        if target.startswith('target'):
            target = target.replace('target', '', 1).strip(': =')
        if not target:
            return cmd, None
        try:
            return cmd, int(target)
        except ValueError:
            return cmd, self.TARGET_NAME_TO_ID.get(target)

    def request_shutdown(self) -> None:
        self._shutdown_requested = True
        try:
            self.master_enabled = False
            self._publish_master_enabled(False)
            self._force_stop_follow()
        except Exception:
            pass

    @staticmethod
    def _safe_goal_call(callable_obj) -> bool:
        try:
            callable_obj()
            return True
        except Exception:
            return False

    def _goal_cb(self, goal_request: PersonFollow.Goal) -> GoalResponse:
        action_name, target_class_id = self._parse_action_name(str(goal_request.action_name))
        if action_name not in (self.CMD_START, self.CMD_STOP, self.CMD_STATUS):
            self.get_logger().warn(
                f'unknown action_name={goal_request.action_name}, expected start/stop/status, reject'
            )
            return GoalResponse.REJECT
        if target_class_id is None and ':' in str(goal_request.action_name) and action_name == self.CMD_START:
            self.get_logger().warn(
                f'unknown target in action_name={goal_request.action_name}, use numeric COCO id or person/fridge/sofa/chair'
            )
            return GoalResponse.REJECT

        # If a new start goal comes while one is active, accept it
        # (the old one will be preempted in execute_cb)
        if action_name == self.CMD_START and self._active_start_goal is not None:
            self.get_logger().info('new start goal received, will preempt previous one')

        return GoalResponse.ACCEPT

    def _cancel_cb(self, _goal_handle) -> CancelResponse:
        self.get_logger().info('cancel request received')
        return CancelResponse.ACCEPT

    async def _execute_cb(self, goal_handle):
        req = goal_handle.request
        action_name, target_class_id = self._parse_action_name(str(req.action_name))

        result = PersonFollow.Result()

        if action_name == self.CMD_STOP:
            self.master_enabled = False
            self._publish_master_enabled(False)
            self._force_stop_follow()
            self.get_logger().info('STOP command: force stop follow')
            goal_handle.succeed()
            result.success = True
            result.status = self.STATUS_IDLE
            return result

        if action_name == self.CMD_STATUS:
            goal_handle.succeed()
            result.success = True
            result.status = self.STATUS_FOLLOWING if self.master_enabled and self.current_fsm_state == 'FOLLOWING' else self.STATUS_IDLE
            return result

        # === START: long-running goal ===
        self.get_logger().info('START command: enabling follow, waiting for stop/cancel...')

        # Enable master
        self.master_enabled = True
        self._publish_master_enabled(True)
        if target_class_id is None:
            target_class_id = 0
        if target_class_id is not None:
            self._publish_target_class(target_class_id)
        self._publish_follow_enabled(True)
        self._active_start_goal = goal_handle
        self._start_time = time.time()
        self._gesture_stop_event = False
        feedback = PersonFollow.Feedback()
        sleep_dt = 1.0 / self.feedback_hz
        was_following = False

        try:
            while rclpy.ok() and not self._shutdown_requested:
                # Check cancel
                if goal_handle.is_cancel_requested:
                    self.get_logger().info('goal cancelled by client')
                    self._force_stop_follow()
                    self.master_enabled = False
                    self._publish_master_enabled(False)
                    self._safe_goal_call(goal_handle.canceled)
                    result.success = False
                    result.status = self.STATUS_CANCELLED
                    return result

                # Check if preempted by a newer goal
                if self._active_start_goal is not goal_handle:
                    self.get_logger().info('goal preempted by newer start goal')
                    self._safe_goal_call(goal_handle.abort)
                    result.success = False
                    result.status = self.STATUS_CANCELLED
                    return result

                # Check gesture stop event (always returns result, like dex_hand pattern)
                if self._gesture_stop_event:
                    self._gesture_stop_event = False
                    self.get_logger().info('gesture_stop_event: completing goal')
                    self.master_enabled = False
                    self._publish_master_enabled(False)
                    feedback.action_percentage = 100.0
                    feedback.status = self.STATUS_STOPPED_BY_GESTURE
                    try:
                        goal_handle.publish_feedback(feedback)
                    except Exception:
                        pass
                    self._safe_goal_call(goal_handle.succeed)
                    result.success = True
                    result.status = self.STATUS_STOPPED_BY_GESTURE
                    return result

                fsm_state = self.current_fsm_state

                # Calculate progress
                elapsed = time.time() - self._start_time

                if fsm_state == 'FOLLOWING':
                    was_following = True
                    feedback.action_percentage = min(99.0, elapsed / 10.0 * 100.0)
                    feedback.status = self.STATUS_FOLLOWING
                elif was_following and fsm_state == 'IDLE':
                    # Was following, now IDLE → gesture 2 stopped it
                    self.get_logger().info('FSM transitioned FOLLOWING -> IDLE (gesture stop detected)')
                    feedback.action_percentage = 100.0
                    feedback.status = self.STATUS_STOPPED_BY_GESTURE
                    try:
                        goal_handle.publish_feedback(feedback)
                    except Exception:
                        pass

                    # Succeed: normal completion
                    self._safe_goal_call(goal_handle.succeed)
                    result.success = True
                    result.status = self.STATUS_STOPPED_BY_GESTURE
                    self.get_logger().info('goal succeeded: stopped by gesture 2')
                    return result
                else:
                    # Still IDLE, waiting for gesture 1 to trigger FOLLOWING
                    feedback.action_percentage = 0.0
                    feedback.status = self.STATUS_IDLE

                try:
                    goal_handle.publish_feedback(feedback)
                except Exception:
                    pass
                time.sleep(sleep_dt)

        except Exception as e:
            if not self._shutdown_requested and rclpy.ok():
                self.get_logger().error(f'execute error: {e}')
            self._safe_goal_call(goal_handle.abort)
            result.success = False
            result.status = self.STATUS_ERROR
            return result
        finally:
            if self._active_start_goal is goal_handle:
                self._active_start_goal = None

        # Fallback: rclpy shutdown
        self._safe_goal_call(goal_handle.abort)
        result.success = False
        result.status = self.STATUS_CANCELLED if self._shutdown_requested else self.STATUS_ERROR
        return result


def main(args=None):
    rclpy.init(args=args)
    node = FollowActionServerNode()
    executor = MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        node.request_shutdown()
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
