#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import TypedDict

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int32, String
from interfaces.srv import Speak


class FSMState(TypedDict):
    mode: str
    event: str


class FollowFSM:
    """LangGraph-based tiny FSM: IDLE <-> FOLLOWING."""

    def __init__(self) -> None:
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(FSMState)
        graph.add_node('transition', self._transition)
        graph.add_edge(START, 'transition')
        graph.add_edge('transition', END)
        self._app = graph.compile()

    @staticmethod
    def _transition(state: FSMState) -> FSMState:
        mode = str(state.get('mode', 'IDLE')).upper()
        event = str(state.get('event', '')).upper()

        if event == 'START':
            mode = 'FOLLOWING'
        elif event == 'STOP':
            mode = 'IDLE'

        return {'mode': mode, 'event': ''}

    def step(self, mode: str, event: str) -> str:
        out = self._app.invoke({'mode': mode, 'event': event})
        return str(out.get('mode', mode)).upper()


class FollowFSMLangGraphNode(Node):

    def __init__(self) -> None:
        super().__init__('follow_fsm_langgraph_node')

        self.declare_parameter('gesture_topic', '/person_follow/gesture_id')
        self.declare_parameter('follow_enabled_topic', '/person_follow/follow_enabled')
        self.declare_parameter('fsm_state_topic', '/person_follow/fsm_state')
        self.declare_parameter('master_enable_topic', '/person_follow/action_master_enabled')
        self.declare_parameter('fsm_force_stop_topic', '/person_follow/fsm_force_stop')

        self.declare_parameter('gesture_start_value', 1)
        self.declare_parameter('gesture_stop_value', 2)
        self.declare_parameter('gesture_switch_value', 3)
        self.declare_parameter('gesture_confirm_count', 1)
        self.declare_parameter('event_cooldown_sec', 1.0)
        self.declare_parameter('start_mode', 'IDLE')

        # whitelist target cycling: person, refrigerator, couch, chair
        self.declare_parameter('target_cycle_ids', [0, 72, 57, 56])
        self.declare_parameter('target_cycle_names_zh', ['人', '冰箱', '沙发', '椅子'])
        self.declare_parameter('target_class_id_topic', '/person_follow/target_class_id')

        # optional tts announce
        self.declare_parameter('enable_tts_announce', True)
        self.declare_parameter('speak_service', '/speak')

        self.gesture_topic = str(self.get_parameter('gesture_topic').value)
        self.follow_enabled_topic = str(self.get_parameter('follow_enabled_topic').value)
        self.fsm_state_topic = str(self.get_parameter('fsm_state_topic').value)
        self.master_enable_topic = str(self.get_parameter('master_enable_topic').value)
        self.fsm_force_stop_topic = str(self.get_parameter('fsm_force_stop_topic').value)

        self.gesture_start_value = int(self.get_parameter('gesture_start_value').value)
        self.gesture_stop_value = int(self.get_parameter('gesture_stop_value').value)
        self.gesture_switch_value = int(self.get_parameter('gesture_switch_value').value)
        self.gesture_confirm_count = max(1, int(self.get_parameter('gesture_confirm_count').value))
        self.event_cooldown_sec = max(0.0, float(self.get_parameter('event_cooldown_sec').value))

        self.target_class_id_topic = str(self.get_parameter('target_class_id_topic').value)
        self.target_cycle_ids = [int(x) for x in list(self.get_parameter('target_cycle_ids').value)]
        self.target_cycle_names_zh = [str(x) for x in list(self.get_parameter('target_cycle_names_zh').value)]
        if not self.target_cycle_ids:
            self.target_cycle_ids = [0, 72, 57, 56]
        if len(self.target_cycle_names_zh) != len(self.target_cycle_ids):
            self.target_cycle_names_zh = ['人', '冰箱', '沙发', '椅子'][:len(self.target_cycle_ids)]
            while len(self.target_cycle_names_zh) < len(self.target_cycle_ids):
                self.target_cycle_names_zh.append(str(self.target_cycle_ids[len(self.target_cycle_names_zh)]))

        self.current_target_index = 0

        self.enable_tts_announce = bool(self.get_parameter('enable_tts_announce').value)
        self.speak_service = str(self.get_parameter('speak_service').value)

        self.mode = str(self.get_parameter('start_mode').value).upper().strip()
        if self.mode not in ('IDLE', 'FOLLOWING'):
            self.mode = 'IDLE'

        self._fsm = FollowFSM()
        self._start_count = 0
        self._stop_count = 0
        self._switch_count = 0
        self._last_event_time = 0.0
        self._last_gesture_log_time = 0.0
        # one-shot latch: a continuous same gesture should trigger only once
        # until gesture changes (or returns to 0).
        self._latched_gesture = 0
        # workflow gate:
        # gesture-3 performs one target switch then forces IDLE,
        # and next gesture-3 is disabled until gesture-1 START occurs.
        self._switch_armed = True
        self._last_switch_ignored_log_time = 0.0
        self.master_enabled = True

        self.follow_enabled_pub = self.create_publisher(Bool, self.follow_enabled_topic, 10)
        self.state_pub = self.create_publisher(String, self.fsm_state_topic, 10)
        self.target_class_pub = self.create_publisher(Int32, self.target_class_id_topic, 10)
        self.create_subscription(Int32, self.gesture_topic, self._on_gesture, 10)
        self.create_subscription(Bool, self.master_enable_topic, self._on_master_enable, 10)
        self.create_subscription(Bool, self.fsm_force_stop_topic, self._on_force_stop, 10)

        self.speak_client = self.create_client(Speak, self.speak_service)

        self._publish_mode()
        self.create_timer(0.5, self._publish_mode)

        self.get_logger().info(
            f'follow_fsm_langgraph_node started. mode={self.mode} '
            f'gesture_topic={self.gesture_topic} '
            f'start={self.gesture_start_value} stop={self.gesture_stop_value} switch={self.gesture_switch_value} '
            f'confirm={self.gesture_confirm_count} target={self._target_desc()}'
        )

    def _publish_mode(self) -> None:
        enabled = Bool()
        enabled.data = (self.mode == 'FOLLOWING')
        self.follow_enabled_pub.publish(enabled)

        s = String()
        s.data = self.mode
        self.state_pub.publish(s)

        t = Int32()
        t.data = int(self.target_cycle_ids[self.current_target_index])
        self.target_class_pub.publish(t)


    def _target_desc(self) -> str:
        tid = int(self.target_cycle_ids[self.current_target_index])
        name = self.target_cycle_names_zh[self.current_target_index]
        return f'{name}(id={tid})'

    def _speak(self, text: str) -> None:
        if not self.enable_tts_announce:
            return
        if not self.speak_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn('speak service unavailable, skip tts announce')
            return
        req = Speak.Request()
        req.speech_text = text
        req.type = 'text'
        fut = self.speak_client.call_async(req)
        fut.add_done_callback(lambda _f: None)

    def _switch_target(self) -> None:
        self.current_target_index = (self.current_target_index + 1) % len(self.target_cycle_ids)
        self._publish_mode()
        desc = self._target_desc()
        self.get_logger().info(f'target switched -> {desc}')
        self._speak(f'现在识别{self.target_cycle_names_zh[self.current_target_index]}')

    def _emit_event(self, event: str, force: bool = False) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        if (not force) and ((now - self._last_event_time) < self.event_cooldown_sec):
            return

        prev = self.mode
        self.mode = self._fsm.step(self.mode, event)
        self._last_event_time = now
        self._publish_mode()

        if self.mode != prev:
            self.get_logger().info(f'FSM transition: {prev} --{event}--> {self.mode}')
        else:
            self.get_logger().info(f'FSM keep: {self.mode} (event={event})')

    def _on_master_enable(self, msg: Bool) -> None:
        self.master_enabled = bool(msg.data)
        if not self.master_enabled and self.mode != 'IDLE':
            self.get_logger().info('master disabled -> force IDLE')
            self._emit_event('STOP', force=True)

    def _on_force_stop(self, msg: Bool) -> None:
        if bool(msg.data):
            self.get_logger().info('fsm_force_stop received -> IDLE')
            self._emit_event('STOP', force=True)

    def _on_gesture(self, msg: Int32) -> None:
        g = int(msg.data)
        now = self.get_clock().now().nanoseconds / 1e9

        if (now - self._last_gesture_log_time) > 0.5 and g != 0:
            self._last_gesture_log_time = now
            self.get_logger().info(f'gesture_rx={g} mode={self.mode} target={self._target_desc()}')

        # clear / unlock when no gesture
        if g == 0:
            self._start_count = 0
            self._stop_count = 0
            self._switch_count = 0
            self._latched_gesture = 0
            return

        if not self.master_enabled:
            # total switch off: ignore all gesture controls
            return

        # one-shot latch: same continuous gesture should not retrigger
        if self._latched_gesture == g:
            return
        if self._latched_gesture != 0 and self._latched_gesture != g:
            self._latched_gesture = 0

        if g == self.gesture_start_value:
            self._start_count += 1
            self._stop_count = 0
            self._switch_count = 0
            if self._start_count >= self.gesture_confirm_count:
                self._start_count = 0
                self._emit_event('START')
                # re-arm gesture-3 switching only after START(gesture-1)
                self._switch_armed = True
                self._latched_gesture = g
            return

        if g == self.gesture_stop_value:
            self._stop_count += 1
            self._start_count = 0
            self._switch_count = 0
            if self._stop_count >= self.gesture_confirm_count:
                self._stop_count = 0
                self._emit_event('STOP')
                self._latched_gesture = g
            return

        if g == self.gesture_switch_value:
            self._switch_count += 1
            self._start_count = 0
            self._stop_count = 0
            if self._switch_count >= self.gesture_confirm_count:
                self._switch_count = 0

                if not self._switch_armed:
                    # keep log rate low for continuous hold
                    if (now - self._last_switch_ignored_log_time) > 1.0:
                        self._last_switch_ignored_log_time = now
                        self.get_logger().info('gesture-3 ignored: waiting START(gesture-1) to re-arm switch')
                else:
                    # required workflow:
                    # - gesture-3 always does one ID switch
                    # - then force FSM into IDLE and lock switch until next START
                    self._switch_target()
                    self._switch_armed = False
                    self._emit_event('STOP', force=True)
                    self.get_logger().info('gesture-3 handled: switched target then forced IDLE, wait gesture-1 to continue')

                self._latched_gesture = g
            return

        # other gesture: clear streaks
        self._start_count = 0
        self._stop_count = 0
        self._switch_count = 0


def main(args=None):
    rclpy.init(args=args)
    node = FollowFSMLangGraphNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
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
