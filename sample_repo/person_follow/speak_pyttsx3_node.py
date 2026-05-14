#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading

import pyttsx3
import rclpy
from interfaces.srv import Speak
from rclpy.node import Node


class SpeakPyttsx3Node(Node):
    """Open-source local TTS service node using pyttsx3.

    Provides service: /speak (interfaces/srv/Speak)
    Request:
      - speech_text: string
      - type: string (optional, ignored for now)
    Response:
      - feedback.success / feedback.msg
    """

    def __init__(self) -> None:
        super().__init__('speak_pyttsx3_node')

        self.declare_parameter('service_name', '/speak')
        self.declare_parameter('rate', 170)
        self.declare_parameter('volume', 1.0)
        # Prefer explicit id because pyttsx3/driver language matching is unreliable.
        self.declare_parameter('voice_id', 'sit/cmn')
        self.declare_parameter('voice_name_contains', 'chinese')

        service_name = str(self.get_parameter('service_name').value)
        rate = int(self.get_parameter('rate').value)
        volume = float(self.get_parameter('volume').value)
        voice_id = str(self.get_parameter('voice_id').value).strip()
        voice_name_contains = str(self.get_parameter('voice_name_contains').value).strip().lower()

        self._lock = threading.Lock()
        self._engine = pyttsx3.init()

        try:
            self._engine.setProperty('rate', rate)
        except Exception:
            pass
        try:
            self._engine.setProperty('volume', max(0.0, min(1.0, volume)))
        except Exception:
            pass

        selected_voice = None
        try:
            voices = self._engine.getProperty('voices')

            # 1) exact id first
            if voice_id:
                for v in voices:
                    if str(getattr(v, 'id', '')) == voice_id:
                        self._engine.setProperty('voice', getattr(v, 'id', ''))
                        selected_voice = getattr(v, 'id', '')
                        break

            # 2) fallback by fuzzy name/id
            if selected_voice is None and voice_name_contains:
                for v in voices:
                    name = (getattr(v, 'name', '') or '').lower()
                    vid = (getattr(v, 'id', '') or '').lower()
                    if voice_name_contains in name or voice_name_contains in vid:
                        self._engine.setProperty('voice', getattr(v, 'id', ''))
                        selected_voice = getattr(v, 'id', '')
                        break

            # 3) final fallback: keep default
            cur = self._engine.getProperty('voice')
            if not selected_voice:
                selected_voice = cur

            self.get_logger().info(f'TTS voice selected: {selected_voice}')
        except Exception as e:
            self.get_logger().warn(f'voice selection failed: {e}')

        self._srv = self.create_service(Speak, service_name, self._on_speak)
        self.get_logger().info(
            f'speak_pyttsx3_node started. service={service_name} rate={rate} volume={volume}'
        )

    def _on_speak(self, request: Speak.Request, response: Speak.Response) -> Speak.Response:
        text = str(request.speech_text).strip()
        if not text:
            response.feedback.success = False
            response.feedback.msg = 'empty speech_text'
            return response

        try:
            with self._lock:
                self._engine.say(text)
                self._engine.runAndWait()
            response.feedback.success = True
            response.feedback.msg = 'ok'
            self.get_logger().info(f'TTS: {text}')
        except Exception as e:
            response.feedback.success = False
            response.feedback.msg = f'tts failed: {e}'
            self.get_logger().error(response.feedback.msg)

        return response


def main(args=None):
    rclpy.init(args=args)
    node = SpeakPyttsx3Node()
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
