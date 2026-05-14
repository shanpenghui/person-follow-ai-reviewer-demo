"""
MIRA2 Person Follow 一键启动

默认优先读取:
1. $ROBOT_PRODUCT_PARAMS_DIR/config_person_follow.yaml
2. share/person_follow/config/person_follow_all.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _default_config_file() -> str:
    package_default = os.path.join(
        get_package_share_directory('person_follow'),
        'config',
        'person_follow_all.yaml',
    )
    robot_product_params_dir = os.environ.get('ROBOT_PRODUCT_PARAMS_DIR', '')
    if robot_product_params_dir:
        product_config = os.path.join(
            robot_product_params_dir,
            'config_person_follow.yaml',
        )
        if os.path.exists(product_config):
            return product_config
    return package_default


def _default_scene_template_file() -> str:
    return os.environ.get('PERSON_FOLLOW_SCENE_TEMPLATE', '')


def generate_launch_description():
    config_file = LaunchConfiguration('config_file')
    enable_scene_template_servo_bridge = LaunchConfiguration('enable_scene_template_servo_bridge')
    enable_person_follow_detector = LaunchConfiguration('enable_person_follow_detector')
    scene_template_template_path = LaunchConfiguration('scene_template_template_path')

    common_parameters = [config_file]

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=_default_config_file(),
            description='person_follow unified parameter yaml',
        ),
        DeclareLaunchArgument(
            'enable_scene_template_servo_bridge',
            default_value='false',
            description='enable the integrated scene-template servo path (scene_servo_node + bridge)',
        ),
        DeclareLaunchArgument(
            'enable_person_follow_detector',
            default_value='true',
            description='enable original yolo-based person_follow_node',
        ),
        DeclareLaunchArgument(
            'scene_template_template_path',
            default_value=_default_scene_template_file(),
            description='scene-template JSON path for scene_servo_node',
        ),

        Node(
            package='person_follow',
            executable='speak_pyttsx3_node',
            name='speak_pyttsx3_node',
            output='screen',
            parameters=common_parameters,
        ),

        Node(
            package='person_follow',
            executable='gesture_detector_node',
            name='gesture_detector_node',
            output='screen',
            parameters=common_parameters,
        ),

        Node(
            package='person_follow',
            executable='follow_fsm_langgraph_node',
            name='follow_fsm_langgraph_node',
            output='screen',
            parameters=common_parameters,
        ),

        Node(
            package='person_follow',
            executable='follow_action_server_node',
            name='follow_action_server_node',
            output='screen',
            parameters=common_parameters,
        ),

        Node(
            package='person_follow',
            executable='scene_servo_node',
            name='scene_servo_node',
            output='screen',
            parameters=[config_file, {'template_path': scene_template_template_path}],
            condition=IfCondition(enable_scene_template_servo_bridge),
        ),

        Node(
            package='person_follow',
            executable='scene_template_servo_bridge_node',
            name='scene_template_servo_bridge_node',
            output='screen',
            parameters=common_parameters,
            condition=IfCondition(enable_scene_template_servo_bridge),
        ),

        Node(
            package='person_follow',
            executable='person_follow_node',
            name='person_follow_node',
            output='screen',
            parameters=common_parameters,
            condition=IfCondition(enable_person_follow_detector),
        ),

        Node(
            package='person_follow',
            executable='person_association_node',
            name='person_association_node',
            output='screen',
            parameters=common_parameters,
        ),

        Node(
            package='person_follow',
            executable='chassis_follow_node',
            name='mira2_chassis_follow_node',
            output='screen',
            parameters=common_parameters,
        ),
    ])
