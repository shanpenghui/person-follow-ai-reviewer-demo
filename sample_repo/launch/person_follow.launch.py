from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='person_follow',
            executable='person_follow_node',
            name='person_follow_node',
            output='screen',
            parameters=[{
                'torso_action_name': '/Torso/torso_action_service',
                'joint_state_topic': '/Torso/joint_states',

                'image_width': 640.0,
                'image_height': 480.0,
                'control_hz': 15.0,

                # D435 color FOV
                'fov_h_deg': 69.0,
                'fov_v_deg': 42.0,

                # 死区
                'deadband': 0.06,

                # 方向
                'yaw_sign': 1.0,
                'pitch_sign': 1.0,

                # bbox 跟随点
                'bbox_target_x_ratio': 0.5,
                'bbox_target_y_ratio': 0.30,

                # 增益（0.5 = 每帧消除50%偏差，指数收敛）
                'gain': 0.5,

                # 速度自适应
                'vel_min': 0.06,
                'vel_max': 0.50,
                'vel_ramp_deg': 15.0,

                # 命令周期
                'command_period_sec': 0.07,

                # 误差低通
                'error_ema_alpha': 0.55,

                # 丢目标
                'lost_timeout_sec': 1.5,
                'lost_behavior': 'hold',
                'lost_home_max_velocity': 0.10,
                'stop_on_lost': False,

                # 限位
                'yaw_min_deg': -90.0,
                'yaw_max_deg': 90.0,
                'pitch_min_deg': -34.0,
                'pitch_max_deg': 19.0,

                # home
                'home_before_start': True,
                'home_torso_height': 0.149,
                'home_torso_yaw': 0.0,
                'home_head_pitch': 0.0,
                'home_head_yaw': 0.0,
                'home_max_velocity': 0.15,
                'home_settle_sec': 1.0,

                'log_period_sec': 1.0,
            }]
        )
    ])
