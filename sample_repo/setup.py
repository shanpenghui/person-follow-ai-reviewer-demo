from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'person_follow'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(include=['person_follow', 'person_follow.*', 'scene_servo', 'scene_servo.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # TensorRT engine files for runtime inference
        (os.path.join('share', package_name, 'models'), glob('models/*.engine')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dev',
    maintainer_email='dev@localhost',
    description='MIRA2 person follow (head visual servo)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'person_follow_node = person_follow.person_follow_node:main',
            'person_association_node = person_follow.person_association_node:main',
            'follow_fsm_langgraph_node = person_follow.follow_fsm_langgraph_node:main',
            'gesture_detector_node = person_follow.gesture_detector_node:main',
            'speak_pyttsx3_node = person_follow.speak_pyttsx3_node:main',
            'follow_action_server_node = person_follow.follow_action_server_node:main',
            'scene_servo_node = scene_servo.scene_servo_node:main',
            'scene_template_servo_node = scene_servo.scene_template_servo_node:main',
            'person_yolo_servo_node = scene_servo.person_yolo_servo_node:main',
            # Test / debug tools (not launched in production)
            'd455_depth_accuracy_test = person_follow.d455_depth_accuracy_test:main',
        ],
    },
)
