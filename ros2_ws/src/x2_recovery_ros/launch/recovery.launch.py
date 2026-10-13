
from launch import LaunchDescription
from launch_ros.actions import Node


CONDA_PYTHON = '/home/islam/miniconda3/envs/frasa/bin/python'


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='x2_recovery_ros',
            executable='recovery_node',
            name='recovery_node',
            output='screen',
            prefix=CONDA_PYTHON,
        ),

        Node(
            package='x2_recovery_ros',
            executable='telemetry_node',
            name='telemetry_node',
            output='screen',
        ),
    ])
