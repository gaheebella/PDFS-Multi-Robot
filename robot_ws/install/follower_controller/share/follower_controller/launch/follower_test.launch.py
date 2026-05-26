from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='follower_controller',
            executable='follower_node',
            name='follower_controller',
            output='screen'
        )
    ])
