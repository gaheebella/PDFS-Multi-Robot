from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    turtlebot3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                turtlebot3_gazebo_dir,
                'launch',
                'turtlebot3_world.launch.py'
            )
        )
    )

    follower_controller_node = Node(
        package='follower_controller',
        executable='follower_node',
        name='follower_controller',
        output='screen'
    )

    topic_relay_node = Node(
        package='follower_controller',
        executable='topic_relay_node',
        name='topic_relay_node',
        output='screen'
    )

    return LaunchDescription([
        gazebo_launch,
        topic_relay_node,
        follower_controller_node,
    ])
