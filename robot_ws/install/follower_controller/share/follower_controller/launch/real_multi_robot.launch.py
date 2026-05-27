from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():

    gazebo = ExecuteProcess(
        cmd=[
            'gz', 'sim', '-v', '4', '-r',
	    '/root/robot_ws/src/multi_robot_sim/worlds/dfs_corridor_world.sdf'
	],
        output='screen'
    )

    leader_spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'leader',
            '-file', '/root/robot_ws/src/multi_robot_sim/models/turtlebot3_leader_sep/model.sdf',
            '-x', '3.0',
            '-y', '0.0',
            '-z', '0.01'
        ],
        output='screen'
    )

    follower_spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'follower',
            '-file', '/root/robot_ws/src/multi_robot_sim/models/turtlebot3_follower_sep/model.sdf',
            '-x', '-1.0',
            '-y', '0.0',
            '-z', '0.01'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        TimerAction(period=3.0, actions=[leader_spawn]),
        TimerAction(period=5.0, actions=[follower_spawn]),
    ])
