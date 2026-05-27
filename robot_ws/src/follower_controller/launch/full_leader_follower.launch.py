from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            '/root/robot_ws/src/follower_controller/launch/real_multi_robot.launch.py'
        )
    )

    bridge = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',

            '/leader/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            '/follower/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            '/follower/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/leader/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/follower/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist'
        ],
        output='screen'
    )

    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            '/root/robot_ws/src/follower_controller/launch/controller_with_relay.launch.py'
        )
    )

    cmd_vel_relay = Node(
        package='follower_controller',
        executable='cmd_vel_relay_node',
        output='screen'
    )

    return LaunchDescription([

        gazebo_launch,

        TimerAction(
            period=5.0,
            actions=[bridge]
        ),

        TimerAction(
            period=8.0,
            actions=[controller_launch]
        ),

        TimerAction(
            period=10.0,
            actions=[cmd_vel_relay]
        ),
    ])
