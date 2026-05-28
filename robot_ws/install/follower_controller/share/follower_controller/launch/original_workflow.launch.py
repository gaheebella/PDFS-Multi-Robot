from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():

    # 1. Gazebo world 실행
    gazebo = ExecuteProcess(
        cmd=[
            'gz', 'sim',
            '/root/robot_ws/src/multi_robot_sim/worlds/dfs_corridor_world.sdf'
        ],
        output='screen'
    )

    # 2. Leader robot 생성
    leader_spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'leader',
            '-file', '/root/robot_ws/src/multi_robot_sim/models/turtlebot3_leader_sep/model.sdf',
            '-x', '-3.0',
            '-y', '0.0',
            '-z', '0.3',
            '-Y', '0'
        ],
        output='screen'
    )

    # 3. Follower robot 생성
    follower_spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'follower1',
            '-file', '/root/robot_ws/src/multi_robot_sim/models/turtlebot3_follower_sep/model.sdf',
            '-x', '-3.8',
            '-y', '0.0',
            '-z', '0.3',
            '-Y', '0'
        ],
        output='screen'
    )

    # 4. ROS2-Gazebo bridge 실행
    bridge = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            '/leader_cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/follower_cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/leader_odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            '/follower_odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            '/follower_scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan'
        ],
        output='screen'
    )

    # 5. Follower controller 실행
    follower_controller = Node(
        package='follower_controller',
        executable='follower_node',
        output='screen'
    )

    return LaunchDescription([
        gazebo,

        # Gazebo world가 먼저 뜬 뒤 순차적으로 실행
        TimerAction(period=3.0, actions=[leader_spawn]),
        TimerAction(period=5.0, actions=[follower_spawn]),
        TimerAction(period=7.0, actions=[bridge]),
        TimerAction(period=9.0, actions=[follower_controller]),
    ])
