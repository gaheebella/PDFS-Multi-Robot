#!/bin/bash

source /opt/ros/jazzy/setup.bash
source /root/robot_ws/install/setup.bash

echo "[1] Start Gazebo world"
gz sim /root/robot_ws/src/multi_robot_sim/worlds/dfs_corridor_world.sdf &
sleep 5

echo "[2] Spawn leader"
ros2 run ros_gz_sim create -name leader \
-file /root/robot_ws/src/multi_robot_sim/models/turtlebot3_leader_sep/model.sdf \
-x -3.0 -y 0.0 -z 0.3 -Y 0

sleep 2

echo "[3] Spawn follower1"
ros2 run ros_gz_sim create -name follower1 \
-file /root/robot_ws/src/multi_robot_sim/models/turtlebot3_follower_sep/model.sdf \
-x -3.8 -y 0.0 -z 0.3 -Y 0

sleep 2

echo "[4] Start bridge"
ros2 run ros_gz_bridge parameter_bridge \
/leader_cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
/follower_cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
/leader_odom@nav_msgs/msg/Odometry@gz.msgs.Odometry \
/follower_odom@nav_msgs/msg/Odometry@gz.msgs.Odometry \
/follower_scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan &

sleep 3

echo "[5] Start follower controller"
ros2 run follower_controller follower_node
