# 실행 방법

## 1. Native ROS2/Gazebo Workflow (Docker 미사용)

### Build

```bash
cd ~/robot_ws
colcon build --packages-select follower_controller
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

### Terminal 1 — Gazebo

```bash
source /opt/ros/jazzy/setup.bash
source ~/.bashrc

gz sim ~/robot_ws/src/multi_robot_sim/worlds/dfs_corridor_world.sdf
```

### Terminal 2 — Leader Spawn

```bash
source /opt/ros/jazzy/setup.bash
source ~/.bashrc

ros2 run ros_gz_sim create -name leader \
-file ~/robot_ws/src/multi_robot_sim/models/turtlebot3_leader_sep/model.sdf \
-x -3.0 -y 0.0 -z 0.3 -Y 0
```

### Terminal 3 — Follower Spawn

```bash
source /opt/ros/jazzy/setup.bash
source ~/.bashrc

ros2 run ros_gz_sim create -name follower1 \
-file ~/robot_ws/src/multi_robot_sim/models/turtlebot3_follower_sep/model.sdf \
-x -3.8 -y 0.0 -z 0.3 -Y 0
```

### Terminal 4 — Bridge

```bash
source /opt/ros/jazzy/setup.bash

ros2 run ros_gz_bridge parameter_bridge \
/leader_cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
/follower_cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
/leader_odom@nav_msgs/msg/Odometry@gz.msgs.Odometry \
/follower_odom@nav_msgs/msg/Odometry@gz.msgs.Odometry \
/follower_scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan
```

### Terminal 5 — Follower Controller

```bash
cd ~/robot_ws

source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 run follower_controller follower_node
```

### Terminal 6 — Keyboard Teleoperation

```bash
source /opt/ros/jazzy/setup.bash

ros2 run teleop_twist_keyboard teleop_twist_keyboard \
--ros-args -r /cmd_vel:=/leader_cmd_vel
```

### Workflow Order

```text
Gazebo
→ Leader Spawn
→ Follower Spawn
→ Bridge
→ Follower Controller
→ Teleop
```

---

## 2. Docker Workflow (권장)

### Run Docker

```bash
./run_docker.sh
```

### Build Workspace

컨테이너 내부:

```bash
cd /root/robot_ws

source /opt/ros/jazzy/setup.bash

colcon build --packages-select follower_controller

source install/setup.bash
```

### Run Multi-Robot Workflow

```bash
/root/robot_ws/run_original_workflow.sh
```

### Teleop

새 host 터미널:

```bash
docker exec -it multi_robot_sim bash
```

컨테이너 내부:

```bash
source /opt/ros/jazzy/setup.bash
source /root/robot_ws/install/setup.bash

ros2 run teleop_twist_keyboard teleop_twist_keyboard \
--ros-args -r /cmd_vel:=/leader_cmd_vel
```
