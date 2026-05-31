# PDFS Multi-Robot Simulation

Physical DFS (PDFS) 기반 Communication-Aware Multi-Robot Exploration Simulator

ROS2 Jazzy + Gazebo Harmonic + Docker 기반 다중 로봇 탐색 연구 플랫폼

---

# Project Overview

본 프로젝트는 Physical DFS(Physical Depth-First Search) 기반 다중 로봇 탐색을 목표로 하는 연구용 시뮬레이션 환경이다.

현재 구현된 시스템은 다음 기능을 포함한다.

* ROS2 Jazzy + Gazebo Harmonic 기반 시뮬레이션
* Docker 기반 재현 가능한 연구 환경
* Leader-Follower Multi-Robot Architecture
* Breadcrumb-Based Path Following
* Communication-Aware Relay Placement
* Multi-Follower Relay Handoff
* LaserScan-Based Topology Extraction
* Junction-Aware Relay Placement
* ROS2 ↔ Gazebo Bridge
* Keyboard Teleoperation

---

# Current Research Progress

## Completed

### Multi-Robot Simulation

* Leader Robot
* Follower1
* Follower2
* ROS2 ↔ Gazebo Integration
* Dockerized Execution Environment

### Navigation

* Breadcrumb-Based Waypoint Tracking
* Lookahead Path Following
* Obstacle Avoidance
* Leader-Follower Tracking

### Communication-Aware Relay Placement

Implemented:

* Distance-Based Relay Placement
* Relay Candidate State
* Stationary Relay State
* Relay Handoff Mechanism
* Multi-Follower Activation

### Topology Extraction

LaserScan 기반 환경 구조 인식

Supported Topologies:

* CORRIDOR
* CORNER
* T_JUNCTION
* CROSS
* DEAD_END

### Junction-Aware Relay Placement

지원 기능:

* Communication Range 기반 배치
* Junction 우선 배치
* Barrier 예상 위치 배치

---

# System Architecture

```text
Leader
   |
   v
Follower1 (Explorer)
   |
   v
Follower2 (Idle)

↓

Relay Placement

↓

Follower1 → Stationary Relay

↓

Follower2 → Explorer

↓

Relay Chain Formation
```

---

# Repository Structure

```text
robot_ws/
├── src/
│   ├── follower_controller/
│   └── multi_robot_sim/
├── run_original_workflow.sh
├── Dockerfile
└── run_docker.sh
```

---

# Execution Guide

## 1. Native ROS2/Gazebo Workflow (Without Docker)

### Build

```bash
cd ~/robot_ws

colcon build --packages-select follower_controller

source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

---

### Terminal 1 — Gazebo

```bash
source /opt/ros/jazzy/setup.bash
source ~/.bashrc

gz sim ~/robot_ws/src/multi_robot_sim/worlds/dfs_corridor_world.sdf
```

---

### Terminal 2 — Leader Spawn

```bash
source /opt/ros/jazzy/setup.bash
source ~/.bashrc

ros2 run ros_gz_sim create -name leader \
-file ~/robot_ws/src/multi_robot_sim/models/turtlebot3_leader_sep/model.sdf \
-x -3.0 -y 0.0 -z 0.3 -Y 0
```

---

### Terminal 3 — Follower Spawn

```bash
source /opt/ros/jazzy/setup.bash
source ~/.bashrc

ros2 run ros_gz_sim create -name follower1 \
-file ~/robot_ws/src/multi_robot_sim/models/turtlebot3_follower_sep/model.sdf \
-x -3.8 -y 0.0 -z 0.3 -Y 0
```

---

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

---

### Terminal 5 — Follower Controller

```bash
cd ~/robot_ws

source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 run follower_controller follower_node
```

---

### Terminal 6 — Keyboard Teleoperation

```bash
source /opt/ros/jazzy/setup.bash

ros2 run teleop_twist_keyboard teleop_twist_keyboard \
--ros-args -r /cmd_vel:=/leader_cmd_vel
```

---

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

# 2. Docker Workflow (Recommended)

### Run Docker

```bash
./run_docker.sh
```

---

### Build Workspace

Inside Container:

```bash
cd /root/robot_ws

source /opt/ros/jazzy/setup.bash

colcon build --packages-select follower_controller

source install/setup.bash
```

---

### Run Multi-Robot Workflow

```bash
/root/robot_ws/run_original_workflow.sh
```

---

### Keyboard Teleoperation

Open a new host terminal:

```bash
docker exec -it multi_robot_sim bash
```

Inside container:

```bash
source /opt/ros/jazzy/setup.bash
source /root/robot_ws/install/setup.bash

ros2 run teleop_twist_keyboard teleop_twist_keyboard \
--ros-args -r /cmd_vel:=/leader_cmd_vel
```

---

# Future Work

## Phase 1

Topology Graph Construction

* DFS Node Creation
* Node Merging
* Graph Edge Generation

## Phase 2

Physical DFS

* Branch Selection
* DFS Stack
* Backtracking

## Phase 3

Communication-Aware Physical DFS

* Relay Chain Maintenance
* Distributed Exploration
* Multi-Robot Physical DFS

---

# Author

Gahee Yang
DRI Lab Multi-Robot Exploration Research
