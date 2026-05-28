# PDFS Multi-Robot Simulation

ROS2 Jazzy + Gazebo Harmonic 기반 Dockerized Leader-Follower Multi-Robot Simulation 프로젝트입니다.

---

# 프로젝트 개요

본 프로젝트는 Docker 환경 내부에서 ROS2 Jazzy와 Gazebo Harmonic을 이용하여 TurtleBot3 기반 멀티로봇 leader-follower 시뮬레이션을 실행합니다.

현재 구현된 기능:

* TurtleBot3 기반 멀티로봇 시뮬레이션
* Keyboard teleoperation 기반 Leader 제어
* Leader-Follower autonomous tracking
* Breadcrumb 기반 waypoint tracking
* ROS2 ↔ Gazebo bridge
* Dockerized reproducible environment
* DFS corridor world

---

# 시스템 구조

```text
Keyboard Teleop
        ↓
/leader_cmd_vel
        ↓
Leader Robot
        ↓
Leader Odometry
        ↓
Follower Controller
        ↓
/follower_cmd_vel
        ↓
Follower Robot
```

---

# 실행 환경

* Ubuntu 22.04+
* Docker
* X11 GUI support

Host PC에는 Docker만 설치되어 있으면 됩니다.

ROS2 Jazzy와 Gazebo Harmonic은 Docker 컨테이너 내부에서 실행됩니다.

---

# 설치 방법

Repository clone:

```bash
git clone https://github.com/gaheebella/PDFS-Multi-Robot.git
cd PDFS-Multi-Robot
```

Docker image build:

```bash
docker build -t multi-robot-sim .
```

---

# Docker 실행

```bash
./run_docker.sh
```

컨테이너 내부 프롬프트:

```bash
root@...:~/robot_ws#
```

---

# Workspace Build

컨테이너 내부에서:

```bash
cd /root/robot_ws

source /opt/ros/jazzy/setup.bash

colcon build --packages-select follower_controller

source install/setup.bash
```

---

# Run Multi-Robot Simulation

컨테이너 내부에서:

```bash
/root/robot_ws/run_original_workflow.sh
```

이 스크립트는 아래 과정을 자동 실행합니다.

```text
1. Gazebo world 실행
2. Leader robot 생성
3. Follower robot 생성
4. ROS2-Gazebo bridge 실행
5. Follower controller 실행
```

---

# Keyboard Teleoperation

새 host 터미널에서:

```bash
docker exec -it multi_robot_sim bash
```

컨테이너 내부에서:

```bash
source /opt/ros/jazzy/setup.bash
source /root/robot_ws/install/setup.bash

ros2 run teleop_twist_keyboard teleop_twist_keyboard \
--ros-args -r /cmd_vel:=/leader_cmd_vel
```

---

# Teleop 조작 키

```text
i : 전진
, : 후진
j : 좌회전
l : 우회전
k : 정지
```

---

# 주요 ROS2 Topics

```text
/leader_cmd_vel
/follower_cmd_vel
/leader_odom
/follower_odom
/follower_scan
```

---

# 현재 구현된 기능

* Dockerized ROS2 Jazzy + Gazebo Harmonic 환경
* TurtleBot3 기반 leader-follower 시뮬레이션
* Keyboard-controlled leader robot
* Autonomous follower tracking
* Breadcrumb waypoint following
* ROS2 ↔ Gazebo bridge integration
* DFS corridor Gazebo world

---

# 프로젝트 구조

```text
robot_ws/src/
├── follower_controller/
│   ├── follower_controller/
│   │   └── follower_node.py
│   ├── launch/
│   ├── package.xml
│   └── setup.py
│
└── multi_robot_sim/
    ├── models/
    │   ├── turtlebot3_leader_sep/
    │   └── turtlebot3_follower_sep/
    └── worlds/
        └── dfs_corridor_world.sdf
```

---

# 향후 계획

* Multi-follower extension
* Obstacle avoidance validation
* Communication range experiments
* Junction detection
* DFS exploration
* Relay robot placement

---

# Author

Gahee Yang
