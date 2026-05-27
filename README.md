
# PDFS Multi-Robot Simulation

ROS2 Jazzy + Gazebo Harmonic 기반 Dockerized Leader-Follower Multi-Robot Simulation

---

# 프로젝트 개요

본 프로젝트는 ROS2 Jazzy와 Gazebo Harmonic 환경에서 동작하는 멀티로봇 leader-follower 시뮬레이션 시스템.

Docker 기반으로 구성되어 환경 재현성을 확보하였으며, 다음 기능들을 포함.

- TurtleBot3 기반 멀티로봇 시뮬레이션
- Leader-Follower 자율 추종
- Breadcrumb 기반 waypoint tracking
- ROS2 ↔ Gazebo bridge
- Keyboard teleoperation
- Launch 자동화
- Dockerized reproducible environment

---

# 시스템 구조

```text
Leader Robot
    ↓
/leader/cmd_vel

Follower Controller
    ↓
/follower_cmd_vel

cmd_vel_relay_node
    ↓
/follower/cmd_vel

Gazebo Follower Robot
```

---

# 실행 환경

- Ubuntu 22.04+
- Docker
- X11 GUI 환경

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

---

# 멀티로봇 시뮬레이션 실행

Docker 컨테이너 내부에서:

```bash
cd /root/robot_ws

source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch follower_controller full_leader_follower.launch.py
```

---

# Keyboard Teleoperation

새 터미널에서:

```bash
docker exec -it multi_robot_sim bash
```

컨테이너 내부에서:

```bash
source /opt/ros/jazzy/setup.bash
source /root/robot_ws/install/setup.bash

ros2 run teleop_twist_keyboard teleop_twist_keyboard \
--ros-args --remap cmd_vel:=/leader/cmd_vel
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

# 현재 구현된 기능

- 실시간 leader-follower 시뮬레이션
- Breadcrumb 기반 경로 추종
- ROS2 topic namespace 분리
- Dockerized 실행 환경
- Gazebo 멀티로봇 시뮬레이션
- Launch 자동화

---

# 프로젝트 구조

```text
src/follower_controller/
├── follower_controller/
├── launch/
├── models/
├── worlds/
└── setup.py
```

---

# 향후 계획

- Multi-follower 확장
- 장애물 회피 검증
- Communication range 실험
- Junction detection
- DFS exploration
- Relay robot placement

---
---
# PDFS Multi-Robot Simulation

ROS2 Jazzy + Gazebo Harmonic 기반 Dockerized Leader-Follower Multi-Robot Simulation

## Features

- Dockerized ROS2 Jazzy environment
- Gazebo Harmonic simulation
- TurtleBot3 multi-robot setup
- Leader-Follower autonomous tracking
- Breadcrumb path following
- ROS2 ↔ Gazebo bridge
- Keyboard teleoperation
- Launch automation

---

# System Architecture

```text
Leader Robot
    ↓
/leader/cmd_vel

Follower Controller
    ↓
/follower_cmd_vel

cmd_vel_relay_node
    ↓
/follower/cmd_vel

Gazebo Follower Robot
```

---

# Requirements

- Ubuntu 22.04+
- Docker
- X11 GUI support

---

# Installation

Clone repository:

```bash
git clone https://github.com/gaheebella/PDFS-Multi-Robot.git
cd PDFS-Multi-Robot
```

Build Docker image:

```bash
docker build -t multi-robot-sim .
```

---

# Run Docker Container

```bash
./run_docker.sh
```

---

# Launch Multi-Robot Simulation

Inside Docker container:

```bash
cd /root/robot_ws

source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch follower_controller full_leader_follower.launch.py
```

---

# Keyboard Teleoperation

Open a new terminal:

```bash
docker exec -it multi_robot_sim bash
```

Inside container:

```bash
source /opt/ros/jazzy/setup.bash
source /root/robot_ws/install/setup.bash

ros2 run teleop_twist_keyboard teleop_twist_keyboard \
--ros-args --remap cmd_vel:=/leader/cmd_vel
```

---

# Teleop Keys

```text
i : forward
, : backward
j : rotate left
l : rotate right
k : stop
```

---

# Current Capabilities

- Real-time leader-follower simulation
- Breadcrumb-based path tracking
- ROS2 topic namespace separation
- Dockerized reproducible environment
- Gazebo multi-robot simulation

---

# Project Structure

```text
src/follower_controller/
├── follower_controller/
├── launch/
├── models/
├── worlds/
└── setup.py
```

---

# Future Work

- Multi-follower extension
- Obstacle avoidance validation
- Communication range experiments
- Junction detection
- DFS exploration
- Relay robot placement

---

# Author

Gahee Yang
