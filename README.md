# PDFS Multi-Robot Simulation

**Physical DFS (PDFS) 기반 Communication-Aware Multi-Robot Exploration Simulator**

ROS2 Jazzy + Gazebo Harmonic + Docker 기반 다중 로봇 탐색 연구 플랫폼

---

# Project Overview

본 프로젝트는 **Physical DFS (Physical Depth-First Search)** 기반 다중 로봇 탐색을 목표로 하는 연구용 시뮬레이션 환경이다.

현재 구현은 **Physical DFS 자체가 아닌**, 향후 Physical DFS 구현을 위한 기반 기술인

* Leader-Follower Navigation
* Communication-Aware Relay Placement
* Multi-Follower Relay Handoff
* Topology Extraction
* Junction-Aware Relay Placement

까지 완료된 상태이다.

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

Implemented:

* Communication Range 기반 Relay Placement
* Junction 우선 Relay Placement
* Corner 기반 Relay Placement
* Barrier 예상 위치 Relay Placement

---

## Verified Through Simulation

실험 로그를 통해 다음 기능의 동작을 확인하였다.

* Follower1의 Breadcrumb 기반 추종
* Distance-Based Relay Placement
* Relay Candidate → Relay Placement 상태 전환
* Relay Event 발행
* Multi-Follower Relay Handoff
* Follower2 자동 Explorer 활성화
* PASS_RELAY 기반 Relay 통과
* Topology Extraction
* Dead-End Detection

---

## Current Limitations

현재 구현은 Relay Placement 및 Topology Extraction 단계까지 완료된 상태이며, 다음과 같은 한계가 존재한다.

* Relay 배치 위치의 최적성 미검증
* 실제 통신 유지 성능 향상 여부 미평가
* 다양한 환경에서의 반복 실험 및 성능 비교 미수행
* Follower2의 PASS_RELAY 동작 안정성 추가 개선 필요
* Dead-End 감지는 가능하지만 Backtracking 기능 미구현
* Junction 정보를 Graph 형태로 저장하지 않음
* DFS Stack 및 Branch 관리 기능 미구현
* Physical DFS 탐색 알고리즘 미구현

---

# System Architecture

초기 상태

```text
Leader
   |
   v
Follower1 (Explorer)
   |
   v
Follower2 (Idle)
```

Relay Placement 발생 후

```text
Leader
   |
   v
Follower1 (Stationary Relay)

Follower2 (Explorer)

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

---

# Future Work

## Phase 1 — Topology Graph Construction

* Junction → DFS Node 생성
* Node 중복 제거
* Graph Edge 생성
* Topology Graph 구축

## Phase 2 — Physical DFS

* Branch Selection
* DFS Stack
* Dead-End Detection
* Backtracking

## Phase 3 — Communication-Aware Physical DFS

* Relay Chain Maintenance
* Distributed Exploration
* Multi-Robot Physical DFS

---

# Current Development Stage

```text
Docker Environment                 완료
Leader-Follower Navigation         완료
Distance-Based Relay Placement     완료
Relay Handoff                      완료
Topology Extraction                완료
Junction-Aware Relay Placement     초기 구현 완료

Topology Graph                     미구현
Backtracking                       미구현
DFS Stack                          미구현
Physical DFS                       미구현
```

---

# Last Update

```text
2026.06.02
```
