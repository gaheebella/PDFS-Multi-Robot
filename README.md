# Pressure-Driven SPH Physical DFS

## 1. Project Overview

A Pygame-based prototype for Physical Depth-First Search using an SPH-based multi-robot swarm in a single-junction environment

Integration of the following functions while maintaining multiple robots as a single continuous fluid-like swarm:

* SPH-based swarm motion
* Inter-robot equilibrium spacing based on Kelvin–Voigt coupling
* Fully distributed branch consensus among NORMAL robots
* Junction Anchor election and distributed consensus storage
* Base communication maintenance using Breadcrumb Relays
* Adaptive Shepherd election
* Dead-end saturation detection
* Pressure-driven backtracking using Shepherd virtual pressure
* Branch Relay recovery and complete swarm return to Base
* Automated experimental metric recording

---

## 2. Research Objective

Implementation of the logical exploration sequence of conventional DFS through the physical motion of a multi-robot swarm

Simultaneous achievement of the following objectives:

* Complete exploration of every branch
* Communication connectivity between the Base and the swarm
* Branch decision-making without centralized control
* Prevention of inter-robot collisions and swarm separation
* Fluid-like branch entry and backtracking
* Breadcrumb Relay deployment only where required
* Release of all temporary roles and return of all robots to Base after exploration

---

## 3. Current Implementation Scope

A cross-shaped map consisting of one junction and three dead-end branches

```text
                 UP
                  │
                  │
LEFT ─────── Junction ───────────── RIGHT
                  │
                  │
                 BASE
```

Current implementation:

* Single Junction `J0`
* Three branches
* Fixed exploration priority: `RIGHT → UP → LEFT`
* Branch entry, saturation, backtracking, and completion handling
* Final gathering at the Junction and return to Base

Not currently implemented:

* Recursive multi-junction DFS
* DFS tree repair across multiple junctions
* Real-time topology construction in unknown environments
* Dynamic replanning for changing obstacles

State separation based on `JunctionState` and `BranchEdge` for future multi-junction extensions

---

## 4. Core Design Principles

### 4.1 Continuous SPH Swarm

Treatment of robots as a continuous fluid-like swarm rather than independent path-following agents

* Density estimation using an SPH kernel
* Pressure computation from density differences
* Swarm expansion and compression through pressure forces
* Velocity-difference reduction through artificial viscosity
* Collision avoidance through short-range repulsion
* Weak goal-direction guidance using an Euclidean Distance Field
* Wall collision constraints and corridor-centering control

Priority on pressure-driven flow instead of strong path-following forces

---

### 4.2 Kelvin–Voigt Viscoelastic Coupling

Viscoelastic inter-robot coupling for spacing and velocity stabilization within the SPH swarm

Components:

* Equilibrium-distance restoration using a spring term
* Relative-velocity damping using a dashpot term
* Adaptive adjustment of inter-robot equilibrium distance
* Local velocity consensus with neighboring robots
* Automatic removal of persistently inactive connections
* Acceleration filtering for prevention of abrupt force changes

Primary purposes:

* Prevention of excessive swarm compression
* Reduction of unnecessary inter-robot oscillations
* Maintenance of swarm connectivity in long corridors
* Suppression of abrupt dispersion during pressure release
* Stabilization of backtracking flow

---

### 4.3 Fully Distributed Branch Consensus

Branch decisions made by NORMAL robots inside the Junction rather than by the Anchor

Distributed consensus process:

1. Local branch-state inspection by NORMAL robots entering the Junction
2. Exchange of branch votes among neighboring NORMAL robots
3. Local selection among unvisited branches according to priority
4. Iterative consensus based on neighboring votes
5. Distributed decision confirmation after quorum satisfaction
6. Storage of the consensus result in the Anchor
7. Opening of only the selected branch and closure of all non-selected branches

Fixed priority:

```text
RIGHT → UP → LEFT
```

Proxy Partition and SPH Rollout costs used for physical branch analysis and logging, while the final exploration order remains determined by fixed-priority consensus among NORMAL robots

---

## 5. Robot Roles

| Role           | Function                                                                  |
| -------------- | ------------------------------------------------------------------------- |
| `NORMAL`       | SPH swarm formation, branch voting, exploration, and backtracking         |
| `ANCHOR`       | Storage and retransmission of Junction state and NORMAL robot consensus   |
| `RELAY`        | Breadcrumb communication maintenance inside a branch                      |
| `TRUNK_RELAY`  | Communication maintenance between the Base and Junction                   |
| `SHEPHERD`     | Virtual boundary formation and pressure-driven backtracking at a dead end |
| `PRE_SHEPHERD` | Preselected Shepherd candidate for the next branch                        |

Temporary election of special roles from NORMAL robots according to the current situation

---

## 6. Junction Anchor

### 6.1 Anchor Responsibilities

Preservation of distributed consensus state rather than centralized decision-making

* Storage of branch consensus produced by NORMAL robots
* Storage of branch visitation states
* Storage of virtual gate states
* Retransmission of selected-branch information
* Management of Junction-specific state sequence numbers
* Communication-aware parking behavior
* Release of the Anchor role during the final return

Functions not performed by the Anchor:

* Independent branch selection
* Centralized robot-role assignment
* Centralized control of the entire swarm
* Arbitrary modification of the DFS order

---

### 6.2 Anchor Candidate Selection

Registration of NORMAL robots entering the Junction Anchor region as candidates

Cost evaluation for candidate–parking-slot combinations after satisfaction of a minimum candidate count or waiting-time condition

Evaluation criteria:

| Cost Component        | Weight | Meaning                                                            |
| --------------------- | -----: | ------------------------------------------------------------------ |
| Arrival cost          |   0.25 | Time of entry into the Anchor region                               |
| Parking-distance cost |   0.25 | Distance from the candidate position to the parking slot           |
| Flow-obstruction cost |   0.30 | Interference caused by the candidate path and parking position     |
| Communication cost    |   0.20 | Number of visible communication neighbors and communication margin |

Election of the candidate and Junction-edge parking-slot combination with the minimum total cost

Tie-breaking order:

1. Minimum total cost
2. Earlier entry into the Anchor region
3. Smaller Robot ID

---

### 6.3 Anchor Parking

Selection of a minimum-cost position from four candidate parking slots near the Junction corners

Parking-position criteria:

* No overlap with the primary flow toward the selected branch
* Minimal interference with NORMAL robots during Anchor movement
* High line-of-sight communication quality with nearby robots
* No obstruction of the central Junction flow

Communication-safety control for limiting Anchor movement when Base connectivity becomes unstable

---

## 7. Branch States and Virtual Gates

DFS state of each branch:

| State       | Meaning                            |
| ----------- | ---------------------------------- |
| `UNVISITED` | Unexplored branch                  |
| `ACTIVE`    | Branch currently under exploration |
| `VISITED`   | Completely explored branch         |

Virtual gate state at each branch entrance:

| State    | Meaning                                 |
| -------- | --------------------------------------- |
| `OPEN`   | Robot movement permitted                |
| `CLOSED` | Entry blocked through virtual repulsion |

Operations after branch selection:

* Opening of the selected branch entrance
* Closure of non-selected branch entrances
* Removal of remaining robots from the previous branch during branch switching
* Closure of all branch gates during the final return to Base
* One-way passage control in the return direction

---

## 8. Branch Analysis Structure

Proxy Partition and SPH Rollout structures for physical movement-cost analysis despite the fixed exploration order

### 8.1 Proxy Partition

Division of free space around the Junction into temporary branch-specific Proxy Regions

Partition criteria:

* Required robot count for each branch
* Local SPH density mass
* Free-space distance to the branch entrance
* Target mass ratio for each Proxy Region
* Assignment of actual robots to Proxy Regions

Virtual analysis regions for individual evaluation of branch candidates rather than simultaneous physical distribution of robots across multiple branches

---

### 8.2 SPH Rollout

Short-horizon predictive simulation using virtual particles for each candidate branch

Evaluation components:

* Expected forward flow
* Density disturbance
* Velocity disturbance
* Wall-collision risk
* Inter-robot collision risk
* Communication-connectivity risk
* Required Relay count
* Stiffness-mode transitions
* Stabilization cost
* Travel distance
* Proxy mass shortage
* Swarm-shape alignment
* Existing flow direction
* Branch-entrance congestion
* Backtracking distance
* Turning cost

Candidate comparison data recorded in the HUD and CSV log

---

## 9. Breadcrumb Relay Communication

### 9.1 Communication Structure

Line-of-sight multi-hop communication structure rooted at the fixed Base

```text
Base
  ↓
Trunk Relay
  ↓
Junction Anchor or rear swarm
  ↓
Branch Breadcrumb Relay
  ↓
Front exploration swarm
```

Communication criteria:

* Inter-robot distance
* Maximum communication range
* Safe communication distance
* Wall-based line-of-sight obstruction
* Connectivity path to Base
* Distance to the parent robot
* Remaining communication margin

---

### 9.2 Reactive Breadcrumb Deployment

Relay deployment during swarm movement rather than predeployment of the complete chain

Deployment process:

1. Inspection of communication conditions at the front of the swarm
2. Inspection of the connectivity ratio and parent distance
3. Detection of potential communication disconnection
4. Search for a Relay candidate among NORMAL robots at the rear of the swarm
5. Deployment of one Breadcrumb Relay at the required position
6. Update of the multi-hop connection to Base

Conversion of only the required robots into the `RELAY` role without leaving static NORMAL guard robots

---

### 9.3 Relay Recovery

Sequential recovery of Branch Relays during backtracking, beginning with Relays closest to the Junction

Recovery conditions:

* Availability of an alternative communication connection
* No obstruction of branch swarm movement
* Maintenance of a safe state for a predefined duration
* Secured connection between the next Relay and Base

Sequential release of Trunk Relays from the Junction side during the final return to Base

---

## 10. Shepherd System

### 10.1 Adaptive Shepherd Count

Automatic calculation of the required Shepherd count according to the corridor width

Limits:

```text
Minimum: 5 robots
Maximum: 14 robots
```

Shepherd count factors:

* Actual corridor width
* Safety margin from corridor walls
* Target spacing between Shepherd slots
* Minimum number of robots required to cover the corridor width

---

### 10.2 Shepherd Election

Beginning of Shepherd election after front robots reach the Capture Region near the dead end

Election process:

1. Detection of NORMAL robots inside the Capture Region
2. Calculation of the required Shepherd count
3. Local candidate-to-slot cost evaluation
4. Candidate assignment to Shepherd slots
5. Formation of a lateral boundary in front of the dead-end wall
6. Application of a virtual Curtain for accumulating NORMAL robots behind the Shepherd boundary

Local-information-based role election without centralized commands from the Base

---

### 10.3 Saturation Detection

Detection of sufficient robot accumulation behind the Shepherd boundary

Saturation criteria:

* Number of robots inside the dead-end region
* Ratio of low-speed robots
* Average density relative to the reference density
* Grid-based spatial occupancy
* Motion stagnation at the swarm front
* Saturation-condition persistence time
* Minimum accumulated robot count according to the Shepherd count

Multi-condition saturation detection combining density, velocity, spatial occupancy, and stagnation

---

### 10.4 Pressure-Driven Backtracking

Piston-like Shepherd movement for transmitting reverse pressure through the swarm after saturation detection

Process:

1. Accumulation of robots behind the Shepherd boundary
2. Confirmation of the saturation state
3. Increase in Shepherd virtual pressure
4. Generation of reverse flow among NORMAL robots
5. Evaluation of the moving-robot ratio and average velocity
6. Confirmation of established backtracking flow
7. Release of the Shepherd role and reintegration into the swarm
8. Sequential recovery of Branch Relays
9. Pressure transmission toward the next branch or Base

Swarm backtracking through combined SPH pressure and Shepherd virtual pressure

---

### 10.5 Pre-Shepherd Pipeline

Advance preparation of Shepherd candidates for the next branch while the current branch is undergoing backtracking

Primary functions:

* Advance election of candidates inside the next branch
* Preliminary formation of the Shepherd boundary
* Inspection of robot accumulation behind the boundary
* Removal of remaining robots from the previous branch
* Immediate pressure Push after branch switching
* Reduction of Junction waiting time

Pipeline structure for connecting the flow between the previous and next branches

---

## 11. Overall State Machine

```text
MOVE_TO_JUNCTION
        ↓
EXPLORE_BRANCH
        ↓
FORM_SHEPHERD_BOUNDARY
        ↓
FILL_BEHIND_SHEPHERD
        ↓
PRESSURE_PUSH
        ↓
FLOW_BACKTRACK
        ↓
JUNCTION_SWITCH
        ↓
Next branch available
   ├─ Yes → EXPLORE_BRANCH
   └─ No  → FINAL_JUNCTION_GATHER
                         ↓
                   RETURN_TO_BASE
                         ↓
                        DONE
```

### Functions by State

| State                    | Function                                                                           |
| ------------------------ | ---------------------------------------------------------------------------------- |
| `MOVE_TO_JUNCTION`       | Base decompression, Junction movement, distributed voting, and Anchor election     |
| `EXPLORE_BRANCH`         | Entry into the selected branch, Relay deployment, and Shepherd candidate detection |
| `FORM_SHEPHERD_BOUNDARY` | Shepherd boundary formation in front of the dead-end wall                          |
| `FILL_BEHIND_SHEPHERD`   | Swarm accumulation behind Shepherds and saturation detection                       |
| `PRESSURE_PUSH`          | Application of Shepherd piston pressure                                            |
| `FLOW_BACKTRACK`         | Reverse swarm flow and Relay recovery                                              |
| `JUNCTION_SWITCH`        | Consensus for the next unvisited branch                                            |
| `FINAL_JUNCTION_GATHER`  | Removal of remaining branch robots and temporary roles                             |
| `RETURN_TO_BASE`         | Anchor release and sequential Trunk Relay recovery                                 |
| `DONE`                   | Completion of exploration and return                                               |

---

## 12. Overall Algorithm Flow

```text
High-density robot initialization inside the Base
→ SPH pressure storage through initial compression
→ Compression release and expansion toward the Junction
→ Reactive Breadcrumb Relay deployment when communication spacing increases
→ Branch consensus among NORMAL robots inside the Junction
→ Minimum-cost Junction Anchor election
→ Storage of the consensus result and virtual gate states by the Anchor
→ Opening of the selected branch
→ Closure of non-selected branches using virtual gates
→ SPH swarm movement toward the selected branch
→ Additional Breadcrumb Relay deployment under communication risk
→ Arrival of front robots at the dead-end Capture Region
→ Shepherd count calculation based on corridor width
→ Shepherd election through a local candidate auction
→ Formation of a lateral Shepherd boundary
→ Accumulation of NORMAL robots behind the Shepherd boundary
→ Saturation detection based on velocity, density, occupancy, and stagnation
→ Initiation of Shepherd piston virtual pressure
→ Formation of SPH pressure-driven backtracking flow
→ Formation of a temporary communication bridge between the Base and the backtracking swarm
→ Release of Shepherd roles and reintegration into the swarm
→ Sequential recovery of Branch Breadcrumb Relays
→ Release of the temporary communication bridge
→ Pressure transmission toward the next branch
→ Advance preparation of the next branch using PRE_SHEPHERD robots
→ Repetition in RIGHT, UP, and LEFT order
→ Completion of all branch visits
→ Final gathering at the Junction
→ Release of the Anchor role
→ Sequential recovery of remaining Relays
→ Application of a one-way return gate
→ Return of all robots to the Base
→ Transition to the DONE state
→ Experimental log storage
```

---

## 13. Main Parameters

### Swarm and SPH Parameters

| Parameter                 |          Default | Meaning                                              |
| ------------------------- | ---------------: | ---------------------------------------------------- |
| `ROBOT_COUNT`             |              680 | Total number of robots                               |
| `SMOOTHING_LENGTH`        | `22 × MAP_SCALE` | SPH neighbor-search radius                           |
| `PRESSURE_GAIN`           |           2800.0 | Pressure-force magnitude                             |
| `STIFFNESS_EXPONENT`      |              0.5 | Stiffness exponent in the pressure equation of state |
| `DAMPING`                 |              4.0 | Velocity damping                                     |
| `REPULSION_GAIN`          |            260.0 | Short-range collision-repulsion gain                 |
| `MOTION_SPEED_MULTIPLIER` |              3.0 | Global movement-speed multiplier                     |

### Kelvin–Voigt Parameters

| Parameter                              | Default | Meaning                              |
| -------------------------------------- | ------: | ------------------------------------ |
| `VISCOELASTIC_ELASTIC_GAIN`            |    42.0 | Spring restoring-force gain          |
| `VISCOELASTIC_DASHPOT_GAIN`            |     8.0 | Relative-velocity damping gain       |
| `VISCOELASTIC_EQUILIBRIUM_ADAPTATION`  |     4.0 | Equilibrium-distance adaptation rate |
| `VISCOELASTIC_VELOCITY_CONSENSUS_GAIN` |     6.0 | Local velocity-consensus gain        |

### Communication Parameters

| Parameter             |                     Default | Meaning                     |
| --------------------- | --------------------------: | --------------------------- |
| `COMM_RANGE`          |            `54 × MAP_SCALE` | Maximum communication range |
| `COMM_SAFE_DISTANCE`  |            `40 × MAP_SCALE` | Safe communication distance |
| `TRUNK_RELAY_SPACING` |            `30 × MAP_SCALE` | Target Trunk Relay spacing  |
| `RELAY_SPACING`       |            `30 × MAP_SCALE` | Target Branch Relay spacing |
| `BREADCRUMB_SPACING`  | `COMM_SAFE_DISTANCE × 0.55` | Target Breadcrumb spacing   |

### Shepherd and Saturation Parameters

| Parameter                    | Default | Meaning                                            |
| ---------------------------- | ------: | -------------------------------------------------- |
| `SHEPHERD_MIN_COUNT`         |       5 | Minimum Shepherd count                             |
| `SHEPHERD_MAX_COUNT`         |      14 | Maximum Shepherd count                             |
| `SATURATION_MIN_TIP_ROBOTS`  |      18 | Minimum front-robot count for saturation detection |
| `SATURATION_DENSITY_RATIO`   |    1.02 | Default saturation-density ratio                   |
| `SATURATION_OCCUPANCY_RATIO` |    0.16 | Default spatial-occupancy threshold                |
| `VIRTUAL_PRESSURE_FORCE`     |   110.0 | Shepherd virtual-pressure magnitude                |

---

## 14. Experimental Metrics

Automatic generation of the following CSV file upon simulation completion or user termination:

```text
sph_dfs_experiment_summary.csv
```

Recorded metrics:

* Termination reason
* Total robot count
* Total simulation time
* Total robot travel distance
* Travel distance in the NORMAL role
* Travel distance in the RELAY role
* Travel distance in the TRUNK_RELAY role
* Travel distance in the SHEPHERD role
* Travel distance in the ANCHOR role
* Communication-disconnection robot-seconds
* Minimum inter-robot distance
* Number of safety-distance violations
* Actual branch visitation order
* Number of branch-selection events
* Branch-cost components for each candidate
* Number of saturation-detection events
* Number of pressure-Push events

---

## 15. HUD Information

Real-time information displayed in the right-side HUD panel:

* Current FPS and simulation phase
* Initial Base compression and pressure-release state
* Distributed consensus result among NORMAL robots
* Anchor ID, election cost, stored state, and communication state
* Virtual gate state for each branch
* Branch visitation order and DFS state
* Mass ratio of each Proxy Region
* SPH Rollout cost for each branch candidate
* Number of Base-connected robots and maximum hop count
* Breadcrumb Relay count and front communication ratio
* Counts of NORMAL, RELAY, and SHEPHERD robots inside the branch
* Saturation-detection metrics
* Shepherd-formation state
* Number of active Kelvin–Voigt connections
* SPH, EDF, Shepherd, and pressure-release forces
* Total travel distance and accumulated communication-disconnection time

---

## 16. Execution Environment

### Requirements

* Python 3.10 or later
* Pygame

### Package Installation

```bash
pip install pygame
```

### Execution

```bash
python single_junction_sph_dfs.py
```

---

## 17. Controls

| Key     | Function                                            |
| ------- | --------------------------------------------------- |
| `SPACE` | Pause or resume the simulation                      |
| `R`     | Reset the simulation                                |
| `D`     | Toggle density-color visualization                  |
| `V`     | Toggle Proxy Region and analysis-area visualization |
| `C`     | Toggle communication-link visualization             |
| `ESC`   | Exit the simulation                                 |

---

## 18. Visualization Colors

| Object             | Color        |
| ------------------ | ------------ |
| NORMAL robot       | Navy blue    |
| ANCHOR             | Bright green |
| SHEPHERD           | Purple       |
| Branch Relay       | Brown        |
| Trunk Relay        | Dark brown   |
| Disconnected robot | Red          |
| RIGHT Branch       | Orange       |
| UP Branch          | Blue         |
| LEFT Branch        | Purple       |
| Virtual gate       | Red          |

---

## 19. Code Structure

```text
1. Display
2. Cross Map
3. State and Branch Metadata
4. Physics and Control Parameters
5. Map Mask and Region Checks
6. General Utilities
7. Experiment Metrics
8. Base Station and Robot
9. Robot Creation and Spatial Hash
10. Base-Rooted Communication
11. Reactive Tail Breadcrumb Communication Trail
12. Anchor Election and Branch Analysis
12-1. Junction Stability Consensus
13. Saturation Detector
14. Adaptive Shepherd Election and Pressure Flow
15. SPH
16. State Machine
17. Initialization
18. Main Loop
```

---

## 20. Future Extensions

* Multi-junction topology
* Independent Anchor election at each Junction
* Recursive DFS tree construction
* Parent–child Junction state transmission
* Recovery of failed branch states
* DFS tree repair after communication disconnection
* Dynamic-obstacle environments
* Integration with ROS 2 and Gazebo robot models
* Automated repeated experiments across parameter settings
* Performance comparison across algorithms
* Application of actual TurtleBot communication ranges and movement speeds

---

## 21. Limitations

* Research prototype for a single-junction environment
* Use of a known and static map
* Two-dimensional Pygame physics environment
* No representation of real sensor noise or actuator errors
* No representation of communication delays or packet loss
* Fixed branch priority
* Proxy Rollout results not directly used for actual branch-order optimization
* SPH-based swarm-control model rather than a Computational Fluid Dynamics model
* Inter-robot viscoelastic control coupling rather than a physical Kelvin–Voigt material model


---
---
# Pressure-Driven SPH Physical DFS

## 1. 프로젝트 개요

SPH 기반 다중 로봇 군집을 이용한 단일 Junction Physical DFS 시뮬레이션

다수의 로봇을 하나의 연속적인 유체 군집으로 유지하면서 다음 기능을 결합한 Pygame 기반 프로토타입

* SPH 기반 군집 이동
* Kelvin–Voigt 기반 로봇 간 평형 간격 유지
* NORMAL 로봇 간 완전분산형 Branch 합의
* Junction Anchor 선출 및 분산 합의 저장
* Breadcrumb Relay 기반 Base 통신 유지
* 적응형 Shepherd 선출
* 막다른 길 포화 감지
* Shepherd 가상압력 기반 Backtracking
* Branch Relay 회수 및 전체 로봇 Base 복귀
* 실험 지표 자동 기록

---

## 2. 연구 목적

기존 DFS의 논리적 탐색 순서를 실제 다중 로봇 군집의 물리적 이동으로 구현하기 위한 연구

주요 목표는 다음과 같은 조건의 동시 만족

* 모든 Branch의 완전탐색
* Base–군집 간 통신 연결 유지
* 중앙 통제 없는 Branch 의사결정
* 로봇 간 충돌 및 군집 분리 방지
* Branch 내부 진입과 복귀 과정의 유체적 이동
* 필요한 위치에만 Breadcrumb Relay 배치
* 탐색 완료 후 모든 특수 역할 해제 및 Base 복귀

---

## 3. 현재 구현 범위

현재 지도는 하나의 Junction과 세 개의 막다른 Branch로 구성된 십자형 구조

```text
                 UP
                  │
                  │
LEFT ─────── Junction ───────────── RIGHT
                  │
                  │
                 BASE
```

현재 구현 범위

* 단일 Junction `J0`
* Branch 세 개
* 고정 탐색 우선순위 `RIGHT → UP → LEFT`
* 각 Branch의 진입, 포화, Backtracking 및 완료 처리
* 최종 Junction 집결 및 Base 복귀

현재 미구현 범위

* 재귀적 다중 Junction DFS
* 다중 Junction 간 DFS Tree 복구
* 미지 환경에서의 실시간 Topology 생성
* 장애물 변화에 따른 동적 경로 재계획

다중 Junction 확장을 위해 `JunctionState`와 `BranchEdge` 단위의 상태 분리 구조 적용

---

## 4. 핵심 설계 원칙

### 4.1 연속적인 SPH 군집

로봇을 개별 경로 추종 객체가 아니라 하나의 연속적인 유체 군집으로 취급

* SPH 커널 기반 밀도 계산
* 밀도 차이에 따른 압력 계산
* 압력력 기반 군집 팽창과 압축
* 인공점성 기반 속도 차이 완화
* 근거리 반발력 기반 충돌 방지
* EDF 기반 약한 목표 방향 유도
* 벽 충돌 제한 및 통로 중심 유지

강한 경로 추종력보다 압력 흐름을 우선하는 구조

---

### 4.2 Kelvin–Voigt 점탄성 결합

SPH 군집 내부의 로봇 간 간격과 속도 안정화를 위한 점탄성 연결

구성 요소

* Spring 항을 이용한 평형거리 복원
* Dashpot 항을 이용한 상대속도 감쇠
* 로봇 간 평형거리의 적응적 조정
* 주변 로봇 속도에 대한 국소 속도 합의
* 오래 유지되지 않은 연결의 자동 제거
* 힘의 급격한 변화 방지를 위한 가속도 필터

주요 목적

* 군집의 과도한 압축 방지
* 로봇 간 불필요한 진동 감소
* 긴 통로에서 군집 연결 유지
* 압력 해제 과정의 급격한 분산 억제
* Backtracking 과정의 흐름 안정화

---

### 4.3 완전분산형 Branch 합의

Branch 결정 주체는 Anchor가 아니라 Junction 내부의 NORMAL 로봇

분산 합의 과정

1. Junction에 진입한 NORMAL 로봇의 로컬 Branch 상태 확인
2. 주변 NORMAL 로봇과 Branch 투표 교환
3. 방문하지 않은 Branch 중 로컬 우선순위 선택
4. 주변 투표 결과를 반영한 반복적 합의
5. 정족수 충족 시 분산 합의 확정
6. 합의 결과를 Anchor에 저장
7. 선택 Branch만 개방하고 나머지 Branch 폐쇄

고정 우선순위

```text
RIGHT → UP → LEFT
```

현재 코드의 Proxy Partition과 SPH Rollout 비용은 후보 Branch의 물리적 특성 분석과 로그 기록을 위한 구조이며, 최종 탐색 순서는 NORMAL 로봇의 고정 우선순위 합의 결과를 기준으로 결정

---

## 5. 로봇 역할

| 역할             | 기능                                         |
| -------------- | ------------------------------------------ |
| `NORMAL`       | SPH 군집 형성, Branch 투표, 탐색 및 Backtracking 참여 |
| `ANCHOR`       | Junction 상태와 NORMAL 합의 결과의 저장 및 재전송        |
| `RELAY`        | Branch 내부 Breadcrumb 통신 연결 유지              |
| `TRUNK_RELAY`  | Base와 Junction 사이의 통신 연결 유지                |
| `SHEPHERD`     | 막다른 길에서 가상 경계 형성 및 압력 Backtracking 유도      |
| `PRE_SHEPHERD` | 다음 Branch에서 미리 형성되는 Shepherd 후보군           |

특수 역할은 고정 로봇이 아니라 상황에 따라 NORMAL 로봇 중에서 선출되는 임시 역할

---

## 6. Junction Anchor

### 6.1 Anchor 역할

Anchor의 핵심 역할은 의사결정이 아닌 분산 합의 상태의 보존

* NORMAL 로봇이 결정한 Branch 합의 저장
* Branch별 방문 상태 저장
* 가상 게이트 상태 저장
* 선택 Branch 정보 재전송
* Junction별 상태 순서 번호 관리
* Base 통신 상태를 고려한 정차
* 최종 복귀 시 Anchor 역할 해제

Anchor가 직접 수행하지 않는 기능

* Branch 단독 선택
* 로봇 역할 중앙 할당
* 군집 전체의 중앙 통제
* DFS 순서 임의 변경

---

### 6.2 Anchor 후보 선정

Junction Anchor 영역에 진입한 NORMAL 로봇을 후보로 등록

최소 후보 수 또는 대기시간 조건 충족 후 후보–주차 위치 조합의 비용 계산

평가 항목

| 비용 항목    |  가중치 | 의미                                 |
| -------- | ---: | ---------------------------------- |
| 진입 비용    | 0.25 | Anchor 영역 진입 시점                    |
| 주차 거리 비용 | 0.25 | 후보 위치에서 주차 슬롯까지의 거리                |
| 흐름 방해 비용 | 0.30 | 이동 경로와 주차 위치가 Junction 흐름을 방해하는 정도 |
| 통신 비용    | 0.20 | 주변 가시 통신 이웃 수와 통신 여유 거리            |

총비용이 가장 작은 후보와 Junction 가장자리 주차 슬롯의 조합을 Anchor로 선출

동률 발생 시 적용 순서

1. 최소 총비용
2. 빠른 Anchor 영역 진입 시점
3. 작은 Robot ID

---

### 6.3 Anchor 주차

Junction 모서리 주변의 네 개 후보 슬롯 중 최소비용 위치 선택

주차 위치 선정 기준

* 선택 Branch의 주 흐름과 겹치지 않는 위치
* Anchor 이동 중 NORMAL 로봇과의 간섭이 적은 위치
* 주변 로봇과의 LOS 통신 품질이 높은 위치
* Junction 중앙 흐름을 막지 않는 위치

Base 연결이 불안정한 경우 Anchor 이동을 제한하는 통신 안전 제어 적용

---

## 7. Branch 상태와 가상 게이트

각 Branch의 DFS 상태

| 상태          | 의미              |
| ----------- | --------------- |
| `UNVISITED` | 미탐색 Branch      |
| `ACTIVE`    | 현재 탐색 중인 Branch |
| `VISITED`   | 탐색 완료 Branch    |

각 Branch 입구의 가상 게이트 상태

| 상태       | 의미             |
| -------- | -------------- |
| `OPEN`   | 로봇 이동 허용       |
| `CLOSED` | 가상 반발력으로 진입 차단 |

Branch 선택 후 적용되는 동작

* 선택 Branch 입구 개방
* 비선택 Branch 입구 폐쇄
* Branch 전환 중 이전 Branch의 잔여 로봇 배출
* 최종 Base 복귀 시 모든 Branch 게이트 폐쇄
* 복귀 방향의 일방향 통과 제어

---

## 8. Branch 분석 구조

현재 탐색 순서는 고정되어 있지만, 각 후보 Branch의 물리적 이동 비용을 분석하기 위한 Proxy Partition과 SPH Rollout 구조 포함

### 8.1 Proxy Partition

Junction 주변의 자유공간을 Branch별 임시 Proxy Region으로 분할

분할 기준

* Branch별 요구 로봇량
* 지역별 SPH 밀도 질량
* Branch 입구까지의 자유공간 거리
* Proxy Region별 목표 질량 비율
* 실제 로봇의 Proxy Region 할당

실제 로봇을 여러 Branch로 동시에 보내는 방식이 아니라 후보 Branch를 개별 평가하기 위한 가상 분석 영역

---

### 8.2 SPH Rollout

각 후보 Branch에 대해 짧은 시간 동안 가상 입자를 이동시키는 예측 시뮬레이션

평가 요소

* 예상 전방 흐름
* 밀도 교란
* 속도 교란
* 벽 충돌 위험
* 로봇 충돌 위험
* 통신 연결 위험
* 필요한 Relay 수
* 강성 모드 변화
* 안정화 비용
* 이동 거리
* Proxy 질량 부족
* 군집 형상 정렬
* 기존 흐름 방향
* Branch 입구 혼잡
* Backtracking 거리
* 방향 전환 비용

분석 결과는 HUD와 CSV 로그에 기록되는 후보별 비교 정보

---

## 9. Breadcrumb Relay 통신

### 9.1 통신 구조

고정 Base를 루트로 사용하는 LOS 기반 다중 홉 통신 구조

```text
Base
  ↓
Trunk Relay
  ↓
Junction Anchor 또는 후방 군집
  ↓
Branch Breadcrumb Relay
  ↓
전방 탐색 군집
```

통신 가능 여부 판단 요소

* 로봇 간 거리
* 최대 통신 범위
* 안전 통신 거리
* 벽에 의한 LOS 차단
* Base까지의 연결 경로
* Parent 로봇과의 거리
* 연결 여유 거리

---

### 9.2 반응형 Breadcrumb 배치

탐색 시작 전에 Relay 전체를 미리 배치하지 않는 방식

배치 과정

1. 군집 전방의 통신 상태 확인
2. 연결 비율과 Parent 거리 확인
3. 통신 단절 위험 감지
4. 군집 후방의 NORMAL 로봇 중 Relay 후보 탐색
5. 필요한 위치에 한 대씩 Breadcrumb 배치
6. Base까지의 다중 홉 통신 연결 갱신

정적인 NORMAL 경비 로봇을 남기지 않고 필요한 로봇만 `RELAY` 역할로 전환하는 구조

---

### 9.3 Relay 회수

Branch Backtracking 과정에서 Junction과 가까운 Relay부터 순차 회수

회수 조건

* 후속 통신 연결 유지 가능
* Branch 내부 군집 이동 방해 없음
* 일정 시간 동안 안전 상태 유지
* 다음 Relay와 Base 연결 확보

최종 Base 복귀 시 Trunk Relay를 Junction 측부터 순차적으로 해제하는 구조

---

## 10. Shepherd 시스템

### 10.1 적응형 Shepherd 수

통로 폭에 따라 필요한 Shepherd 수 자동 계산

제한 범위

```text
최소 5대
최대 14대
```

Shepherd 수 결정 요소

* 실제 통로 폭
* 벽과의 안전 여유
* 목표 Shepherd 슬롯 간격
* 전체 폭을 차단할 수 있는 최소 로봇 수

---

### 10.2 Shepherd 선출

Branch 선두 로봇이 막다른 길의 Capture Region에 도달한 이후 선출 시작

선출 과정

1. Capture Region 내부 NORMAL 로봇 탐색
2. 필요한 Shepherd 수 확인
3. 로컬 후보 간 슬롯 비용 계산
4. Shepherd 슬롯별 후보 배정
5. 막다른 벽 앞 횡방향 경계 형성
6. 일반 로봇이 Shepherd 뒤쪽에 쌓이도록 가상 Curtain 적용

Base의 중앙 명령이 아닌 로컬 로봇 정보 기반 역할 선출

---

### 10.3 포화 감지

Shepherd 경계 뒤에 충분한 로봇이 쌓인 시점 판단

포화 판단 요소

* 막다른 영역 내부 로봇 수
* 저속 로봇 비율
* 기준 대비 평균 밀도
* 격자 기반 공간 점유율
* 군집 전방의 진행 정체
* 포화 조건 유지시간
* Shepherd 수에 따른 최소 적재 로봇 수

단일 조건이 아니라 밀도·속도·공간 점유·정체를 결합한 다중 조건 포화 감지

---

### 10.4 압력 Backtracking

포화 감지 후 Shepherd가 피스톤 형태로 이동하며 군집에 역방향 압력 전달

처리 과정

1. Shepherd 뒤쪽 로봇 밀집
2. 포화 상태 확정
3. Shepherd 가상압력 증가
4. NORMAL 로봇의 역방향 흐름 발생
5. 이동 로봇 비율과 평균속도 확인
6. Backtracking 흐름 성립 판정
7. Shepherd 역할 해제 및 군집 합류
8. Branch Relay 순차 회수
9. 다음 Branch 또는 Base 방향으로 압력 전달

SPH 압력과 Shepherd 가상압력의 결합을 통한 군집 Backtracking

---

### 10.5 Pre-Shepherd 파이프라인

현재 Branch가 Backtracking 중일 때 다음 Branch의 Shepherd 후보를 미리 준비하는 구조

주요 기능

* 다음 Branch 내부 후보 사전 선출
* Shepherd 경계 사전 형성
* 경계 뒤 로봇 적재 여부 확인
* 이전 Branch 잔여 로봇 배출
* Branch 전환 직후 압력 Push 시작
* Junction 정지시간 감소

이전 Branch와 다음 Branch의 흐름을 연결하기 위한 파이프라인 방식

---

## 11. 전체 상태 머신

```text
MOVE_TO_JUNCTION
        ↓
EXPLORE_BRANCH
        ↓
FORM_SHEPHERD_BOUNDARY
        ↓
FILL_BEHIND_SHEPHERD
        ↓
PRESSURE_PUSH
        ↓
FLOW_BACKTRACK
        ↓
JUNCTION_SWITCH
        ↓
다음 Branch 존재
   ├─ 예 → EXPLORE_BRANCH
   └─ 아니요 → FINAL_JUNCTION_GATHER
                         ↓
                   RETURN_TO_BASE
                         ↓
                        DONE
```

### 상태별 기능

| 상태                       | 기능                                        |
| ------------------------ | ----------------------------------------- |
| `MOVE_TO_JUNCTION`       | Base 압축 해제, Junction 이동, 분산 투표, Anchor 선출 |
| `EXPLORE_BRANCH`         | 선택 Branch 진입, Relay 배치, Shepherd 후보 탐색    |
| `FORM_SHEPHERD_BOUNDARY` | 막다른 벽 앞 Shepherd 경계 형성                    |
| `FILL_BEHIND_SHEPHERD`   | Shepherd 뒤 군집 적재 및 포화 감지                  |
| `PRESSURE_PUSH`          | Shepherd 피스톤 압력 적용                        |
| `FLOW_BACKTRACK`         | 역방향 군집 흐름과 Relay 회수                       |
| `JUNCTION_SWITCH`        | 다음 미탐색 Branch 합의                          |
| `FINAL_JUNCTION_GATHER`  | Branch 잔여 로봇과 특수 역할 정리                    |
| `RETURN_TO_BASE`         | Anchor 해제 및 Trunk Relay 순차 회수             |
| `DONE`                   | 전체 탐색 및 복귀 완료                             |

---

## 12. 전체 알고리즘 흐름

```text
Base 내부 고밀도 로봇 배치
→ 초기 압축을 통한 SPH 압력 저장
→ 압축 해제 및 Junction 방향 팽창
→ 통신 간격 증가 시 Reactive Breadcrumb Relay 배치
→ Junction 내부 NORMAL 로봇 간 Branch 합의
→ 최소비용 Junction Anchor 선출
→ Anchor의 합의 결과 및 가상 게이트 상태 저장
→ 선택 Branch 개방
→ 비선택 Branch 가상 게이트 폐쇄
→ 선택 Branch 방향 SPH 군집 이동
→ 통신 위험 발생 시 추가 Breadcrumb Relay 배치
→ 선두 로봇의 막다른 Capture Region 도달
→ 통로 폭 기반 Shepherd 수 계산
→ 로컬 후보 경매 기반 Shepherd 선출
→ Shepherd 횡방향 경계 형성
→ Shepherd 뒤쪽 NORMAL 로봇 적재
→ 속도·밀도·점유율·정체 기반 포화 감지
→ Shepherd 피스톤 가상압력 시작
→ SPH 압력 기반 Backtracking 흐름 형성
→ Base–Backtracking 군집 사이 임시 통신 브리지 형성
→ Shepherd 역할 해제 및 군집 합류
→ Branch Breadcrumb Relay 순차 회수
→ 임시 통신 브리지 역할 해제
→ 다음 Branch 방향으로 압력 전달
→ PRE_SHEPHERD를 이용한 다음 Branch 선행 준비
→ RIGHT, UP, LEFT 순서 반복
→ 전체 Branch 방문 완료
→ Junction 최종 집결
→ Anchor 역할 해제
→ 잔여 Relay 순차 회수
→ 단방향 복귀 게이트 적용
→ 전체 로봇 Base 복귀
→ DONE 상태 전환
→ 실험 로그 저장
```

---

## 13. 주요 파라미터

### 군집 및 SPH

| 파라미터                      |              기본값 | 의미             |
| ------------------------- | ---------------: | -------------- |
| `ROBOT_COUNT`             |              680 | 전체 로봇 수        |
| `SMOOTHING_LENGTH`        | `22 × MAP_SCALE` | SPH 이웃 탐색 반경   |
| `PRESSURE_GAIN`           |           2800.0 | 압력 크기          |
| `STIFFNESS_EXPONENT`      |              0.5 | 압력 상태방정식 강성 지수 |
| `DAMPING`                 |              4.0 | 속도 감쇠          |
| `REPULSION_GAIN`          |            260.0 | 근거리 충돌 반발력     |
| `MOTION_SPEED_MULTIPLIER` |              3.0 | 전체 이동속도 배율     |

### Kelvin–Voigt

| 파라미터                                   |  기본값 | 의미          |
| -------------------------------------- | ---: | ----------- |
| `VISCOELASTIC_ELASTIC_GAIN`            | 42.0 | Spring 복원력  |
| `VISCOELASTIC_DASHPOT_GAIN`            |  8.0 | 상대속도 감쇠     |
| `VISCOELASTIC_EQUILIBRIUM_ADAPTATION`  |  4.0 | 평형거리 적응 속도  |
| `VISCOELASTIC_VELOCITY_CONSENSUS_GAIN` |  6.0 | 국소 속도 합의 강도 |

### 통신

| 파라미터                  |                         기본값 | 의미                 |
| --------------------- | --------------------------: | ------------------ |
| `COMM_RANGE`          |            `54 × MAP_SCALE` | 최대 통신 거리           |
| `COMM_SAFE_DISTANCE`  |            `40 × MAP_SCALE` | 안전 통신 거리           |
| `TRUNK_RELAY_SPACING` |            `30 × MAP_SCALE` | Trunk Relay 목표 간격  |
| `RELAY_SPACING`       |            `30 × MAP_SCALE` | Branch Relay 목표 간격 |
| `BREADCRUMB_SPACING`  | `COMM_SAFE_DISTANCE × 0.55` | Breadcrumb 목표 간격   |

### Shepherd 및 포화

| 파라미터                         |   기본값 | 의미               |
| ---------------------------- | ----: | ---------------- |
| `SHEPHERD_MIN_COUNT`         |     5 | 최소 Shepherd 수    |
| `SHEPHERD_MAX_COUNT`         |    14 | 최대 Shepherd 수    |
| `SATURATION_MIN_TIP_ROBOTS`  |    18 | 포화 판단 최소 선두 로봇 수 |
| `SATURATION_DENSITY_RATIO`   |  1.02 | 기본 밀도 포화 기준      |
| `SATURATION_OCCUPANCY_RATIO` |  0.16 | 기본 공간 점유율 기준     |
| `VIRTUAL_PRESSURE_FORCE`     | 110.0 | Shepherd 가상압력 크기 |

---

## 14. 실험 지표

시뮬레이션 완료 또는 사용자 종료 시 다음 CSV 파일 자동 생성

```text
sph_dfs_experiment_summary.csv
```

저장 항목

* 종료 사유
* 전체 로봇 수
* 전체 시뮬레이션 시간
* 전체 로봇 이동거리
* NORMAL 역할 이동거리
* RELAY 역할 이동거리
* TRUNK_RELAY 역할 이동거리
* SHEPHERD 역할 이동거리
* ANCHOR 역할 이동거리
* 통신 단절 Robot-Seconds
* 최소 로봇 간 거리
* 안전거리 위반 횟수
* 실제 Branch 방문 순서
* Branch 선택 이벤트 수
* 후보별 Branch 비용 성분
* 포화 감지 이벤트 수
* 압력 Push 이벤트 수

---

## 15. HUD 표시 정보

화면 오른쪽 HUD 패널의 실시간 정보

* 현재 FPS와 시뮬레이션 단계
* Base 초기 압축 및 압력 해제 상태
* NORMAL 로봇의 분산 합의 결과
* Anchor ID, 비용, 저장 상태 및 통신 상태
* Branch별 가상 게이트 상태
* Branch 방문 순서와 DFS 상태
* Proxy Region별 질량 비율
* 후보 Branch별 SPH Rollout 비용
* Base 연결 로봇 수와 최대 Hop 수
* Breadcrumb Relay 수와 전방 통신 비율
* Branch 내부 NORMAL, RELAY, SHEPHERD 수
* 포화 감지 지표
* Shepherd 형성 상태
* Kelvin–Voigt 활성 연결 수
* SPH, EDF, Shepherd 및 압력 해제 힘
* 전체 이동거리와 누적 통신 단절 시간

---

## 16. 실행 환경

### 요구사항

* Python 3.10 이상
* Pygame

### 패키지 설치

```bash
pip install pygame
```

### 실행

```bash
python single_junction_sph_dfs.py
```

---

## 17. 조작 방법

| 키       | 기능                        |
| ------- | ------------------------- |
| `SPACE` | 일시정지 및 재개                 |
| `R`     | 시뮬레이션 초기화                 |
| `D`     | 밀도 색상 표시 전환               |
| `V`     | Proxy Region과 분석 영역 표시 전환 |
| `C`     | 통신 연결선 표시 전환              |
| `ESC`   | 시뮬레이션 종료                  |

---

## 18. 시각화 색상

| 객체           | 색상     |
| ------------ | ------ |
| NORMAL 로봇    | 남색     |
| ANCHOR       | 밝은 초록색 |
| SHEPHERD     | 보라색    |
| Branch Relay | 갈색     |
| Trunk Relay  | 진한 갈색  |
| 통신 단절 로봇     | 붉은색    |
| RIGHT Branch | 주황색    |
| UP Branch    | 파란색    |
| LEFT Branch  | 보라색    |
| 가상 게이트       | 빨간색    |

---

## 19. 코드 구성

```text
1. Display
2. Cross Map
3. State and Branch Metadata
4. Physics and Control Parameters
5. Map Mask and Region Checks
6. General Utilities
7. Experiment Metrics
8. Base Station and Robot
9. Robot Creation and Spatial Hash
10. Base-Rooted Communication
11. Reactive Tail Breadcrumb Communication Trail
12. Anchor Election and Branch Analysis
12-1. Junction Stability Consensus
13. Saturation Detector
14. Adaptive Shepherd Election and Pressure Flow
15. SPH
16. State Machine
17. Initialization
18. Main Loop
```

---

## 20. 확장 방향

* 다중 Junction Topology 적용
* Junction별 독립 Anchor 선출
* 재귀적 DFS Tree 생성
* Parent–Child Junction 상태 전달
* 탐색 실패 Branch의 상태 복구
* 통신 단절 시 DFS Tree Repair
* 동적 장애물 환경 적용
* ROS 2 및 Gazebo 기반 실제 로봇 모델 연동
* 파라미터별 반복 실험 자동화
* 알고리즘별 성능 비교
* 실제 TurtleBot 통신 범위와 이동속도 반영

---

## 21. 제한사항

* 단일 Junction을 대상으로 한 연구 프로토타입
* 알려진 고정 지도 사용
* 2차원 Pygame 물리환경 사용
* 실제 로봇의 센서 노이즈와 구동기 오차 미반영
* 통신 지연과 패킷 손실 미반영
* 고정 Branch 우선순위 사용
* Proxy Rollout 결과를 실제 Branch 순서 최적화에 직접 사용하지 않는 구조
* 실제 유체를 재현하기 위한 CFD 모델이 아닌 SPH 기반 군집 제어 모델
* 실제 Kelvin–Voigt 물질을 재현하는 모델이 아닌 로봇 간 점탄성 제어 결합 구조

---
---
Last update : 2026.07.29
