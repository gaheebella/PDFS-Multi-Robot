# Communication-Maintained Physical DFS with Flow-Preserving SPH

## Proxy-Region-Based SPH-Aware Single-Junction Multi-Robot Exploration

본 프로젝트는 다중 로봇 군집을 유체 입자처럼 제어하는 **Smoothed Particle Hydrodynamics(SPH)**와 그래프 탐색 알고리즘인 **Depth-First Search(DFS)**를 결합한 Pygame 기반 연구 프로토타입이다.

고정된 Base에서 출발한 로봇 군집이 Junction까지 이동하고, Base와의 멀티홉 통신을 유지하면서 미탐색 Branch를 하나씩 탐색한다.

Junction에서는 단순히 고정된 순서나 Branch 길이만을 기준으로 다음 Branch를 선택하지 않는다. 현재 로봇 군집의 밀도, 속도, 형상, 진행 방향, Branch 입구 혼잡도, 통신 상태, 예상 Relay 수를 분석한다. 또한 각 후보 Branch를 선택했을 때의 짧은 가상 SPH 시뮬레이션을 수행해 군집의 현재 흐름을 가장 적게 방해하는 Branch를 선택한다.

Branch의 막다른 지점에서는 선두 로봇 일부를 Shepherd로 전환한다. Shepherd는 복도 폭을 가로지르는 경계를 형성하고, 일반 로봇이 그 뒤에 충분히 밀집하면 Junction 방향으로 이동하는 피스톤처럼 작동한다. 이를 통해 Dead-end에 도달한 전체 군집의 Backtracking 흐름을 생성한다.

---

# 1. 프로젝트 목적

일반적인 DFS는 다음과 같은 논리적 탐색 순서를 결정한다.

```text
Branch 진입
→ 공간 탐색
→ Dead-end 또는 탐색 완료
→ Parent Junction으로 Backtracking
→ 다음 미탐색 Branch 선택
```

하지만 실제 다중 로봇 시스템에서는 DFS 순서만으로 해결할 수 없는 문제가 존재한다.

* 탐색 전방과 Base 사이의 통신을 지속적으로 유지해야 한다.
* 좁은 복도와 Junction에서 로봇 간 충돌을 방지해야 한다.
* Branch 방향 전환 시 군집이 급격하게 압축되거나 흩어지는 현상을 줄여야 한다.
* Dead-end에서 일부 로봇만 복귀하고 나머지가 잔류하는 현상을 방지해야 한다.
* 필요 이상의 Relay를 배치해 발생하는 이동비용을 줄여야 한다.
* Backtracking 중 Relay 회수 순서로 인한 통신 단절을 방지해야 한다.
* 고정된 Branch 순서가 현재 군집의 실제 흐름과 맞지 않는 문제를 해결해야 한다.
* 탐색 완료 후 Anchor, Branch Relay, Trunk Relay를 모두 회수해야 한다.

본 프로젝트는 이러한 문제를 해결하기 위해 다음 구조를 결합한다.

```text
Physical DFS
+
Flow-Preserving SPH Branch Ordering
+
Base-Rooted LOS Communication
+
Adaptive Relay Deployment
+
Shepherd Pressure Backtracking
```

핵심 목표는 다음과 같다.

> DFS의 완전탐색 구조는 유지하면서, SPH 군집의 현재 물리 상태와 통신 상태를 이용해 불필요한 이동, 압축, 충돌, 정지시간, 통신 단절을 줄인다.

---

# 2. 현재 구현 범위

현재 코드는 하나의 Cross/T Junction 환경을 대상으로 한다.

```text
                  UP
                   │
                   │
LEFT ───────── JUNCTION ───────────────── RIGHT
                   │
                   │
                  BASE
```

현재 구현된 기능은 다음과 같다.

* 하나의 Junction에 연결된 `UP`, `LEFT`, `RIGHT` Branch를 관리한다.
* 각 Branch의 탐색 상태를 관리한다.
* Junction에서 다음 DFS Child Branch를 선택한다.
* 선택된 Branch만 활성화한다.
* Branch 끝까지 로봇 군집을 이동시킨다.
* 전방 통신 상태에 따라 Branch Relay를 배치한다.
* Dead-end 포화 상태를 감지한다.
* Shepherd를 이용해 Backtracking 흐름을 생성한다.
* Backtracking 중 Branch Relay를 역순으로 회수한다.
* Junction으로 복귀한 뒤 남은 Branch를 다시 평가한다.
* 모든 Branch 탐색 후 Anchor와 Trunk Relay를 회수한다.
* 전체 로봇이 Base로 복귀하면 탐색을 종료한다.
* 시뮬레이션 결과를 CSV 파일로 저장한다.

현재 구현되지 않은 기능은 다음과 같다.

* 여러 Junction을 재귀적으로 탐색하는 전체 DFS Tree
* DFS Stack 기반의 다중 깊이 Parent–Child 관리
* Junction별 독립 Anchor 관리
* 동적 장애물에 따른 DFS Tree Repair
* Loop가 존재하는 일반 Topological Graph 탐색
* 다층 건물 및 3차원 환경
* 실제 차동구동 로봇 동역학
* 센서 노이즈, 통신 지연, 패킷 손실
* 로봇 고장 및 Relay 실패

따라서 현재 버전은 다중 Junction Physical DFS로 확장하기 전에 하나의 Junction에서 필요한 핵심 제어와 의사결정 구조를 검증하는 **Single-Junction Baseline**이다.

---

# 3. 시뮬레이션 환경

## 3.1 지도 구조

현재 지도는 다음 네 개 영역으로 구성된다.

| 영역         | 설명                                |
| ---------- | --------------------------------- |
| `BOTTOM`   | Base와 초기 로봇 군집이 위치하는 입구 복도        |
| `JUNCTION` | Branch 선택과 Anchor 배치가 이루어지는 중앙 영역 |
| `UP`       | 위쪽 Branch                         |
| `LEFT`     | 왼쪽 Branch                         |
| `RIGHT`    | 오른쪽 Branch                        |

`RIGHT` Branch는 다른 Branch보다 길게 설정되어 있다.

```python
normal_length = 180
right_length = normal_length * 2
```

이를 통해 Branch 길이에 따라 다음 요소가 어떻게 달라지는지 확인할 수 있다.

* 전체 이동거리
* Backtracking 거리
* 필요한 Relay 수
* 통신 단절 위험
* Branch 탐색시간
* 군집 압축과 팽창 정도
* Branch 선택 비용

---

## 3.2 로봇 구성

기본 로봇 수는 220대다.

```python
ROBOT_COUNT = 220
SPAWN_MODE = "grid"
ROBOT_RADIUS = 2
GRID_SPACING = 7
```

로봇은 초기에는 모두 `NORMAL` 역할을 가진다.

탐색 과정에서 상황에 따라 다음 역할로 변경된다.

| 역할            | 설명                                          |
| ------------- | ------------------------------------------- |
| `NORMAL`      | 일반 탐색과 군집 이동을 수행하는 로봇                       |
| `ANCHOR`      | Junction에서 DFS 상태와 Branch 선택을 관리하는 로봇       |
| `TRUNK_RELAY` | Base와 Junction 사이의 통신 Backbone을 유지하는 로봇     |
| `RELAY`       | Junction과 활성 Branch 전방 사이의 통신을 유지하는 로봇      |
| `SHEPHERD`    | Dead-end에서 압력 경계를 만들고 Backtracking을 유도하는 로봇 |

로봇의 역할은 고정되지 않는다. 탐색 상황에 따라 Normal Robot이 Anchor, Relay, Shepherd로 변경되고, 임무가 끝나면 다시 Normal Robot으로 복귀한다.

---

# 4. 전체 시스템 흐름

전체 시스템 흐름은 다음과 같다.

```text
Base에서 로봇 군집 출발
        ↓
Base–Junction Trunk Relay 배치
        ↓
Junction Anchor 선발
        ↓
Junction 안정성 확인
        ↓
미탐색 Branch 후보 수집
        ↓
완전탐색 및 자원 가능성 검사
        ↓
Junction Proxy Region 분할
        ↓
후보 Branch별 가상 SPH Rollout
        ↓
Flow-Preserving Branch 선택
        ↓
선택 Branch 탐색
        ↓
필요 시 Branch Relay 배치
        ↓
Dead-end 도착
        ↓
Shepherd 선발 및 경계 형성
        ↓
Dead-end 포화 상태 감지
        ↓
Moving Piston Pressure Push
        ↓
Backtracking 흐름 생성
        ↓
Branch Relay 역순 회수
        ↓
Junction 복귀
        ↓
남은 Branch 재평가
        ↓
모든 Branch 탐색 완료
        ↓
Anchor와 Trunk Relay 회수
        ↓
전체 로봇 Base 복귀
```

---

# 5. SPH 기반 군집 제어

## 5.1 SPH 사용 목적

SPH는 유체를 여러 입자로 나누어 밀도, 압력, 점성 등의 상호작용을 계산하는 방식이다.

본 프로젝트에서는 각 로봇을 하나의 유체 입자로 간주한다. 로봇은 주변 로봇과의 거리와 상대속도를 이용해 밀도와 압력을 계산하고, 군집 전체가 하나의 유체처럼 이동하도록 제어된다.

SPH를 사용하는 이유는 다음과 같다.

* 고정 Formation보다 좁은 복도에 유연하게 적응할 수 있다.
* Branch 방향이 바뀌어도 군집 전체가 자연스럽게 변형될 수 있다.
* 로봇 간 간격을 일정하게 유지하는 데 도움이 된다.
* Dead-end 압축 상태를 밀도와 압력으로 표현할 수 있다.
* Shepherd가 만든 압력을 군집 내부로 전달할 수 있다.
* Backtracking 과정에서 국소적인 움직임이 전체 흐름으로 확산될 수 있다.

---

## 5.2 밀도

각 로봇은 일정 거리 안에 있는 이웃 로봇을 확인한다.

주변 로봇이 많고 가까울수록 밀도가 높아진다. 주변 로봇이 적거나 멀리 퍼져 있으면 밀도가 낮아진다.

밀도는 다음 상황을 판단하는 데 사용된다.

* 로봇 군집의 과밀 여부
* Branch 입구 혼잡 여부
* Dead-end 포화 여부
* Shepherd Pressure Push 시작 여부
* Junction 안정화 여부
* Branch별 Proxy Mass 계산
* 후보 Branch Rollout의 밀도 교란 평가

현재 SPH 이웃 범위는 다음과 같다.

```python
SMOOTHING_LENGTH = 28.0
```

---

## 5.3 압력

밀도가 기준보다 높아지면 로봇을 주변으로 밀어내는 압력이 발생한다.

압력은 다음 역할을 수행한다.

* 로봇 간 과도한 밀집을 완화한다.
* 좁은 복도에서 로봇 간 간격을 유지한다.
* Junction에서 군집이 자연스럽게 퍼지도록 한다.
* Branch 입구의 병목을 완화한다.
* Dead-end에서 군집의 압축 상태를 표현한다.
* Shepherd가 생성한 Backtracking 압력을 주변 로봇에 전달한다.
* Backtracking 후 군집이 다시 넓게 분포하도록 한다.

현재 기본 압력 크기는 다음과 같다.

```python
PRESSURE_GAIN = 1650.0
```

---

## 5.4 점성

점성은 서로 접근하는 로봇 사이의 상대속도 차이를 줄이는 역할을 한다.

점성이 없으면 압력에 의해 로봇이 튕기거나, 방향 전환 시 군집이 심하게 진동할 수 있다.

점성은 다음 현상을 완화한다.

* 로봇 사이의 급격한 상대속도 차이
* Junction 전환 시 진동
* 압력에 의한 튕김
* Shepherd Push 직후의 불안정
* 복도 내부의 반복적인 압축과 팽창
* 군집 내부 속도 불균형

현재 점성 관련 값은 다음과 같다.

```python
VISCOSITY_XI1 = 0.9
VISCOSITY_XI2 = 1.2
```

---

## 5.5 충돌 반발력

두 로봇 사이의 거리가 안전거리보다 작아지면 서로 반대 방향으로 밀어내는 반발력을 적용한다.

```python
SAFE_RADIUS = 7.5
REPULSION_GAIN = 260.0
```

이 힘은 SPH 압력과 별도로 작동한다.

SPH 압력은 군집 전체의 밀도 분포를 조절하고, 충돌 반발력은 매우 가까워진 두 로봇 사이의 직접 충돌을 방지한다.

---

## 5.6 감쇠

모든 이동 로봇에는 속도에 반대되는 감쇠력이 적용된다.

```python
DAMPING = 2.3
```

감쇠는 다음 역할을 수행한다.

* 로봇의 과도한 가속을 줄인다.
* Branch 입구에서 진동을 줄인다.
* Shepherd Pressure Push 후 군집이 지나치게 튀어나가는 현상을 줄인다.
* Junction에 복귀한 로봇이 빠르게 안정화되도록 한다.

---

## 5.7 고립 로봇 복구

주변 이웃 수가 기준보다 적은 로봇은 군집에서 분리될 위험이 있다고 판단한다.

```python
ISOLATION_NEIGHBOR_THRESHOLD = 4
ISOLATION_ROUTE_BOOST = 1.1
LOCAL_COHESION_GAIN = 20.0
```

고립된 로봇에는 다음 처리를 적용한다.

* 목표 방향으로 이동하는 Route Force를 증가시킨다.
* 주변 이웃의 중심 방향으로 Cohesion Force를 적용한다.
* Base와 연결된 군집으로 다시 합류하도록 유도한다.

Pressure Push 단계에서는 Backtracking 흐름을 방해하지 않도록 일부 Cohesion 처리를 제한한다.

---

# 6. Geodesic EDF 경로 유도

단순히 목표점 방향으로 직선 Attraction Force를 적용하면 로봇이 벽을 향해 이동할 수 있다.

현재 구현은 로봇이 위치한 영역에 따라 다음 중간 목표점을 선택한다.

```text
BOTTOM 또는 다른 Branch에 있는 경우
→ 현재 복도의 Junction 입구로 이동한다.

Junction에 있는 경우
→ 선택된 Branch 입구로 이동한다.

선택된 Branch 안에 있는 경우
→ Branch 내부 목표점으로 이동한다.
```

이를 통해 로봇이 벽을 통과하는 직선이 아니라 실제 자유공간을 따라 이동하도록 한다.

현재 Cross Map은 직사각형 복도와 Junction으로 구성되어 있으므로 별도의 Raster Distance Field를 생성하지 않고 기하학적인 Region 정보로 이동 방향을 계산한다.

---

# 7. 가상 Branch Valve

DFS에서는 한 번에 하나의 Branch만 탐색해야 한다.

선택되지 않은 Branch 입구에는 Junction 중심 방향으로 작용하는 가상 Valve Force를 생성한다.

```text
선택 Branch
→ 개방한다.

미선택 Branch
→ 가상 Valve로 폐쇄한다.
```

가상 Valve는 다음 문제를 방지한다.

* 로봇이 여러 Branch로 동시에 분산되는 현상
* DFS 순서가 무너지는 현상
* 미선택 Branch에 로봇이 잔류하는 현상
* Branch 전환 과정에서 잘못된 방향으로 로봇이 유출되는 현상

가상 Valve는 실제 벽이 아니라 부드러운 힘으로 구현된다. 따라서 로봇이 Branch 입구에 가까워질수록 Junction 안쪽으로 밀어내는 힘이 증가한다.

---

# 8. Junction Anchor

## 8.1 Anchor 역할

Junction Anchor는 다음 역할을 수행한다.

* Junction의 로컬 DFS 상태를 관리한다.
* 각 Branch의 상태를 저장한다.
* 다음 Branch를 선택한다.
* 현재 활성 Branch를 저장한다.
* Branch 완료 상태를 기록한다.
* 탐색 명령을 통신망에 전달한다.
* Branch 전환 시 Junction 기준점 역할을 수행한다.

Anchor는 통신망의 Root가 아니다.

통신 Root는 항상 고정 Base다. Anchor는 Base와 연결되어 있을 때만 정상적인 탐색 명령을 생성한다.

Anchor가 Base와 연결되어 있지 않으면 로봇 군집은 다음 명령을 받는다.

```text
WAIT_FOR_BASE_LINK
```

---

## 8.2 다중 기준 Anchor 선발

Junction에 가장 먼저 도착한 로봇을 단순히 Anchor로 지정하지 않는다.

Anchor 후보는 다음 기준으로 평가한다.

| 평가 항목         | 설명                                   |  가중치 |
| ------------- | ------------------------------------ | ---: |
| Arrival       | Junction에 얼마나 일찍 도착했는지 평가한다.         | 0.30 |
| Parking       | Anchor 주차 위치에 얼마나 가까운지 평가한다.         | 0.20 |
| Direction     | 현재 이동 방향이 Anchor 위치로 이동하기 적합한지 평가한다. | 0.15 |
| Communication | 주변 LOS 이웃 수와 통신 여유를 평가한다.            | 0.35 |

현재 설정은 다음과 같다.

```python
ANCHOR_ELECTION_MIN_CANDIDATES = 4
ANCHOR_ELECTION_WAIT_TIME = 0.22

ANCHOR_WEIGHT_ARRIVAL = 0.30
ANCHOR_WEIGHT_PARKING = 0.20
ANCHOR_WEIGHT_DIRECTION = 0.15
ANCHOR_WEIGHT_COMMUNICATION = 0.35
```

가장 높은 종합 점수를 받은 로봇을 Junction Anchor로 선발한다.

---

## 8.3 Anchor 이동 제약

Anchor로 선발된 로봇은 Junction 내부의 지정된 위치로 이동한다.

```python
ANCHOR_PARK_POSITION = pygame.Vector2(center_x - 25, center_y - 25)
```

Anchor는 다음 조건에서만 이동한다.

* Base와 연결되어 있어야 한다.
* 직접 연결된 주변 로봇이 존재해야 한다.
* 통신거리가 위험 수준을 넘지 않아야 한다.

통신 Margin이 감소하면 Anchor 이동속도를 줄인다. 통신거리가 위험 수준에 도달하면 Anchor 이동을 정지한다.

이를 통해 Anchor가 주차 위치로 이동하는 과정에서 Base 연결이 끊기는 문제를 방지한다.

---

# 9. Base-Rooted LOS 통신

## 9.1 통신 링크 조건

두 노드는 다음 조건을 모두 만족할 때 연결된다.

1. 두 노드 사이 거리가 최대 통신거리 이하이다.
2. 두 노드 사이에 Line-of-Sight가 존재한다.
3. 두 노드를 잇는 선분이 벽을 통과하지 않는다.

기본 통신 파라미터는 다음과 같다.

```python
COMM_RANGE = 46.0
COMM_SAFE_DISTANCE = 34.0
COMM_BARRIER_START = COMM_RANGE * 0.84
COMM_LOS_SAMPLE_SPACING = 6.0
```

LOS 검사는 두 로봇 사이를 일정 간격으로 Sampling하고, 각 Sample Point가 이동 가능 영역에 포함되는지 확인한다.

이를 통해 벽을 통과하는 비현실적인 통신 링크를 방지한다.

---

## 9.2 Base 기준 통신 경로

모든 로봇의 연결 여부는 고정 Base를 기준으로 계산한다.

통신 경로는 단순히 Hop 수가 가장 적은 경로를 사용하지 않는다.

각 경로에서 가장 위험한 링크의 통신 여유를 확인하고, 그 값이 가장 큰 경로를 선택한다.

즉 다음과 같은 경로를 우선한다.

```text
Hop 수는 조금 더 많지만 모든 링크가 안정적인 경로
```

다음과 같은 경로는 우선순위가 낮다.

```text
Hop 수는 적지만 한 링크가 최대 통신거리와 매우 가까운 경로
```

각 로봇은 다음 통신 정보를 저장한다.

* Base 연결 여부
* Base까지의 Hop 수
* 통신 Parent
* 경로 전체의 최소 통신 Margin
* 수신한 Branch 명령
* 수신한 Phase 명령
* 명령 Sequence 번호

---

## 9.3 연결 복구력

Base 연결이 끊긴 Normal Robot은 주변에서 Base와 연결된 가장 가까운 로봇을 탐색한다.

연결 가능한 로봇이 발견되면 해당 로봇 방향으로 이동하는 복구력을 적용한다.

```python
COMM_RECOVERY_RANGE = 84.0
COMM_RECOVERY_GAIN = 2.2
```

연결 복구력은 다음 상황에서 사용된다.

* 군집 가장자리 로봇이 통신망에서 이탈한 경우
* Branch 전환 중 일부 로봇이 Junction에 남은 경우
* Relay 배치 직전 전방 군집이 일시적으로 단절된 경우
* Backtracking 중 후방 로봇이 군집에서 분리된 경우

---

# 10. Base–Junction Trunk Relay

## 10.1 Trunk Relay 목적

Base와 Junction Anchor 사이에는 전체 탐색 과정에서 유지되는 통신 Backbone이 필요하다.

```text
Base
  │
Trunk Relay 1
  │
Trunk Relay 2
  │
Junction Anchor
```

Trunk Relay는 Branch 내부 Relay와 구분된다.

| 역할            | 연결 구간                     |
| ------------- | ------------------------- |
| `TRUNK_RELAY` | Base와 Junction 사이         |
| `RELAY`       | Junction과 활성 Branch 전방 사이 |

---

## 10.2 Trunk Relay 배치

Base와 Anchor 주차 위치 사이에 일정 간격으로 Relay Slot을 생성한다.

```python
TRUNK_RELAY_SPACING = 30.0
TRUNK_RELAY_SELECTION_RADIUS = 50.0
TRUNK_RELAY_DEPLOY_LOOKAHEAD = 12.0
```

전방 로봇이 다음 Relay Slot에 가까워지면 주변 Normal Robot 중 적합한 후보를 선택한다.

후보 조건은 다음과 같다.

1. Base와 연결되어 있어야 한다.
2. `BOTTOM` 또는 `JUNCTION` 영역에 있어야 한다.
3. Relay Slot과 일정 거리 이내에 있어야 한다.
4. Relay Slot까지의 거리가 가까운 로봇을 우선한다.
5. 이동속도가 낮은 로봇을 우선한다.
6. 조건이 같으면 Robot ID를 기준으로 선택한다.

---

## 10.3 Trunk Relay 순차 회수

모든 Branch 탐색이 끝나면 Trunk Relay를 Junction에 가까운 순서부터 Base 방향으로 회수한다.

```text
Junction 쪽 Trunk Relay 회수
→ 전체 연결 비율 확인
→ 짧은 안정시간 유지
→ 다음 Trunk Relay 회수
→ 마지막 Relay까지 반복
```

현재 주요 파라미터는 다음과 같다.

```python
RETURN_TRUNK_RETRACT_DWELL = 0.55
RETURN_TRUNK_READY_CONNECTED_RATIO = 0.97
RETURN_TRUNK_FORCE_RELEASE_TIMEOUT = 2.50
```

전체 연결 비율이 기준 이상이면 일정 시간 후 다음 Relay를 회수한다.

특정 로봇이 Region 경계에 걸려 회수가 영구적으로 멈추는 문제를 방지하기 위해 강제 회수 Timeout도 사용한다.

순차 회수는 다음 문제를 해결한다.

* Trunk Relay가 마지막까지 남는 문제
* Base 복귀 상태가 종료되지 않는 문제
* 모든 Relay를 동시에 해제해 통신이 끊기는 문제
* 경계선에 걸린 로봇 때문에 회수가 멈추는 문제

---

# 11. Branch Relay

## 11.1 Branch Relay 목적

Branch가 길어질수록 Junction Anchor와 전방 로봇 사이의 직접 연결이 불가능해진다.

Branch Relay는 다음 통신 체인을 형성한다.

```text
Base
  │
Trunk Relay
  │
Junction Anchor
  │
Branch Relay 1
  │
Branch Relay 2
  │
전방 로봇 군집
```

---

## 11.2 Relay Slot 생성

선택된 Branch가 결정되면 Junction Anchor에서 Branch 끝 방향으로 Relay Slot을 생성한다.

```python
RELAY_SPACING = 30.0
RELAY_END_CLEARANCE = 24.0
RELAY_LANE_MARGIN = 22.0
```

Relay Slot은 현재 활성 Branch에 대해서만 생성한다.

Branch가 완료되면 이전 Branch의 Relay Plan을 제거하고, 다음 Branch에 맞는 새로운 Relay Plan을 생성한다.

---

## 11.3 전방 통신 상태 평가

Branch 내부에서 가장 깊이 이동한 일부 로봇을 전방 집합으로 정의한다.

전방 집합에 대해 다음 값을 계산한다.

* Base와 연결된 로봇 비율
* 통신 경로의 Robust Margin
* Relay가 추가로 필요한지 여부

현재 기준은 다음과 같다.

```python
RELAY_FRONT_FRACTION = 0.20
RELAY_FRONT_MIN_COUNT = 10
RELAY_FRONT_REQUIRED_CONNECTED_RATIO = 0.90
RELAY_DEPLOY_MARGIN = 5.0
```

다음 중 하나를 만족하면 Relay가 필요하다고 판단한다.

```text
전방 연결 비율이 기준보다 낮다.
또는
전방 통신 Margin이 기준보다 낮다.
```

---

## 11.4 Relay 배치 중 이동 제한

통신 위험이 감지되면 군집의 전방 이동속도를 감소시킨다.

```python
RELAY_FORMING_SPEED_SCALE = 0.40
RELAY_WAIT_SPEED_SCALE = 0.18
```

동작 순서는 다음과 같다.

```text
전방 통신 위험 감지
→ 군집 이동속도 감소
→ 다음 Relay Slot 확인
→ Relay 후보 선발
→ Relay가 Slot으로 이동
→ Relay 안정화 확인
→ 군집 이동속도 복원
```

전방 연결 비율이 매우 낮으면 탐색 군집의 이동을 완전히 정지한다.

이를 통해 로봇 군집이 Relay 배치보다 빠르게 전진해 통신이 끊기는 현상을 방지한다.

---

## 11.5 Branch Relay 회수

Backtracking 단계에서는 Branch 끝에 가장 가까운 Relay부터 역순으로 회수한다.

Relay 회수 전에는 다음 조건을 확인한다.

* Relay보다 Branch 끝 방향에 남아 있는 이동 로봇이 없어야 한다.
* Branch와 Junction의 이동 로봇이 Base와 연결되어 있어야 한다.
* 조건이 일정 시간 유지되어야 한다.
* 이전 Relay 회수 후 Cooldown이 끝나야 한다.

조건을 만족하면 Relay를 `NORMAL` 역할로 전환하고 Junction 방향 초기속도를 부여한다.

```python
RELAY_RETRACT_DWELL_TIME = 0.40
RELAY_RETRACT_COOLDOWN = 0.45
RELAY_RELEASE_SPEED = 16.0 * MOTION_SPEED_MULTIPLIER
```

---

# 12. Flow-Preserving DFS Branch Ordering

## 12.1 Branch 선택 원칙

다음 Branch는 고정 순서나 길이만으로 선택하지 않는다.

현재 군집의 물리 상태와 통신 상태를 이용해 가장 자연스러운 Branch를 선택한다.

평가 요소는 다음과 같다.

* 현재 군집의 밀도 분포
* 현재 군집의 속도 방향
* 군집의 전체 형상
* Branch 입구 혼잡도
* 후보 Branch 방향 예상 유량
* 예상 밀도 교란
* 예상 속도 교란
* 벽 충돌 위험
* 로봇 간 충돌 위험
* 예상 통신 단절 위험
* 필요한 Relay 수
* Branch 길이
* Backtracking 거리
* 이전 이동 방향과의 회전각
* Branch 전환 시 필요한 SPH 강성 변화

전체 Branch 선택 과정은 다음과 같다.

```text
미탐색 Branch 수집
→ 자원 가능성 검사
→ 완전탐색 우선순위 검사
→ Junction Proxy Region 분할
→ 로봇 질량을 Proxy Region에 임시 할당
→ 후보별 SPH Short Rollout
→ 후보별 비용 계산
→ 최소 비용 Branch 선택
```

---

## 12.2 Branch 상태

각 Branch는 다음 상태 중 하나를 가진다.

```text
UNVISITED
ACTIVE
VISITED
```

| 상태          | 설명                |
| ----------- | ----------------- |
| `UNVISITED` | 아직 탐색하지 않은 Branch |
| `ACTIVE`    | 현재 탐색 중인 Branch   |
| `VISITED`   | 탐색을 완료한 Branch    |

다중 Junction과 동적 장애물 환경으로 확장할 경우 다음 상태를 추가할 수 있다.

```text
TEMP_WAIT
BLOCKED
UNREACHABLE
```

---

## 12.3 완전탐색 우선순위

Branch 선택에서 가장 먼저 확인하는 것은 완전탐색 가능성이다.

각 후보 Branch를 사용할 수 없다고 가정한 뒤 Base에서 도달 가능한 Target을 계산한다.

해당 Branch가 막혔을 때 더 많은 미탐색 Target이 도달 불가능해진다면 그 Branch는 우선적으로 탐색할 필요가 있다고 판단한다.

현재 단일 Junction Map에서는 각 Branch가 하나의 Target과 연결되어 있어 차이가 크지 않다.

하지만 코드 구조는 향후 다중 Junction Topological Graph로 확장할 수 있도록 다음 정보를 분리해 관리한다.

* Graph 인접관계
* Branch와 Target Node의 연결
* Branch별 방문 상태
* Base 기준 도달 가능 Node

완전탐색 우선순위는 일반 비용과 단순히 합산하지 않는다.

먼저 완전탐색 관점에서 우선 후보군을 결정하고, 같은 우선순위를 가진 후보 사이에서 SPH 효율 비용을 비교한다.

---

## 12.4 Branch 자원 가능성 검사

Branch를 탐색하려면 Branch 길이에 따른 Relay와 복도 폭에 따른 Shepherd가 필요하다.

필요한 역할 수는 다음 요소로 추정한다.

* Branch 내부에 배치해야 하는 예상 Relay 수
* 폭 적응형 Shepherd 수

현재 Base와 연결된 Normal Robot 수가 필요한 역할 수보다 많은 Branch를 우선 후보로 사용한다.

모든 Branch가 자원 조건을 통과하지 못하면 탐색을 완전히 중단하지 않고, 미탐색 Branch 전체를 Fallback 후보로 사용한다.

---

# 13. Junction Proxy Region

## 13.1 Proxy Region 개념

Junction 전체를 하나의 분석 영역으로 보고, 현재 미탐색 Branch 수에 따라 임시 하위 영역으로 나눈다.

Proxy Region은 실제 로봇을 여러 Branch로 동시에 보내기 위한 공간이 아니다.

> Proxy Region은 각 후보 Branch가 현재 군집 상태에서 얼마나 자연스럽게 활성화될 수 있는지 평가하기 위한 가상 의사결정 영역이다.

---

## 13.2 Branch별 요구량

각 Branch가 필요로 하는 로봇 자원은 다음 요소로 추정한다.

* 폭에 따라 필요한 Shepherd 수
* Branch 길이에 따라 필요한 Relay 수
* Branch 길이에 따른 전방 유체층 수
* 복도 폭에 따라 필요한 횡방향 로봇 수

길이가 긴 Branch는 더 많은 Relay와 전방 유체 질량이 필요하므로 더 큰 Proxy 영역을 할당받을 수 있다.

---

## 13.3 Capacity-Constrained Partition

Junction을 일정 크기의 Grid Cell로 나눈다.

```python
PROXY_CELL_SIZE = 10
```

각 Cell은 다음 요소를 이용해 Branch에 할당한다.

* Cell과 Branch 입구 사이의 거리
* Branch가 필요로 하는 목표 면적
* 현재 할당 면적과 목표 면적의 차이

각 Branch가 요구량에 비례하는 면적을 갖도록 Branch별 Bias를 반복적으로 조정한다.

```python
PROXY_PARTITION_ITERATIONS = 160
PROXY_BIAS_LEARNING_RATE = 0.075
```

이 방식은 로봇을 무작위로 나누지 않고, Branch 입구를 기준으로 공간적으로 연속된 Proxy Subregion을 생성한다.

---

## 13.4 로봇 위치 투영

실제 로봇의 위치를 Junction Proxy Region으로 투영한다.

| 실제 로봇 위치   | Proxy 투영 위치       |
| ---------- | ----------------- |
| `JUNCTION` | 현재 Junction 내부 위치 |
| `UP`       | Junction 위쪽 경계    |
| `LEFT`     | Junction 왼쪽 경계    |
| `RIGHT`    | Junction 오른쪽 경계   |
| `BOTTOM`   | Junction 아래쪽 경계   |

투영된 위치가 포함된 Cell의 Branch를 해당 로봇의 임시 Proxy Assignment로 사용한다.

이 Assignment는 Branch 선택 계산에만 사용한다. 실제 로봇의 역할이나 이동 방향은 변경하지 않는다.

---

## 13.5 Proxy Mass

단순히 로봇 수만 세지 않고 각 로봇의 현재 밀도도 반영한다.

밀도가 높은 로봇은 더 큰 유체 질량을 가진 것으로 간주한다.

다만 매우 낮거나 높은 밀도가 결과를 과도하게 지배하지 않도록 값을 제한한다.

```python
PROXY_DENSITY_MASS_MIN = 0.50
PROXY_DENSITY_MASS_MAX = 2.00
```

각 Branch에 대해 다음 값을 계산한다.

* 요구 Proxy 면적 비율
* 실제 할당 Cell 수
* 할당된 로봇 수
* 밀도 기반 실제 Proxy Mass
* 요구량 대비 부족 정도

요구량에 비해 실제 Proxy Mass가 부족한 Branch는 Branch 선택 비용이 증가한다.

---

# 14. 후보 Branch별 SPH Short Rollout

## 14.1 Rollout 목적

Branch를 실제로 선택하기 전에 짧은 가상 SPH 시뮬레이션을 수행한다.

이를 통해 각 Branch를 선택했을 때 발생할 수 있는 다음 변화를 예측한다.

* Branch 방향 예상 흐름
* 밀도 변화
* 속도 변화
* 벽 충돌 위험
* 로봇 간 충돌 위험
* Base 통신 위험
* Branch 진입 비율
* 안정화 비용

가상 Rollout은 실제 로봇의 위치, 속도, 역할, DFS 상태를 변경하지 않는다.

---

## 14.2 Rollout 시간

현재 Rollout 설정은 다음과 같다.

```python
FLOW_ROLLOUT_HORIZON = 0.50
FLOW_ROLLOUT_DT = 0.05
FLOW_ROLLOUT_MAX_ROBOTS = 190
FLOW_ROLLOUT_TARGET_DEPTH = 54.0
```

각 후보 Branch에 대해 약 0.5초의 미래 상태를 예측한다.

Rollout은 매 Frame 수행하지 않고 Junction에서 Branch를 선택하는 시점에만 수행한다.

---

## 14.3 Primary Particle

Primary Particle은 해당 Branch의 Proxy Region에 임시 할당된 로봇이다.

Primary Particle에는 다음 제어가 적용된다.

* 후보 Branch 방향 EDF Route Force
* 미선택 Branch의 Virtual Valve Force
* SPH Pressure
* Artificial Viscosity
* Collision Repulsion
* Centering Force
* Damping

Primary Particle만 Branch 성능 평가 지표에 직접 포함된다.

---

## 14.4 Context Particle

Context Particle은 후보 Proxy Region의 경계에서 SPH Support Length 안에 있는 주변 로봇이다.

Context Particle은 다음 목적으로 포함한다.

* 후보 영역 경계의 SPH 상호작용을 유지한다.
* 인접 Proxy Region의 유체 영향을 반영한다.
* 후보 Branch가 전체 군집을 잘못 끌어당기는 현상을 방지한다.
* 후보 영역과 주변 영역 사이의 압력과 점성 영향을 유지한다.

Context Particle은 원래 위치 주변에 머물도록 약한 Hold Force를 받는다.

```python
PROXY_CONTEXT_HOLD_GAIN = 3.2
PROXY_CONTEXT_MAX_SPEED_SCALE = 0.28
```

---

## 14.5 최소 Primary Particle 수

특정 Branch의 Proxy Region에 할당된 로봇이 너무 적으면 의미 있는 Rollout이 어려울 수 있다.

따라서 최소 Primary Particle 수를 보장한다.

```python
PROXY_ROLLOUT_MIN_PRIMARY = 6
```

할당된 로봇이 부족하면 해당 Branch 입구까지의 자유공간 거리가 가까운 로봇을 추가한다.

---

# 15. Branch 평가 항목

각 후보 Branch에 대해 다음 항목을 계산한다.

| 항목                        | 설명                                 |
| ------------------------- | ---------------------------------- |
| `predicted_flow`          | Branch 방향으로 예상되는 자연스러운 유량          |
| `density_disturbance`     | Rollout 전후 밀도 변화                   |
| `velocity_disturbance`    | Rollout 전후 속도 변화                   |
| `wall_risk`               | 벽 충돌과 벽 근접 위험                      |
| `collision_risk`          | 로봇 간 안전거리 침범 위험                    |
| `rollout_comm`            | 예상 Base 통신 단절 위험                   |
| `rollout_connected_ratio` | Rollout 후 예상 Base 연결 비율            |
| `rollout_margin`          | 예상 통신 Robust Margin                |
| `stabilization`           | 밀도와 속도 교란의 종합값                     |
| `lambda_mode`             | 방향 전환에 필요한 SPH 물성 변화량              |
| `predicted_entry_ratio`   | 후보 Branch에 진입한 Primary Particle 비율 |
| `transport`               | 현재 로봇 위치에서 Branch 입구까지의 이동비용       |
| `proxy_mass`              | Branch 요구량 대비 Proxy Mass 부족 정도     |
| `shape`                   | 현재 군집 형상과 Branch 방향의 정렬 정도         |
| `flow_prior`              | 현재 평균속도와 Branch 방향의 정렬 정도          |
| `congestion`              | Branch 입구의 밀도 혼잡도                  |
| `relay`                   | 예상 Relay 요구량                       |
| `backtrack`               | Branch 길이에 따른 복귀비용                 |
| `switch`                  | 이전 진행 방향과의 회전비용                    |

---

## 15.1 Predicted Flow

Branch 입구 주변의 Primary Particle이 해당 Branch 방향으로 얼마나 자연스럽게 이동하는지 측정한다.

Branch 방향 속도가 빠르고 입구에 가까울수록 높은 점수를 가진다.

Predicted Flow가 높다는 것은 다음을 의미한다.

* 현재 군집의 흐름이 후보 Branch 방향과 잘 맞는다.
* 방향 전환을 위해 큰 외력이 필요하지 않다.
* 군집의 속도와 밀도 교란이 작을 가능성이 높다.
* Branch 입구에 자연스러운 유량이 형성될 가능성이 높다.

Predicted Flow는 비용이 아니라 보상으로 사용한다.

---

## 15.2 Density Disturbance

Rollout 전후 각 Primary Particle의 밀도 변화를 측정한다.

밀도 교란이 크다는 것은 다음을 의미한다.

* 후보 Branch 진입 시 군집이 크게 압축된다.
* Branch 입구에서 병목이 발생할 가능성이 높다.
* 방향 전환 후 재팽창이 많이 필요하다.
* 로봇 간 충돌 위험이 증가할 수 있다.

---

## 15.3 Velocity Disturbance

Rollout 전후 각 Primary Particle의 속도 변화량을 측정한다.

속도 교란이 크다는 것은 다음을 의미한다.

* 현재 흐름을 크게 꺾어야 한다.
* 급격한 가속 또는 감속이 필요하다.
* Junction에서 로봇이 서로 충돌할 가능성이 높다.
* Branch 전환 후 안정화 시간이 증가할 수 있다.

---

## 15.4 Wall Risk

Rollout 중 벽과 충돌하거나 벽 가까이 이동한 횟수를 측정한다.

Wall Risk가 높은 Branch는 다음 문제를 가질 수 있다.

* Branch 입구 회전각이 크다.
* 현재 군집 형상과 Branch 방향이 맞지 않는다.
* 좁은 입구로 많은 로봇이 동시에 진입한다.
* 속도와 압력이 벽 방향으로 집중된다.

---

## 15.5 Collision Risk

Rollout 중 로봇 간 거리가 안전거리보다 작아진 정도를 측정한다.

충돌 위험이 높은 Branch는 다음 상황일 수 있다.

* Branch 입구에 로봇이 과도하게 집중된다.
* 방향 전환으로 로봇 궤적이 교차한다.
* 밀도와 속도 변화가 동시에 크게 발생한다.

---

## 15.6 Communication Risk

Rollout Particle과 다음 고정 노드를 이용해 예상 통신 그래프를 구성한다.

* Base
* Trunk Relay
* Junction Anchor
* Primary Particle
* Context Particle

Context Particle은 통신 중계 노드로 사용할 수 있지만, 최종 연결 비율은 Primary Particle만 평가한다.

통신 위험은 다음 요소를 결합한다.

* Base와 연결되지 못한 Primary Particle 비율
* Base까지의 경로에서 가장 위험한 링크의 통신 Margin

---

## 15.7 Transport Cost

Proxy Region에 할당된 로봇들이 후보 Branch 입구까지 이동해야 하는 평균 거리를 계산한다.

단순 직선거리가 아니라 다음 경로를 따른다.

```text
현재 복도
→ 현재 복도의 Junction 입구
→ Junction 중심
→ 후보 Branch 입구
```

이를 통해 벽을 가로지르는 비현실적인 최단거리를 사용하지 않는다.

---

## 15.8 Shape Cost

Proxy Region에 포함된 로봇들의 위치 분포를 이용해 군집이 길게 늘어진 방향을 계산한다.

군집의 긴 축이 후보 Branch 방향과 잘 정렬되어 있으면 Shape Cost가 낮아진다.

예를 들어 군집이 세로로 길게 형성되어 있다면 `UP` Branch로 이동할 때 적은 변형이 필요하다.

반대로 세로로 긴 군집이 `LEFT` 또는 `RIGHT`로 이동하려면 큰 방향 전환과 형상 변화가 필요하다.

---

## 15.9 Flow Prior

현재 로봇 군집의 평균속도가 후보 Branch 방향과 얼마나 일치하는지 측정한다.

현재 군집이 이미 오른쪽으로 이동하고 있다면 `RIGHT` Branch의 Flow Prior Cost가 낮아질 수 있다.

다만 평균속도가 매우 낮으면 현재 흐름 방향을 신뢰하기 어렵기 때문에 Flow Prior의 영향도 감소한다.

---

## 15.10 Congestion Cost

Branch 입구 주변의 로봇 밀도를 측정한다.

입구 주변 밀도가 기준보다 높으면 Congestion Cost가 증가한다.

이를 통해 이미 혼잡한 Branch 입구를 즉시 선택해 더 큰 압축을 발생시키는 것을 방지한다.

---

## 15.11 Relay Cost

Branch 길이에 따라 필요한 예상 Relay 수를 계산한다.

긴 Branch는 더 많은 Relay를 필요로 하므로 Relay Cost가 증가한다.

다만 Relay Cost만으로 Branch를 선택하지 않는다. 자연스러운 유량, 통신 위험, 밀도 교란 등과 함께 평가한다.

---

## 15.12 Backtracking Cost

Branch 길이가 길수록 탐색 후 Junction으로 복귀해야 하는 거리가 증가한다.

따라서 긴 Branch는 Backtracking Cost가 높아진다.

---

## 15.13 Switch Cost

이전 진행 방향과 후보 Branch 방향 사이의 회전각을 평가한다.

회전각이 클수록 다음 문제가 발생할 가능성이 높다.

* 군집의 속도 방향을 크게 바꿔야 한다.
* Junction에서 압축이 발생할 수 있다.
* 벽 충돌 위험이 증가할 수 있다.
* SPH 강성을 더 부드럽게 변경해야 할 수 있다.

---

# 16. Branch 최종 선택

각 Branch의 총비용은 다음 요소를 가중합해 계산한다.

* 예상 유량은 보상으로 적용한다.
* 밀도 교란은 비용으로 적용한다.
* 속도 교란은 비용으로 적용한다.
* 벽 위험은 비용으로 적용한다.
* 충돌 위험은 비용으로 적용한다.
* 통신 위험은 비용으로 적용한다.
* Relay 요구량은 비용으로 적용한다.
* SPH 강성 변경량은 비용으로 적용한다.
* 안정화 비용은 비용으로 적용한다.
* 이동거리 비용은 비용으로 적용한다.
* Proxy Mass 부족은 비용으로 적용한다.
* 군집 형상 불일치는 비용으로 적용한다.
* 현재 흐름 방향 불일치는 비용으로 적용한다.
* 입구 혼잡도는 비용으로 적용한다.
* Backtracking 거리는 비용으로 적용한다.
* 방향 전환각은 비용으로 적용한다.

현재 주요 가중치는 다음과 같다.

| 항목                    |  가중치 |
| --------------------- | ---: |
| Predicted Flow Reward | 0.24 |
| Density Disturbance   | 0.11 |
| Velocity Disturbance  | 0.10 |
| Rollout Communication | 0.09 |
| Transport             | 0.08 |
| Proxy Mass            | 0.12 |
| Congestion            | 0.08 |
| Wall Risk             | 0.07 |
| Collision Risk        | 0.07 |
| Relay                 | 0.07 |
| Shape                 | 0.05 |
| Flow Prior            | 0.06 |
| Lambda Mode           | 0.04 |
| Stabilization         | 0.04 |
| Backtracking          | 0.04 |
| Switch                | 0.04 |

완전탐색 우선순위가 같은 후보 중 최종 비용이 가장 작은 Branch를 선택한다.

---

# 17. Adaptive SPH Stiffness

## 17.1 목적

SPH 압력의 강성을 모든 상황에서 고정하면 큰 방향 전환 시 군집이 지나치게 단단하게 움직일 수 있다.

반대로 항상 부드럽게 설정하면 Dead-end Pressure Push에서 압력 전달이 약해질 수 있다.

따라서 현재 상태와 방향 전환량에 따라 SPH 강성을 변경한다.

---

## 17.2 상태별 강성

현재 주요 값은 다음과 같다.

```python
STIFFNESS_EXPONENT_RIGID = 0.50
STIFFNESS_EXPONENT_SOFT = 0.22
STIFFNESS_EXPONENT_PRESSURE_PUSH = 0.62
BRANCH_STIFFNESS_RECOVERY_TIME = 1.20
```

상태별 동작은 다음과 같다.

| 상태              | 강성 설정           | 목적                        |
| --------------- | --------------- | ------------------------- |
| 일반 이동           | 기본 강성           | 안정적인 군집 간격을 유지한다.         |
| 큰 Branch 방향 전환  | 낮은 강성           | 군집이 부드럽게 변형되도록 한다.        |
| Branch 진입 직후    | 낮은 값에서 기본값으로 복원 | 급격한 물성 변화를 방지한다.          |
| Pressure Push   | 높은 강성           | Shepherd 압력을 군집 내부로 전달한다. |
| Junction Switch | 낮은 강성           | Branch 전환을 유연하게 만든다.      |

Branch 방향 전환각이 클수록 초기 강성을 더 낮게 설정한다.

Branch에 진입한 뒤 일정 시간 동안 강성을 기본값으로 점진적으로 복원한다.

---

# 18. Junction 안정성 확인

## 18.1 필요성

Branch를 완료하고 Junction으로 복귀한 직후에는 로봇 군집이 계속 움직이고 있을 수 있다.

이 상태에서 즉시 다음 Branch를 선택하면 다음 문제가 발생할 수 있다.

* 이전 Branch 방향의 속도가 Branch 평가에 과도하게 반영된다.
* Junction에 충분한 로봇이 도착하지 않은 상태에서 선택한다.
* Proxy Region의 로봇 질량 분포가 불안정하다.
* Branch 전환 중 로봇 충돌이 증가한다.

반대로 모든 로봇이 완전히 정지할 때까지 기다리면 Junction 대기시간이 지나치게 길어진다.

---

## 18.2 Stable Path

다음 조건을 일정 시간 만족하면 Junction이 안정되었다고 판단한다.

* 최소 수 이상의 Normal Robot이 Junction에 존재한다.
* 일정 비율 이상의 로봇이 Base와 연결되어 있다.
* 일정 비율 이상의 로봇 속도가 기준보다 낮다.
* 평균속도가 기준보다 낮다.
* 평균 밀도 변화량이 기준보다 낮다.
* 조건이 일정 Dwell Time 동안 유지된다.

현재 주요 값은 다음과 같다.

```python
JUNCTION_CONSENSUS_MIN_COUNT = 14
JUNCTION_CONSENSUS_STABLE_RATIO = 0.62
JUNCTION_CONSENSUS_SPEED_THRESHOLD = 5.5
JUNCTION_CONSENSUS_DENSITY_DELTA_RATIO = 0.14
JUNCTION_CONSENSUS_DWELL_TIME = 0.18
```

---

## 18.3 Fast Path

Junction에 충분한 로봇이 이미 도착한 경우 조금 완화된 조건으로 더 빠르게 Branch 선택을 시작한다.

```python
JUNCTION_FAST_READY_MIN_COUNT = 18
JUNCTION_FAST_READY_STABLE_RATIO = 0.50
JUNCTION_FAST_READY_SPEED_THRESHOLD = 8.0
JUNCTION_FAST_READY_DENSITY_DELTA_RATIO = 0.22
JUNCTION_FAST_READY_DWELL_TIME = 0.10
```

Fast Path는 모든 로봇이 거의 정지할 때까지 기다리지 않고, Branch 선택에 필요한 최소한의 안정성만 확보한다.

---

## 18.4 Fallback Path

Stable Path와 Fast Path를 만족하지 못하더라도 일정 시간이 지난 뒤 최소한의 안정 조건을 만족하면 Branch 평가를 진행한다.

```python
JUNCTION_CONSENSUS_FALLBACK_TIME = 0.85
JUNCTION_FALLBACK_MIN_COUNT = 12
JUNCTION_FALLBACK_STABLE_RATIO = 0.35
JUNCTION_FALLBACK_SPEED_THRESHOLD = 10.0
```

이를 통해 Junction에서 무기한 대기하는 현상을 방지한다.

---

# 19. Dead-End Saturation Detection

## 19.1 목적

로봇이 Dead-end 근처에 도착했다고 바로 Backtracking을 시작하지 않는다.

일반 로봇이 Shepherd 경계 뒤쪽에 충분히 밀집해 압력을 전달할 수 있는 상태인지 확인해야 한다.

---

## 19.2 포화 평가 요소

Dead-end 포화 판정에는 다음 요소를 사용한다.

| 항목              | 설명                              |
| --------------- | ------------------------------- |
| Tip Count       | Dead-end 근처에 위치한 Normal Robot 수 |
| Low-Speed Ratio | 낮은 속도로 정체된 로봇 비율                |
| Density Ratio   | 기준 밀도 대비 평균 밀도                  |
| Occupancy Ratio | Dead-end Grid Cell 점유 비율        |
| Front Delta     | 가장 앞선 로봇의 진행량 변화                |
| Dwell Time      | 모든 조건이 유지된 시간                   |

현재 주요 값은 다음과 같다.

```python
SATURATION_MIN_TIP_ROBOTS = 18
SATURATION_LOW_SPEED_THRESHOLD = 4.0
SATURATION_LOW_SPEED_RATIO = 0.65
SATURATION_DENSITY_RATIO = 1.02
SATURATION_OCCUPANCY_RATIO = 0.16
SATURATION_FRONT_PROGRESS_EPSILON = 2.2
SATURATION_DWELL_TIME = 0.32
```

다음 상태가 동시에 나타나면 포화 상태로 판단한다.

```text
Dead-end에 충분한 로봇이 존재한다.
+
대부분의 로봇이 느리게 움직인다.
+
평균 밀도가 높다.
+
Dead-end 영역이 충분히 점유되어 있다.
+
전방 로봇이 더 이상 진행하지 않는다.
+
이 상태가 일정 시간 유지된다.
```

---

# 20. Adaptive Shepherd

## 20.1 Shepherd 선발 시점

Shepherd는 Branch 탐색이 시작되자마자 선발하지 않는다.

선두 Normal Robot이 Dead-end 근처의 Early Capture Region에 도착한 뒤 선발한다.

```text
Branch 탐색
→ 선두 로봇이 Dead-end Capture Region 진입
→ 필요한 후보 수 확인
→ 전방 통신 상태 확인
→ Shepherd 선발
```

현재 Capture Region 깊이는 다음과 같다.

```python
EARLY_CAPTURE_DEPTH = 34
```

---

## 20.2 폭 적응형 Shepherd 수

Shepherd 수는 고정값이 아니다.

다음 요소를 이용해 필요한 Shepherd 수를 계산한다.

* 복도 전체 폭
* 벽과 Shepherd 사이의 안전 Margin
* Shepherd Slot 사이의 목표 간격
* 최소 Shepherd 수
* 최대 Shepherd 수

현재 설정은 다음과 같다.

```python
SHEPHERD_MIN_COUNT = 5
SHEPHERD_MAX_COUNT = 14
SHEPHERD_EDGE_MARGIN = 12.0
SHEPHERD_TARGET_SLOT_SPACING = 12.5
```

복도 폭이 넓어지면 더 많은 Shepherd가 필요하고, 복도 폭이 좁아지면 더 적은 Shepherd가 필요하다.

---

## 20.3 Shepherd 후보 선발

Shepherd 후보는 다음 조건을 만족해야 한다.

* 현재 역할이 `NORMAL`이어야 한다.
* Base와 연결되어 있어야 한다.
* 현재 활성 Branch 안에 있어야 한다.
* Dead-end Capture Region 안에 있어야 한다.

후보는 Branch 끝에 가까운 순서대로 정렬한다.

필요한 Shepherd 수만큼 선두 로봇을 선택한다.

---

## 20.4 Shepherd Slot 할당

Shepherd Slot은 복도 폭 방향으로 균등하게 생성한다.

```text
Dead-end Wall

S   S   S   S   S   S   S   S

일반 로봇 군집
```

각 Shepherd 후보는 자신과 가장 가까운 미사용 Slot에 할당된다.

이를 통해 Shepherd들이 서로 교차하며 이동하는 현상을 줄인다.

---

# 21. Continuous Shepherd Curtain

## 21.1 필요성

실제 Shepherd Robot만으로 경계를 만들면 로봇 사이에 틈이 생길 수 있다.

특히 Shepherd가 Slot으로 이동하는 동안 일반 로봇이 다음과 같이 누출될 수 있다.

```text
S       S       S
    Normal Robot 통과
```

이를 방지하기 위해 복도 폭 전체를 가로지르는 가상 Curtain을 사용한다.

---

## 21.2 Curtain 활성화 시점

Curtain은 Shepherd가 모든 Slot에 도착한 뒤가 아니라 Shepherd 선발 직후 활성화한다.

다음 상태에서 Curtain이 유지된다.

```text
FORM_SHEPHERD_BOUNDARY
FILL_BEHIND_SHEPHERD
PRESSURE_PUSH
```

---

## 21.3 Smooth Curtain Force

Normal Robot이 Curtain에 가까워질수록 Junction 방향으로 밀어내는 힘이 증가한다.

로봇이 Dead-end 방향으로 빠르게 이동하고 있다면 추가적인 속도 감쇠도 적용한다.

```python
SHEPHERD_CURTAIN_INTERACTION_DEPTH = 24.0
SHEPHERD_CURTAIN_FORCE = 860.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_CURTAIN_VELOCITY_DAMPING = 18.0
```

---

## 21.4 Hard Safety Projection

Smooth Force만으로는 한 Frame 안에 빠른 로봇이 Curtain을 통과할 수 있다.

이를 방지하기 위해 최종 위치 보정도 적용한다.

Curtain을 통과한 Normal Robot에는 다음 처리를 적용한다.

1. 로봇 위치를 Curtain의 안전한 Junction 방향 위치로 되돌린다.
2. Dead-end 방향 속도를 제거한다.
3. Junction 방향으로 작은 복구 속도를 부여한다.

```python
SHEPHERD_CURTAIN_RECOVERY_SPEED = 10.0 * MOTION_SPEED_MULTIPLIER
```

Smooth Force는 자연스러운 움직임을 담당하고, Hard Projection은 수치적인 누출을 최종적으로 방지한다.

---

# 22. Moving Piston Pressure Backtracking

## 22.1 Shepherd Boundary 형성

선발된 Shepherd는 지정된 Slot으로 이동한다.

모든 Shepherd가 목표 위치의 허용오차 안에 들어오면 Boundary가 완성된 것으로 판단한다.

```python
SHEPHERD_FORM_TOLERANCE = 3.0
SHEPHERD_FORM_TIMEOUT = 2.4
```

정해진 시간 안에 Boundary가 완성되지 않으면 Shepherd 역할을 해제하고 Branch 탐색 단계로 돌아가 다시 선발한다.

불완전한 Boundary 상태에서 Pressure Push를 시작하지 않는다.

---

## 22.2 Fill Behind Shepherd

Boundary가 완성되면 일반 로봇을 Shepherd 뒤쪽으로 이동시킨다.

이 단계의 목적은 다음과 같다.

* Shepherd와 일반 로봇 사이의 빈 공간을 줄인다.
* 일반 로봇 군집을 충분히 압축한다.
* Shepherd 압력이 군집 전체로 전달될 수 있게 한다.
* Dead-end 포화 상태를 안정적으로 감지한다.

---

## 22.3 Pressure Push

포화 상태가 확인되면 Shepherd Pressure Push를 시작한다.

Pressure Push에서 Shepherd는 다음 역할을 동시에 수행한다.

* 높은 압력을 가진 경계 입자
* 복도 폭을 막는 실제 로봇 경계
* 연속 가상 Curtain
* Junction 방향으로 이동하는 Piston
* 일반 로봇에게 Junction 방향 Body Force를 전달하는 경계

현재 주요 값은 다음과 같다.

```python
SHEPHERD_PISTON_SPEED = 10.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_PISTON_MAX_TRAVEL = 24.0
SHEPHERD_PRESSURE_FACTOR = 5.2
VIRTUAL_PRESSURE_RADIUS = 60.0
VIRTUAL_PRESSURE_FORCE = 135.0
PRESSURE_RAMP_TIME = 0.8
```

Pressure는 즉시 최대값으로 적용하지 않고 일정 시간에 걸쳐 증가시킨다.

이를 통해 갑작스러운 충격과 로봇 튕김을 줄인다.

---

## 22.4 Backtracking 흐름 확인

Shepherd를 바로 해제하지 않고, Shepherd 근처 Normal Robot이 실제로 Junction 방향으로 움직이기 시작했는지 확인한다.

평가 항목은 다음과 같다.

* Junction 방향 속도가 기준 이상인 로봇 비율
* Junction 방향 평균속도
* 평가 가능한 Normal Robot 수
* 최소 Push 시간
* 조건 유지시간

현재 주요 값은 다음과 같다.

```python
SHEPHERD_MIN_PUSH_TIME = 0.20
FLOW_SPEED_THRESHOLD = 1.5
FLOW_RATIO_THRESHOLD = 0.45
FLOW_AVERAGE_SPEED_THRESHOLD = 1.8
FLOW_ESTABLISH_DWELL_TIME = 0.12
FLOW_MIN_NORMAL_COUNT = 6
FLOW_FALLBACK_TIME = 1.25
```

흐름이 형성되면 Shepherd를 다시 Normal Robot으로 전환하고 Junction 방향 초기속도를 부여한다.

흐름 조건이 충분히 형성되지 않더라도 Fallback Time이 지나면 Backtracking 단계로 전환한다.

---

# 23. 상태 머신

현재 시뮬레이션은 다음 10개 상태를 사용한다.

```python
MOVE_TO_JUNCTION
EXPLORE_BRANCH
FORM_SHEPHERD_BOUNDARY
FILL_BEHIND_SHEPHERD
PRESSURE_PUSH
FLOW_BACKTRACK
JUNCTION_SWITCH
FINAL_JUNCTION_GATHER
RETURN_TO_BASE
DONE
```

---

## 23.1 `MOVE_TO_JUNCTION`

초기 Base 영역에서 Junction으로 이동하는 단계다.

수행 기능은 다음과 같다.

* 로봇을 Junction 방향으로 이동시킨다.
* 초기 Lane을 유지한다.
* Junction에 가까워질수록 감속한다.
* Trunk Relay를 순차 배치한다.
* Anchor 후보의 Junction 도착시간을 기록한다.
* Junction Anchor를 선발한다.
* Anchor 주차 완료를 확인한다.
* Trunk Relay Plan 완료를 확인한다.
* Junction 안정성을 확인한다.
* 첫 번째 Branch를 선택한다.

---

## 23.2 `EXPLORE_BRANCH`

선택된 Branch를 탐색하는 단계다.

수행 기능은 다음과 같다.

* 선택된 Branch를 활성화한다.
* 미선택 Branch에 Virtual Valve를 적용한다.
* Geodesic EDF 방향으로 로봇을 이동시킨다.
* 전방 통신 상태를 지속적으로 확인한다.
* 필요할 경우 Branch Relay를 배치한다.
* Dead-end Capture Region 진입 여부를 확인한다.
* Shepherd 후보를 선발한다.

---

## 23.3 `FORM_SHEPHERD_BOUNDARY`

Shepherd가 복도 폭 방향으로 경계를 형성하는 단계다.

수행 기능은 다음과 같다.

* 폭 적응형 Shepherd Slot을 생성한다.
* Shepherd를 각 Slot으로 이동시킨다.
* Continuous Curtain을 즉시 활성화한다.
* Normal Robot의 전방 누출을 방지한다.
* Shepherd Boundary 완성을 확인한다.
* 형성 실패 시 Shepherd를 해제하고 재시도한다.

---

## 23.4 `FILL_BEHIND_SHEPHERD`

일반 로봇을 Shepherd 뒤쪽에 채우는 단계다.

수행 기능은 다음과 같다.

* 일반 로봇을 Shepherd 뒤쪽 목표점으로 이동시킨다.
* Branch Relay 연결을 유지한다.
* Dead-end 포화 상태를 측정한다.
* 충분한 압축과 정체가 확인되면 Pressure Push로 전환한다.

---

## 23.5 `PRESSURE_PUSH`

Shepherd가 Moving Piston으로 작동하는 단계다.

수행 기능은 다음과 같다.

* Shepherd 압력을 증가시킨다.
* Shepherd 경계를 Junction 방향으로 이동시킨다.
* Curtain도 Shepherd와 함께 이동시킨다.
* 일반 로봇에 약한 Junction 방향 Body Force를 적용한다.
* Backtracking 흐름이 형성되었는지 확인한다.
* 조건을 만족하면 Shepherd를 Normal Robot으로 해제한다.

---

## 23.6 `FLOW_BACKTRACK`

전체 군집이 Branch에서 Junction으로 복귀하는 단계다.

수행 기능은 다음과 같다.

* 모든 이동 로봇을 Junction 방향으로 유도한다.
* Branch 끝에 가까운 Relay부터 역순으로 회수한다.
* Branch 내부 잔여 로봇 수를 확인한다.
* Junction 복귀 로봇 수를 확인한다.
* Branch Relay가 모두 회수되었는지 확인한다.
* 활성 Branch를 `VISITED`로 변경한다.

---

## 23.7 `JUNCTION_SWITCH`

다음 Branch를 선택하기 위해 Junction에서 군집을 안정화하는 단계다.

수행 기능은 다음과 같다.

* 로봇을 Junction Staging 위치로 유도한다.
* Stable Path를 확인한다.
* Fast Path를 확인한다.
* 필요하면 Fallback Path를 사용한다.
* 남은 Branch에 대해 Proxy Region을 다시 생성한다.
* 후보별 SPH Short Rollout을 다시 수행한다.
* 다음 Branch를 선택한다.

Branch 선택은 최초 한 번만 수행하지 않는다. 로봇 군집이 Junction으로 복귀할 때마다 현재 상태를 다시 측정해 다음 Branch를 결정한다.

---

## 23.8 `FINAL_JUNCTION_GATHER`

모든 Branch 탐색 후 전체 로봇을 Junction에 모으는 단계다.

다음 조건을 확인한다.

* Branch 내부에 남은 로봇이 없어야 한다.
* Branch Relay가 남아 있지 않아야 한다.
* Shepherd가 남아 있지 않아야 한다.
* 모든 로봇이 Base와 연결되어 있어야 한다.
* 조건이 일정 시간 유지되어야 한다.

---

## 23.9 `RETURN_TO_BASE`

전체 군집이 Base로 복귀하는 단계다.

수행 기능은 다음과 같다.

* Junction Anchor 역할을 해제한다.
* Branch에 남은 로봇을 Junction으로 이동시킨다.
* Junction과 Bottom 로봇을 Base 방향으로 이동시킨다.
* Trunk Relay를 Junction 쪽부터 순차 회수한다.
* 전체 로봇의 Bottom 도착 여부를 확인한다.
* 모든 특수 역할이 해제되었는지 확인한다.

---

## 23.10 `DONE`

다음 조건을 모두 만족해야 시뮬레이션을 종료한다.

* 모든 로봇이 Bottom Region에 도착했다.
* Anchor 역할의 로봇이 없다.
* Branch Relay가 없다.
* Trunk Relay가 없다.
* Shepherd가 없다.

종료 시 실험 결과를 CSV 파일로 저장한다.

---

# 24. 시각화

## 24.1 Map 색상

각 Branch는 고유한 색상을 가진다.

| Branch  | 색상     |
| ------- | ------ |
| `UP`    | Blue   |
| `LEFT`  | Purple |
| `RIGHT` | Orange |

실제 Branch, Proxy Region, Branch Label, Proxy Assignment Point는 동일한 색상 계열을 사용한다.

이를 통해 실제 Branch와 분석용 Proxy Region의 관계를 쉽게 확인할 수 있다.

---

## 24.2 로봇 역할 색상

| 역할             | 색상    |
| -------------- | ----- |
| Normal         | 기본 청색 |
| Anchor         | 녹색    |
| Trunk Relay    | 갈색    |
| Branch Relay   | 주황색   |
| Shepherd       | 보라색   |
| Base 연결이 끊긴 로봇 | 붉은색   |

Density View를 활성화하면 Normal Robot의 색상이 밀도에 따라 변한다.

```text
낮은 밀도
→ 밝은 하늘색

중간 밀도
→ 파란색

높은 밀도
→ 짙은 남색
```

---

## 24.3 통신 링크

Base까지의 Widest Path에 포함된 Parent Link를 화면에 표시한다.

링크 색상은 통신거리에 따라 변경된다.

| 링크 상태        | 색상  |
| ------------ | --- |
| 안전거리 이하      | 녹색  |
| 경고거리         | 노란색 |
| 최대 통신거리에 가까움 | 붉은색 |

---

## 24.4 분석 영역

다음 영역을 화면에 표시할 수 있다.

* Junction 영역
* Anchor 선발 영역
* Anchor 주차 위치
* Base 위치
* Trunk Relay Slot
* Branch Relay Slot
* Shepherd Capture Region
* Dead-end Saturation Region
* Shepherd Slot
* Continuous Shepherd Curtain
* Proxy Region Partition
* Proxy Robot Assignment

---

# 25. HUD Panel

오른쪽 HUD Panel에는 다음 정보가 표시된다.

* 현재 FPS
* 전체 로봇 수
* 현재 Phase
* Anchor Robot ID
* Anchor 선발 점수
* 현재 활성 Branch
* 지금까지의 Branch 탐색 순서
* 최근 Branch 선택 비용
* Predicted Flow
* Density Disturbance
* Velocity Disturbance
* Communication Risk
* Branch별 Proxy Quota
* Branch별 실제 Proxy Mass
* 후보 Branch별 총비용
* 후보별 Primary Particle 수
* 후보별 Context Particle 수
* 현재 SPH 강성
* Branch 진입 초기 강성
* 강성 복원 진행시간
* Junction Consensus 상태
* Branch별 방문 상태
* Base 연결 로봇 수
* 최대 통신 Hop 수
* 최소 통신 Margin
* Trunk Relay 수
* Branch Relay 수
* 전방 통신 비율
* Relay 필요 여부
* Dead-end Tip Robot 수
* 저속 로봇 비율
* Dead-end 평균 밀도
* Dead-end 점유율
* 전방 정체량
* Saturation Dwell Time
* Shepherd 목표 수
* Shepherd Boundary 완성 여부
* Pressure Push 시간
* 전체 로봇 누적 이동거리
* 누적 통신 단절 Robot-Seconds

HUD는 Map 위에 겹치지 않고 별도의 오른쪽 Panel에 표시된다.

---

# 26. 실험 로그

시뮬레이션이 `DONE` 상태에 도달하거나 사용자가 프로그램을 종료하면 다음 파일을 생성한다.

```text
sph_dfs_experiment_summary.csv
```

저장되는 주요 정보는 다음과 같다.

## 26.1 전체 실험 정보

* 종료 이유
* 전체 로봇 수
* 전체 시뮬레이션 시간
* 전체 로봇 이동거리
* Normal 역할 이동거리
* Branch Relay 이동거리
* Trunk Relay 이동거리
* Shepherd 이동거리
* Anchor 이동거리

## 26.2 통신과 안전성

* 누적 통신 단절 Robot-Seconds
* 시뮬레이션 중 최소 로봇 간 거리
* 안전거리 위반 횟수

## 26.3 DFS Branch 선택

* 실제 Branch 탐색 순서
* Branch 선택 이벤트 수
* 각 Branch 선택 시점
* 선택된 Branch
* 선택 Branch의 총비용
* 완전탐색 우선순위
* 후보 Branch별 총비용
* 후보 Branch별 세부 비용

## 26.4 Dead-end와 Pressure Push

* Saturation Event 수
* Pressure Event 수
* Pressure Push 시작 시점
* Backtracking 흐름 형성 시점
* Pressure Push부터 흐름 형성까지 걸린 시간

---

# 27. 주요 파라미터

## 27.1 로봇과 SPH

| 파라미터                 |  기본값 | 설명        |
| -------------------- | ---: | --------- |
| `ROBOT_COUNT`        |  220 | 전체 로봇 수   |
| `ROBOT_RADIUS`       |    2 | 로봇 반지름    |
| `GRID_SPACING`       |    7 | 초기 로봇 간격  |
| `SMOOTHING_LENGTH`   |   28 | SPH 이웃 범위 |
| `PRESSURE_GAIN`      | 1650 | 압력 크기     |
| `STIFFNESS_EXPONENT` |  0.5 | 기본 SPH 강성 |
| `DAMPING`            |  2.3 | 속도 감쇠     |
| `SAFE_RADIUS`        |  7.5 | 충돌 안전거리   |
| `REPULSION_GAIN`     |  260 | 충돌 반발력    |

---

## 27.2 통신과 Relay

| 파라미터                                   |  기본값 | 설명              |
| -------------------------------------- | ---: | --------------- |
| `COMM_RANGE`                           |   46 | 최대 통신거리         |
| `COMM_SAFE_DISTANCE`                   |   34 | 안전 통신거리         |
| `COMM_RECOVERY_RANGE`                  |   84 | 연결 복구 탐색거리      |
| `TRUNK_RELAY_SPACING`                  |   30 | Trunk Relay 간격  |
| `RELAY_SPACING`                        |   30 | Branch Relay 간격 |
| `RELAY_FRONT_REQUIRED_CONNECTED_RATIO` | 0.90 | 전방 연결 요구비율      |
| `RETURN_TRUNK_READY_CONNECTED_RATIO`   | 0.97 | Trunk 회수 연결비율   |

---

## 27.3 Shepherd

| 파라미터                                 |  기본값 | 설명               |
| ------------------------------------ | ---: | ---------------- |
| `SHEPHERD_MIN_COUNT`                 |    5 | 최소 Shepherd 수    |
| `SHEPHERD_MAX_COUNT`                 |   14 | 최대 Shepherd 수    |
| `SHEPHERD_TARGET_SLOT_SPACING`       | 12.5 | Shepherd Slot 간격 |
| `SHEPHERD_FORM_TIMEOUT`              |  2.4 | Boundary 형성 제한시간 |
| `SHEPHERD_CURTAIN_INTERACTION_DEPTH` |   24 | Curtain 영향범위     |
| `SHEPHERD_PISTON_MAX_TRAVEL`         |   24 | Piston 최대 이동량    |
| `SHEPHERD_PRESSURE_FACTOR`           |  5.2 | Shepherd 압력 증가계수 |
| `PRESSURE_RAMP_TIME`                 |  0.8 | 압력 증가시간          |

---

## 27.4 Branch Rollout

| 파라미터                        |   기본값 | 설명                    |
| --------------------------- | ----: | --------------------- |
| `FLOW_ROLLOUT_HORIZON`      | 0.50초 | 가상 예측시간               |
| `FLOW_ROLLOUT_DT`           | 0.05초 | Rollout 시간간격          |
| `FLOW_ROLLOUT_MAX_ROBOTS`   |   190 | 최대 Rollout Particle 수 |
| `FLOW_ROLLOUT_TARGET_DEPTH` |    54 | 가상 Branch 진입 깊이       |
| `PROXY_CELL_SIZE`           |    10 | Proxy Grid 크기         |
| `PROXY_ROLLOUT_MIN_PRIMARY` |     6 | 최소 Primary Particle 수 |

---

# 28. 설치 방법

## 28.1 저장소 Clone

```bash
git clone <repository-url>
cd <repository-directory>
```

---

## 28.2 가상환경 생성

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

Linux 또는 macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 28.3 의존성 설치

`pygame-ce`를 사용하는 경우 다음 명령을 실행한다.

```bash
pip install pygame-ce
```

일반 `pygame`을 사용하는 경우 다음 명령을 실행한다.

```bash
pip install pygame
```

`requirements.txt`를 사용하는 경우 다음과 같이 설치한다.

```bash
pip install -r requirements.txt
```

예시 `requirements.txt`는 다음과 같다.

```text
pygame-ce>=2.5
```

---

# 29. 실행 방법

메인 실행 파일은 다음과 같다.

```text
single_junction_sph_dfs.py
```

다음 명령으로 실행한다.

```bash
python single_junction_sph_dfs.py
```

Windows 환경에서 `python` 명령이 동작하지 않는 경우 다음 명령을 사용할 수 있다.

```powershell
py single_junction_sph_dfs.py
```

프로그램이 정상적으로 실행되면 Terminal에 다음과 같은 로그가 출력된다.

```text
[Base Trunk] slots=...
robots=220, mean_density=..., rho0=...
[Anchor] robot=..., score=...
[Proxy Rollout] branch=...
[DFS] selected=...
```

---

# 30. 키보드 조작

| 키       | 기능                        |
| ------- | ------------------------- |
| `SPACE` | 시뮬레이션 일시정지 또는 재개          |
| `R`     | 시뮬레이션 초기화                 |
| `D`     | Density Color 표시 전환       |
| `V`     | Proxy Region과 분석 영역 표시 전환 |
| `C`     | 통신 링크 표시 전환               |
| `ESC`   | 프로그램 종료                   |

---

# 31. 권장 프로젝트 구조

```text
PDFS-Multi-Robot/
├── README.md
├── requirements.txt
├── pygame_simulator/
│   ├── single_junction_sph_dfs.py
│   └── sph_dfs_experiment_summary.csv
├── docs/
│   ├── images/
│   │   ├── simulation_overview.png
│   │   ├── proxy_partition.png
│   │   ├── shepherd_boundary.png
│   │   └── communication_relay.png
│   └── research_notes/
└── results/
    ├── experiment_01.csv
    └── experiment_02.csv
```

README에 시뮬레이션 이미지를 추가할 경우 다음 형식을 사용할 수 있다.

```html
<p align="center">
  <img src="docs/images/simulation_overview.png" width="900">
</p>
```

동영상을 GIF로 변환해 추가할 경우 다음과 같이 사용할 수 있다.

```html
<p align="center">
  <img src="docs/images/single_junction_demo.gif" width="900">
</p>
```

---

# 32. 기존 방식과의 차이

## 32.1 고정 Branch 순서

기존 방식은 Branch를 미리 정해진 순서로 탐색한다.

```text
UP → LEFT → RIGHT
```

현재 방식은 Junction에 복귀할 때마다 군집 상태를 다시 평가한다.

```text
현재 군집 상태 측정
→ Proxy Region 생성
→ 후보별 SPH Short Rollout
→ 유량, 밀도, 속도, 통신, Relay 비용 비교
→ 가장 자연스러운 Branch 선택
```

---

## 32.2 고정 Shepherd 수

기존 방식은 복도 폭과 관계없이 항상 같은 수의 Shepherd를 사용한다.

현재 방식은 복도 유효 폭과 Slot 간격을 이용해 필요한 Shepherd 수를 계산한다.

```text
복도 폭 측정
→ 벽 Margin 제외
→ 필요한 Shepherd Slot 수 계산
→ 필요한 수만큼 선두 로봇 선발
```

---

## 32.3 개별 Shepherd 경계

개별 Shepherd Robot만 사용하는 경우 로봇 사이의 틈으로 일반 로봇이 누출될 수 있다.

현재 방식은 다음 세 요소를 함께 사용한다.

```text
실제 Shepherd Robot
+
연속 가상 Curtain Force
+
Hard Safety Projection
```

---

## 32.4 Relay 일괄 회수

모든 Relay를 동시에 해제하면 통신이 급격히 끊길 수 있다.

현재 방식은 탐색 진행 방향의 역순으로 Relay를 회수한다.

```text
가장 먼 Branch Relay부터 회수
→ Junction 복귀
→ 모든 Branch 완료
→ Junction 쪽 Trunk Relay부터 회수
→ 전체 Base 복귀
```

---

# 33. 현재 한계

## 33.1 단일 Junction

현재 Topological Graph는 다음 구조만 포함한다.

```text
BASE
  │
JUNCTION
 ├─ UP_TARGET
 ├─ LEFT_TARGET
 └─ RIGHT_TARGET
```

다중 Junction DFS를 구현하려면 다음 기능이 필요하다.

* Node별 Parent 저장
* Node별 Child Branch 목록
* DFS Stack
* Junction별 Anchor
* Junction별 Branch 상태
* Parent Junction 복귀 경로
* 다중 깊이 Relay 관리
* 재귀적인 Branch Ordering

---

## 33.2 이상적인 로봇 이동 모델

현재 로봇은 평면에서 모든 방향으로 이동할 수 있는 입자에 가깝다.

실제 TurtleBot과 같은 차동구동 로봇에 적용하려면 다음 기능이 필요하다.

* 목표 평면속도를 선속도와 각속도로 변환
* 로봇 Heading 관리
* 최대 회전속도 제한
* 회전반경 제한
* 모터 응답 지연
* Wheel Slip
* Odometry 오차

---

## 33.3 단순 통신 모델

현재 통신 모델은 거리와 LOS를 중심으로 구성된다.

포함되지 않은 요소는 다음과 같다.

* RSSI
* 벽 재질에 따른 신호 감쇠
* 다중경로
* 패킷 손실
* 통신 지연
* 대역폭
* 통신 간섭
* 메시지 충돌

---

## 33.4 동적 장애물 없음

현재 Branch는 항상 물리적으로 통과 가능하다고 가정한다.

화재 환경과 동적 장애물을 적용하려면 다음 Edge 상태가 필요하다.

```text
UNVISITED
ACTIVE
VISITED
TEMP_WAIT
BLOCKED
UNREACHABLE
```

---

## 33.5 중앙집중식 계산

현재 다음 기능은 하나의 Pygame 시뮬레이터에서 중앙집중적으로 계산된다.

* Anchor 선발
* 통신 그래프 계산
* Branch 후보 평가
* Proxy Region 생성
* SPH Short Rollout
* Relay 선발
* Shepherd 선발
* 전체 State Machine

실제 분산 시스템으로 확장하려면 다음 기능이 필요하다.

* 로봇별 독립 Node
* 국소 Neighbor 정보
* 분산 Anchor 합의
* Branch Score 교환
* 메시지 지연 처리
* 로봇별 독립 State Machine
* 통신 실패 처리
* 로봇 고장 처리

---

# 34. 향후 개발 계획

## Phase 1: Single-Junction Baseline

* [x] SPH 기반 군집 이동
* [x] Base-rooted LOS 통신
* [x] 영구 Trunk Relay
* [x] 적응형 Branch Relay
* [x] Junction Anchor 선발
* [x] Proxy Region 분할
* [x] 후보별 SPH Short Rollout
* [x] Flow-Preserving Branch Ordering
* [x] Adaptive SPH Stiffness
* [x] Dead-end Saturation Detection
* [x] Width-Adaptive Shepherd
* [x] Continuous Shepherd Curtain
* [x] Moving Piston Backtracking
* [x] Branch Relay 순차 회수
* [x] Trunk Relay 순차 회수
* [x] 실험 CSV Logging

---

## Phase 2: Multi-Junction Physical DFS

* [ ] 일반 Topological Graph 입력
* [ ] Rooted DFS Tree 생성
* [ ] DFS Stack 구현
* [ ] Junction별 Parent–Child 관계 관리
* [ ] Junction별 Anchor 생성과 회수
* [ ] 다중 깊이 Branch 탐색
* [ ] Parent Junction 기반 Backtracking
* [ ] 여러 Junction에 걸친 Relay Chain 관리
* [ ] Junction별 Proxy Region과 Branch Rollout

---

## Phase 3: Dynamic Fire Environment

* [ ] 동적 장애물 생성
* [ ] Edge 상태 Online Update
* [ ] `TEMP_WAIT` 처리
* [ ] `BLOCKED` 처리
* [ ] Base 기준 Reachability 재계산
* [ ] Reachability-Aware DFS Tree Repair
* [ ] 비트리 Edge를 이용한 대체 경로 탐색
* [ ] 도달 불가능 영역 기록
* [ ] 위험도 기반 Branch 우선순위

---

## Phase 4: ROS 2와 Gazebo

* [ ] TurtleBot3 모델 적용
* [ ] 차동구동 제어기 구현
* [ ] ROS 2 Topic 분리
* [ ] 로봇별 분산 Node
* [ ] Gazebo Harmonic 환경 구축
* [ ] LiDAR 기반 장애물 감지
* [ ] Odometry 기반 위치 추정
* [ ] 실제 통신 품질 모델
* [ ] Pygame 결과와 Gazebo 결과 비교

---

## Phase 5: 실험과 성능 비교

다음 Baseline과 비교할 예정이다.

* Fixed DFS Branch Order
* Shortest Branch First
* Longest Branch First
* Random Branch Order
* Relay Cost Only
* Current Flow Direction Only
* No Proxy Region
* No SPH Rollout
* No Shepherd Curtain
* Fixed Shepherd Count
* Full Proxy-Region SPH Rollout

평가 지표는 다음과 같다.

* 전체 탐색시간
* 전체 로봇 이동거리
* Relay 이동거리
* Shepherd 이동거리
* Branch 전환시간
* Junction 대기시간
* 통신 단절 Robot-Seconds
* 최소 로봇 간 거리
* 안전거리 위반 횟수
* Branch 선택 계산시간
* Density Disturbance
* Velocity Disturbance
* Pressure-to-Flow Latency
* Branch Relay 회수시간
* Trunk Relay 회수시간
* 최종 Base 복귀 성공률

---

# 35. 연구 기여 후보

## 35.1 SPH 상태 기반 DFS Child Ordering

고정된 DFS Child 순서를 사용하지 않고 현재 군집의 물리 상태를 이용해 다음 Child Branch를 선택한다.

평가 상태에는 다음이 포함된다.

* 밀도 분포
* 속도장
* 군집의 주축
* Branch 입구 혼잡도
* 예상 유량
* 통신 위험
* Relay 요구량
* 방향 전환량
* SPH 물성 전환량

---

## 35.2 Proxy Region 기반 국소 Rollout

각 후보 Branch를 평가할 때 전체 군집을 모두 사용하는 대신 해당 Branch의 Proxy Region에 할당된 Primary Particle과 경계 Context Particle을 사용한다.

이를 통해 다음 효과를 목표로 한다.

* 후보 평가 계산량 감소
* Branch별 국소 유체상태 반영
* 인접 유체 경계 영향 유지
* 실제 군집 상태 비변경
* 후보별 로봇 질량 요구량 반영

---

## 35.3 DFS 상태 기반 Adaptive Compressibility

SPH 강성을 고정하지 않고 다음 상태에 따라 변경한다.

* Branch 방향 전환
* Branch 진입
* Junction Switch
* Dead-end Pressure Push
* Backtracking

이를 통해 방향 전환 시에는 군집을 부드럽게 변형하고, Pressure Push 시에는 압력을 강하게 전달한다.

---

## 35.4 Shepherd Curtain과 Moving Piston

개별 Shepherd Robot만 사용하는 대신 다음을 결합한다.

* 폭 적응형 Shepherd Robot
* 연속 가상 Curtain
* Hard Safety Projection
* Moving Piston
* Pressure Ramp
* 흐름 감지 기반 Shepherd 해제

---

## 35.5 통신을 보존하는 Relay 회수

Branch Relay와 Trunk Relay를 탐색 진행 방향의 역순으로 회수한다.

Relay 회수 조건에 Base 연결 비율과 통신 Margin을 사용해 회수 과정의 통신 단절을 줄인다.

---

# 36. 프로젝트 상태

현재 프로젝트 상태는 다음과 같다.

```text
Research Prototype
Single-Junction Validation
Pygame-Based Simulation
```

본 코드는 완성된 실제 로봇 시스템이 아니라 다음 단계의 연구를 위한 프로토타입이다.

```text
단일 Junction 알고리즘 검증
→ 성능 지표 수집
→ Baseline 비교
→ 다중 Junction 확장
→ ROS 2와 Gazebo 적용
→ 실제 로봇 검증
```

---
---
Last update : 2026.07.16
