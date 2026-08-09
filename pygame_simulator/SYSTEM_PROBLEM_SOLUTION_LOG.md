# SPH-DFS 시스템 상태 및 문제-해결 이력

이 문서는 single_junction_sph_dfs_environment.py의 현재 구현 상태와 지금까지 발견한 문제, 원인, 해결 방법, 검증 범위를 다른 로컬에서도 이어갈 수 있도록 기록한다.

## 1. 다른 로컬에서 재개하기

저장소: https://github.com/gaheebella/PDFS-Multi-Robot.git
작업 브랜치: claude/pygame-simulator-review-aj16vc
Draft PR: https://github.com/gaheebella/PDFS-Multi-Robot/pull/1

새 Windows 로컬:

    git clone https://github.com/gaheebella/PDFS-Multi-Robot.git
    cd PDFS-Multi-Robot
    git switch claude/pygame-simulator-review-aj16vc
    powershell -ExecutionPolicy Bypass -File .\pygame_simulator\run_single_junction_sph_dfs.ps1

이미 clone한 로컬:

    cd PDFS-Multi-Robot
    git switch claude/pygame-simulator-review-aj16vc
    git pull --ff-only
    powershell -ExecutionPolicy Bypass -File .\pygame_simulator\run_single_junction_sph_dfs.ps1

주요 파일:

- pygame_simulator/single_junction_sph_dfs_environment.py: 주 시뮬레이터
- pygame_simulator/README_SINGLE_JUNCTION_SPH_DFS.md: 설치와 실행, 알고리즘 요약
- pygame_simulator/requirements-sph-dfs.txt: Python 의존성
- pygame_simulator/run_single_junction_sph_dfs.ps1: Windows 자동 실행기
- pygame_simulator/SYSTEM_PROBLEM_SOLUTION_LOG.md: 현재 문서

## 2. 현재 시스템 개요

현재 시스템은 Base 하나, 3방향 Junction 하나, dead-end Branch 세 개를 사용하는 SPH 기반 Physical DFS prototype이다.

역할:

- NORMAL: SPH 압력과 로컬 이웃 상호작용으로 이동
- ANCHOR: Junction 합의와 방문 상태 저장 및 중계
- RELAY/TRUNK_RELAY: Base 연결을 유지하는 breadcrumb 통신 로봇
- JUNCTION_GUARD: 미탐색 Branch 입구의 물리 gatekeeper
- FRONTIER_SHEPHERD: 선택 Branch에서 NORMAL 앞을 따라 이동하는 조밀한 대열
- SHEPHERD: dead-end 접촉 후 같은 ID로 backtracking하는 대열
- PRE_SHEPHERD: 다음 Branch 경계를 미리 준비하는 역할

연구 아이디어 대응:

- Emmons 계열: 로컬 군집 분포, 횡방향 확장, Branch crossing cohort로 Junction 추론
- Eguchi 계열: 명령 속도와 관측 속도의 tracking error로 간접 접촉 증거 축적
- SPH: 밀도, 압력, 점성, 평형거리 기반 군집 이동
- DFS: 로컬 비용 합의로 미방문 Branch를 선택하고 dead-end 후 Junction으로 복귀

## 3. 현재 실행 흐름

1. 모든 Branch를 열어 둔 상태로 Base에서 자유 확산한다.
2. 로봇이 실제로 Branch 입구를 넘은 crossing cohort가 누적되어야 Branch를 발견한다.
3. 횡방향 확장과 둘 이상의 유효 cohort가 유지되면 Junction을 확정한다.
4. 각 발견 Branch의 바깥쪽 terminal 로봇을 leader로 정한다.
5. leader 기준 통신 그래프에서 최대 4-hop 로봇을 모집한다.
6. 모든 Branch에 최초 물리 guard 횡단면을 만든 뒤 분산 투표로 탐색 Branch를 선택한다.
7. 미선택 Branch는 2~4층 K-hop mouth wall을 만든다.
8. 선택 Branch는 통로 폭과 SAFE_RADIUS로 조밀한 moving line을 만든다.
9. 현재 통로에서는 moving line 목표가 17대이며 부족한 수는 K-hop으로 추가 모집한다.
10. moving line은 NORMAL 선두보다 일정 간격만 앞서 함께 이동한다.
11. 여러 frontier shepherd가 충분한 횡폭에서 직접 전방 접촉하고 속도가 낮아야 dead-end를 확정한다.
12. 같은 shepherd ID가 dead-end 횡단면으로 정렬된 뒤 NORMAL을 몰며 backtracking한다.
13. 미탐색 Branch의 thick wall은 해체하지 않고 같은 ID, anchor, 열, 층으로 유지한다.
14. 모든 Branch 방문 후 임시 역할을 해제하고 Base로 귀환한다.

## 4. 핵심 정책과 파라미터

- JUNCTION_GUARD_MAX_HOPS = 4
- Thick mouth wall = 2~4 layers
- SHEPHERD_MAX_COUNT = 22
- Moving shepherd 목표 간격 = SAFE_RADIUS × 0.85
- 현재 corridor의 moving shepherd 목표 = 17대
- Mouth wall policy = ADAPTIVE_KHOP_LAYERED_MOUTH_WALL_V1
- Junction inference = LOCAL_CROSSING_DENSITY_COHORT_V2
- Indirect contact = EGUCHI_TRACKING_ERROR_V1
- Frontier policy = PERSISTENT_CROSS_SECTION_SHEPHERD_V1

Layer 수는 미래 유입량을 정확히 예측하는 모델이 아니다. 전체 swarm 크기, 입구 주변 NORMAL 수, 평균 밀도를 사용하는 2~4층 heuristic이다. 한 번 완성된 미탐색 wall은 이후 더 좁게 재계산하거나 1열로 축소하지 않는다.

## 5. 문제 상황과 해결 방법

### 5.1 도달하지 않은 Branch를 미리 아는 문제

문제:

- 초기 구현은 Branch 좌표와 방향으로 모든 출구를 미리 아는 것처럼 동작했다.
- 로봇이 UP Branch에 도달하기 전 존재를 아는 것은 unknown-map 주장과 충돌했다.

해결:

- 초기 gate를 모두 열고 자유 확산한다.
- 로봇이 입구를 실제로 넘은 cohort의 수, 깊이, 지속시간을 사용한다.
- 관측되지 않은 Branch에는 guard를 만들지 않는다.

### 5.2 Junction 인식이 지나치게 오래 걸리는 문제

문제:

- 새로운 crossing 로봇이 일정 시간 전혀 없어야 한다는 조건 때문에 발견 후에도 오래 대기했다.
- 낮은 FPS에서는 짧은 simulation dwell도 실제 시간으로 길어졌다.

해결:

- 동일 Branch의 crossing 증가를 새로운 환경 특징으로 보지 않는다.
- 유효 Branch 집합이 바뀌거나 Junction signature가 사라질 때만 settle timer를 초기화한다.

### 5.3 Shepherd가 dead-end를 미리 알고 돌진하는 문제

문제:

- Branch 길이와 dead-end 좌표로 만든 슬롯을 향해 shepherd가 NORMAL보다 먼저 달렸다.
- 실제 로봇은 접촉하지 않은 dead-end 위치를 알 수 없다.

해결:

- 선택 입구의 같은 FRONTIER_SHEPHERD ID가 NORMAL 선두를 따라 이동한다.
- terminal 좌표가 아니라 frontier의 직접 전방 접촉으로 dead-end 전환을 시작한다.

### 5.4 Dead-end 도달 전 backtracking하는 문제

문제:

- 밀도나 정체만으로 dead-end를 오인해 Branch 중간에서 backtracking했다.

해결:

- 여러 frontier shepherd의 직접 접촉 비율을 요구한다.
- 접촉점이 통로 횡폭의 충분한 비율을 덮어야 한다.
- 낮은 전방 속도와 dwell 조건을 함께 요구한다.
- moving line 전체가 실제 접촉 깊이에 도달한 뒤 return shepherd로 전환한다.

### 5.5 미선택 Branch를 가상 벽으로 막는 문제

문제:

- Gate=CLOSED가 is_walkable 이동 거부로 구현되어 시뮬레이터만 가능한 invisible wall이었다.

해결:

- 논리 gate 명령과 물리 구현을 분리했다.
- 미선택 Branch에 실제 JUNCTION_GUARD를 남긴다.
- guard 방향과 상대 위치로 NORMAL을 Junction 쪽으로 유도한다.
- HUD에서 Gate commands와 Physical mouth guards를 구분한다.

주의:

- selected dead-end return boundary에는 아직 simulator curtain 메커니즘이 남아 있다.
- 따라서 전체 시스템을 완전한 no-virtual-boundary 또는 localization-free 구현이라고 주장하면 안 된다.

### 5.6 미선택 Branch wall이 얇아 로봇이 새는 문제

문제:

- 최초 guard가 통로 폭만 덮는 1열이어서 지속적인 SPH 압력을 견디지 못했다.

해결:

- leader를 seed로 최대 4-hop NORMAL을 모집한다.
- swarm 크기, 입구 주변 mass와 density에 따라 2~4층 wall을 만든다.
- 모든 미선택 wall이 anchor에 도착할 때까지 선택 flow를 시작하지 않는다.

### 5.7 Branch 전환 때 thick wall이 1열로 붕괴하는 문제

문제:

- begin_junction_guard_formation이 모든 guard를 NORMAL로 해제했다.
- 다음 선택에서 통로 폭이 다시 작게 추정되어 9열×3층이 7열×3층으로 축소되기도 했다.
- 재형성 중 틈으로 로봇이 유출됐다.

해결:

- 미탐색 thick wall의 robot ID, anchor, column, layer를 보존한다.
- Branch 전환 때 기존 wall을 해체하거나 재선출하지 않는다.
- 조금 밀려도 같은 anchor로 복귀시키고 안정될 때까지 선택 flow를 보류한다.
- 해당 Branch가 실제 선택될 때만 wall을 연다.

### 5.8 선택 Branch moving shepherd가 5대로 줄어드는 문제

문제:

- 최초 선택 Branch는 thick_mouth_guard_columns가 아직 0이었다.
- fallback 최소값 5가 원래 8~9대 횡단면도 5대로 잘랐다.
- dead-end와 backtracking에도 큰 틈이 남았다.

해결:

- moving line 수를 통로 유효 폭과 SAFE_RADIUS로 계산한다.
- 슬롯 간격을 SAFE_RADIUS × 0.85로 설정해 영향 영역이 겹치게 한다.
- 목표 수가 부족하면 leader-rooted K-hop pool에서 추가 모집한다.
- 현재 통로에서는 17대를 사용한다.
- 동일 17개 ID를 dead-end와 Junction backtracking까지 유지한다.

### 5.9 화면에 분홍색 흔적이 누적되는 문제

문제:

- Eguchi tracking-error contact point를 분홍색 원으로 그려 이동 흔적처럼 보였다.

해결:

- contact point 데이터와 inference 계산은 유지한다.
- draw_collision_points 호출만 제거해 화면에는 그리지 않는다.
- 검증 이미지에서 contact 색상 (220, 45, 150) 픽셀이 0개임을 확인했다.

## 6. 검증 결과

정적 검사:

    python -m py_compile pygame_simulator\single_junction_sph_dfs_environment.py
    git diff --check

두 검사 모두 통과했다.

160대 장시간 headless 검증:

- UP moving frontier = 17 IDs
- UP dead-end 전환 = 동일 17 IDs
- UP backtracking = line retained to Junction=17
- RIGHT moving frontier = 17 IDs
- RIGHT dead-end 전환 = 동일 17 IDs
- RIGHT backtracking = line retained to Junction=17
- 다음 LEFT 선택에서도 17대 line 구성
- LEFT 미탐색 wall = 5 columns × 3 layers = 15대 유지
- 분홍 contact trace pixel = 0

680대 검증 범위:

- 이전 시각 검증에서 미선택 wall이 Branch별 24~27대, 즉 8~9 columns × 3 layers로 형성됐다.
- 최신 persistent/dense-line 코드의 72초 smoke run은 오류 없이 종료했다.
- 낮은 FPS 때문에 해당 시간 안에 완전한 두 번째 Branch 전환까지 도달하지 않았다.
- 680대의 세 Branch 전체 완료는 후속 장시간 회귀 항목이다.

## 7. 현재 전제와 연구 한계

현재 구현은 unknown-map 방향으로 개선됐지만 완전히 localization-free하지 않다.

남은 global/map-aware 요소:

- 고정된 cross-shaped fixture
- UP, LEFT, RIGHT Branch 축과 영역 판정
- Junction 절대 영역과 Anchor parking slot
- Branch-relative Shepherd/Guard target slot
- 렌더러와 collision mask가 아는 벽 형상
- selected dead-end return boundary의 simulator curtain
- 완료 Branch Pebble/Marker 미구현
- multi-junction recursive DFS와 tree repair 미구현

정확한 현재 표현:

- Junction/Branch 발견 trigger는 local distribution과 physical crossing evidence 기반
- dead-end trigger는 frontier direct contact 기반
- unselected Branch 차단은 physical guard 기반이고 logical CLOSED는 geofence가 아님
- 전체 controller가 완전한 map-free/localization-free 단계는 아님

## 8. 다음 작업 권장 순서

1. 680대로 세 Branch 전체 장시간 회귀 및 누출 수 계측
2. selected dead-end simulator curtain 제거 후 17대 물리 line만으로 회귀
3. 완료 Branch에 physical Pebble/Marker를 남겨 재진입 방지
4. Anchor를 고정 slot이 아닌 선출 당시 current pose에 유지
5. Branch 축과 영역 판정을 cohort-derived local frame으로 교체
6. multi-junction DFS stack, Marker propagation, relay tree repair 구현

## 9. 이어서 작업할 때 확인

    git status -sb
    git log -5 --oneline
    python -m py_compile pygame_simulator\single_junction_sph_dfs_environment.py

실행 후 HUD에서 확인:

- Gate commands (no geofence)
- Physical mouth guards
- Thick K-hop walls
- Persistent frontier
- Dead-end inference

한 번에 하나의 메커니즘만 변경하고 160대 빠른 회귀 후 680대 장시간 회귀를 수행한다. 생성되는 CSV, screenshot, .venv-sph-dfs, __pycache__, .pre_*, .bak-* 파일은 Git에 포함하지 않는다.
