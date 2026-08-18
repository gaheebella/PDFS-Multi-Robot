# PDFS-Multi-Robot

## Overview

프로젝트 전체 목표:
localization-free / map-free Physical DFS,
LEFT/RIGHT/UP 같은 고정 방향 없이
LiDAR, local communication, swarm motion으로
Junction/Branch를 발견하고 DFS 탐색하는 시스템.

## Current Research Status

### 1. Stable-Motion Branch Orientation
- SPH 로봇의 안정적인 군집 흐름으로 Branch 방향 추정
- nominal에서는 잘 됨
- wide/production-scale에서는 안정적으로 틀릴 수 있음
- Motion 단독 사용 불가

### 2. Point Cloud Branch Orientation
- opening center 방식은 Anchor offset에서 bias 발생
- corridor wall 방향을 fitting하는 wall-parallel 방식으로 변경
- arbitrary angle, Anchor offset, rotation 등에 강건
- 현재 Geometry 방향의 기본 방식

### 3. Geometry + Motion Fusion
- Wall geometry와 Stable-motion은 실패 조건이 다름
- Geometry reliability + Motion reliability를 이용해 fusion
- 서로 일치하면 결합
- 한쪽이 약하면 다른 쪽을 사용
- 강하게 충돌하면 conflict로 abstain
- Branch orientation interface는 현재 freeze 가능한 수준

## Current Work: Point Cloud Detector Redesign

기존 Point Cloud detector의 magic-number heuristic을
sensor noise, angular resolution, physical width,
boundary uncertainty 같은 물리/통계 기준으로 바꾸는 중.

새 uncertainty-aware detector는:
- boundary accuracy 개선
- wall availability 개선
- 4° resolution 문제 해결
- 기존 26~30° fusion failure 해결

하지만 dropout / partial visibility에서
false opening이 크게 증가함.

현재 핵심 문제:

real opening
→ max_range

dropout / missing return
→ max_range

두 경우가 single scan에서는 동일하게 보일 수 있음.

## Next Steps

1. LiDAR sensor representation audit
2. real opening과 dropout-induced max-range 구분
   - explicit no-return / invalid-return 정보가 있는지 확인
   - 없으면 temporal multi-scan persistence 검증
3. false opening 억제 후 detector freeze
4. final Branch orientation과 연결
5. Anchor / Guard / Shepherd / Handoff / Backflow / DFS 통합
6. end-to-end unknown-map evaluation
