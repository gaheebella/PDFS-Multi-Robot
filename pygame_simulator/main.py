import math
import random
import sys
from collections import deque
from enum import Enum, auto

import pygame


pygame.init()

# =========================================================
# 1. 화면 설정
# =========================================================

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
FPS = 60
SUBSTEPS = 2

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("SPH + DFS Anchor + Temporary Relay Chain")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)
small_font = pygame.font.SysFont(None, 20)

# 부드러운 색상
BACKGROUND_COLOR = (248, 249, 252)
FLOOR_COLOR = (235, 239, 246)
WALL_COLOR = (96, 106, 124)
TEXT_COLOR = (58, 67, 82)
ROBOT_BASE_COLOR = (115, 165, 208)
SHEPHERD_COLOR = (171, 145, 205)
ANCHOR_COLOR = (102, 174, 137)
RELAY_COLOR = (226, 151, 82)
RELAY_SLOT_COLOR = (196, 129, 65)
JUNCTION_COLOR = (153, 164, 181)
END_REGION_COLOR = (224, 171, 115)

# 통신 상태 시각화 색상
COMM_LINK_SAFE_COLOR = (132, 190, 158)
COMM_LINK_WARNING_COLOR = (226, 177, 96)
COMM_LINK_DANGER_COLOR = (214, 103, 103)
DISCONNECTED_COLOR = (205, 96, 96)
COMM_RANGE_COLOR = (142, 154, 174)

# =========================================================
# 2. 십자가 맵
# =========================================================

center_x = 400
center_y = 350
corridor_width = 120
half_width = corridor_width // 2
normal_length = 180
right_length = normal_length * 2

cross_points = [
    (center_x - half_width, center_y - half_width - normal_length),
    (center_x + half_width, center_y - half_width - normal_length),
    (center_x + half_width, center_y - half_width),
    (center_x + half_width + right_length, center_y - half_width),
    (center_x + half_width + right_length, center_y + half_width),
    (center_x + half_width, center_y + half_width),
    (center_x + half_width, center_y + half_width + normal_length),
    (center_x - half_width, center_y + half_width + normal_length),
    (center_x - half_width, center_y + half_width),
    (center_x - half_width - normal_length, center_y + half_width),
    (center_x - half_width - normal_length, center_y - half_width),
    (center_x - half_width, center_y - half_width),
]

junction_rect = pygame.Rect(
    center_x - half_width,
    center_y - half_width,
    corridor_width,
    corridor_width,
)

# Junction 내부의 작은 사각형에 가장 먼저 진입한 일반 로봇을 Anchor로 선발합니다.
ANCHOR_REGION_SIZE = 70
anchor_election_rect = pygame.Rect(
    center_x - ANCHOR_REGION_SIZE // 2,
    center_y - ANCHOR_REGION_SIZE // 2,
    ANCHOR_REGION_SIZE,
    ANCHOR_REGION_SIZE,
)

# 중앙 통행을 막지 않도록 Junction 가장자리 쪽에 Anchor를 주차합니다.
ANCHOR_PARK_POSITION = pygame.Vector2(
    center_x - half_width + 14,
    center_y - half_width + 14,
)

# Branch 전환 중 일반 로봇이 잠시 모이는 지점입니다.
JUNCTION_STAGING_POSITION = pygame.Vector2(
    center_x + 10,
    center_y + 10,
)

up_rect = pygame.Rect(
    center_x - half_width,
    center_y - half_width - normal_length,
    corridor_width,
    normal_length,
)

left_rect = pygame.Rect(
    center_x - half_width - normal_length,
    center_y - half_width,
    normal_length,
    corridor_width,
)

right_rect = pygame.Rect(
    center_x + half_width,
    center_y - half_width,
    right_length,
    corridor_width,
)

bottom_rect = pygame.Rect(
    center_x - half_width,
    center_y + half_width,
    corridor_width,
    normal_length,
)

END_REGION_DEPTH = 28

dead_end_regions = {
    "UP": pygame.Rect(
        center_x - half_width,
        center_y - half_width - normal_length,
        corridor_width,
        END_REGION_DEPTH,
    ),
    "LEFT": pygame.Rect(
        center_x - half_width - normal_length,
        center_y - half_width,
        END_REGION_DEPTH,
        corridor_width,
    ),
    "RIGHT": pygame.Rect(
        center_x + half_width + right_length - END_REGION_DEPTH,
        center_y - half_width,
        END_REGION_DEPTH,
        corridor_width,
    ),
}

# Shepherd는 군집 전체가 포화될 때까지 기다리지 않고,
# dead-end에 가장 먼저 도착한 로봇 8대가 이 영역에 들어오면 즉시 선발됩니다.
early_capture_regions = {
    "UP": pygame.Rect(
        center_x - half_width,
        center_y - half_width - normal_length,
        corridor_width,
        34,
    ),
    "LEFT": pygame.Rect(
        center_x - half_width - normal_length,
        center_y - half_width,
        34,
        corridor_width,
    ),
    "RIGHT": pygame.Rect(
        center_x + half_width + right_length - 34,
        center_y - half_width,
        34,
        corridor_width,
    ),
}

# =========================================================
# 3. DFS / Shepherd 상태
# =========================================================


class SimulationPhase(Enum):
    MOVE_TO_JUNCTION = auto()
    EXPLORE_BRANCH = auto()
    FORM_SHEPHERD_BOUNDARY = auto()
    PRESSURE_PUSH = auto()
    FLOW_BACKTRACK = auto()
    JUNCTION_SWITCH = auto()
    RETURN_TO_BASE = auto()
    DONE = auto()


DFS_ORDER = ["UP", "LEFT", "RIGHT"]
BRANCH_DIRECTIONS = {
    "UP": pygame.Vector2(0.0, -1.0),
    "LEFT": pygame.Vector2(-1.0, 0.0),
    "RIGHT": pygame.Vector2(1.0, 0.0),
}

phase = SimulationPhase.MOVE_TO_JUNCTION
active_branch = DFS_ORDER[0]
branch_states = {
    "UP": "UNVISITED",
    "LEFT": "UNVISITED",
    "RIGHT": "UNVISITED",
}

junction_anchor = None
simulation_time = 0.0
junction_switch_timer = 0.0

saturation_timer = 0.0
shepherd_form_timer = 0.0
pressure_push_timer = 0.0
flow_establish_timer = 0.0

# Anchor 메시지 버전과 통신 상태
communication_sequence = 0
last_message_signature = None

# 현재 Branch의 임시 Relay 배치 계획
relay_slots = []
relay_deploy_cooldown = 0.0
relay_retract_cooldown = 0.0
relay_retract_clear_timer = 0.0
relay_motion_scale = 1.0

# =========================================================
# 4. SPH / 로봇 파라미터
# =========================================================

ROBOT_COUNT = 220
SPAWN_MODE = "grid"  # "grid" 또는 "random"
ROBOT_RADIUS = 2
GRID_SPACING = 7

SMOOTHING_LENGTH = 28.0
PRESSURE_GAIN = 1650.0
STIFFNESS_EXPONENT = 0.5
VISCOSITY_XI1 = 0.9
VISCOSITY_XI2 = 1.2
# 전체 이동 속도 조절값입니다.
# 1.0 = 기존 속도, 1.5 = 빠름, 2.0 = 현재 권장 속도
MOTION_SPEED_MULTIPLIER = 2.0

# 감쇠가 너무 크면 유도력이 커도 최종 속도가 낮아지므로 기존 2.8에서 낮췄습니다.
DAMPING = 2.3

SAFE_RADIUS = 7.5
REPULSION_GAIN = 260.0

ROUTE_FORCE = 52.0 * MOTION_SPEED_MULTIPLIER
OUTLET_FORCE = 44.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_BACKTRACK_FORCE = 50.0 * MOTION_SPEED_MULTIPLIER
CENTERING_GAIN = 1.2

MAX_SPEED = 78.0 * MOTION_SPEED_MULTIPLIER
MAX_ACCELERATION = 520.0 * MOTION_SPEED_MULTIPLIER
EPSILON = 1e-8

# 고립 로봇이 무리를 따라오도록 하는 보조항
ISOLATION_NEIGHBOR_THRESHOLD = 4
ISOLATION_ROUTE_BOOST = 1.1
LOCAL_COHESION_GAIN = 20.0

# Junction 진입 후 첫 branch 탐색 시작 조건
JUNCTION_ENTRY_COUNT = 18
# Junction Anchor는 일반 탐색 속도 배율과 분리합니다.
# 선발 직후 주차점으로 너무 빨리 달리면 주변 로봇과 직접 링크가 끊어지므로
# 낮은 기본 속도 + 통신 여유거리 기반 자동 감속을 사용합니다.
ANCHOR_MOVE_SPEED = 42.0
ANCHOR_POSITION_TOLERANCE = 2.5
JUNCTION_SWITCH_COUNT = 18
JUNCTION_SWITCH_DWELL_TIME = 0.25
# 최종에는 Anchor까지 역할을 해제하고 220대 전부 시작 통로로 복귀해야 완료합니다.
RETURN_BOTTOM_TARGET_COUNT = ROBOT_COUNT

# Shepherd boundary
# dead-end에 가장 먼저 도착한 8대를 즉시 선발하기 위한 조기 감지 깊이
SHEPHERD_COUNT = 8
EARLY_CAPTURE_DEPTH = 34
SHEPHERD_EDGE_MARGIN = 12
SHEPHERD_FORM_KP = 9.0
SHEPHERD_HOLD_KP = 9.5
SHEPHERD_HOLD_KD = 3.8
SHEPHERD_FORM_SPEED = 110.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_RELEASE_SPEED = 12.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_FORM_TOLERANCE = 3.0
SHEPHERD_FORM_TIMEOUT = 0.9

# Shepherd의 가상 압력
SHEPHERD_PRESSURE_FACTOR = 5.2
VIRTUAL_PRESSURE_RADIUS = 60.0
VIRTUAL_PRESSURE_FORCE = 145.0
PRESSURE_RAMP_TIME = 0.8

# 일반 로봇의 backtracking 흐름이 형성되면 Shepherd를 즉시 해제
FLOW_SPEED_THRESHOLD = 2.5
FLOW_RATIO_THRESHOLD = 0.72
FLOW_AVERAGE_SPEED_THRESHOLD = 3.0
FLOW_ESTABLISH_DWELL_TIME = 0.25
FLOW_MIN_NORMAL_COUNT = 8
FLOW_FALLBACK_TIME = 3.0

# 조기 해제 뒤 전체가 함께 빠져나가는 단계
FLOW_BACKTRACK_FORCE = 46.0 * MOTION_SPEED_MULTIPLIER
BRANCH_CLEAR_LIMIT = 1

# 모든 Branch가 끝난 뒤에만 시작 위치로 최종 복귀합니다.
FINAL_RETURN_DISTANCE_THRESHOLD = 48.0

# =========================================================
# 4-1. 명시적 로봇 통신 / 연결성 파라미터
# =========================================================

# 두 로봇 사이 거리가 이 값 이하일 때만 직접 통신할 수 있습니다.
COMM_RANGE = 46.0

# 이 거리까지는 충분히 안전한 통신 링크로 봅니다.
COMM_SAFE_DISTANCE = 34.0
COMM_BARRIER_START = COMM_RANGE * 0.84

# BFS 통신 트리의 부모 링크가 길어질 때 작용하는 연결 유지력
CONNECTIVITY_GAIN = 4.5
CONNECTIVITY_BARRIER_GAIN = 78.0

# 이미 Anchor 통신망에서 떨어진 로봇이 다시 연결된 무리를 찾는 범위와 힘
COMM_RECOVERY_RANGE = 84.0
COMM_RECOVERY_GAIN = 2.2

# 화면에 통신 트리를 표시할지의 초기값
SHOW_COMM_LINKS_DEFAULT = True

# Junction Anchor 배치 중 통신 보호
# Anchor와 가장 가까운 일반 로봇의 거리가 WARNING을 넘으면 감속하고,
# STOP에 도달하면 주차점 이동을 멈춘 채 군집이 따라올 때까지 기다립니다.
ANCHOR_LINK_WARNING_DISTANCE = COMM_SAFE_DISTANCE * 0.82
ANCHOR_LINK_STOP_DISTANCE = COMM_RANGE * 0.90
ANCHOR_MIN_DIRECT_NEIGHBORS = 1
ANCHOR_READY_DIRECT_NEIGHBORS = 3

# =========================================================
# 4-2. 임시 Relay Chain 파라미터
# =========================================================

# Relay는 통신 한계보다 충분히 짧은 간격으로 Branch 측면에 배치됩니다.
RELAY_SPACING = 30.0
RELAY_DEPLOY_LOOKAHEAD = 12.0
RELAY_SELECTION_RADIUS = 52.0
RELAY_END_CLEARANCE = 24.0
RELAY_LANE_MARGIN = 22.0

# Relay가 slot으로 이동하고 위치를 유지하는 속도/허용 오차
RELAY_MOVE_SPEED = 125.0 * MOTION_SPEED_MULTIPLIER
RELAY_POSITION_TOLERANCE = 2.5
RELAY_DEPLOY_COOLDOWN = 0.10

# Backtracking 군집이 Relay에 도착하면 가장 먼 Relay부터 해제됩니다.
RELAY_RETRACT_RADIUS = 30.0
# 첫 번째 복귀 로봇이 Relay에 닿았다고 바로 회수하지 않습니다.
# 해당 Relay보다 Branch 끝쪽에 남은 이동 로봇이 0대인 상태가 일정 시간 유지돼야 합니다.
RELAY_PASS_MARGIN = 6.0
RELAY_RETRACT_DWELL_TIME = 0.40
RELAY_RETRACT_COOLDOWN = 0.45
RELAY_RELEASE_SPEED = 16.0 * MOTION_SPEED_MULTIPLIER

# Relay 배치가 늦으면 탐색 군집을 감속하여 링크 단절을 예방합니다.
RELAY_FORMING_SPEED_SCALE = 0.30
RELAY_WAIT_SPEED_SCALE = 0.08
RELAY_LINK_WARNING_DISTANCE = COMM_SAFE_DISTANCE
RELAY_LINK_STOP_DISTANCE = COMM_RANGE * 0.94

# 통신 탐색도 같은 Spatial Hashing을 사용합니다.
CELL_SIZE = max(
    SMOOTHING_LENGTH,
    VIRTUAL_PRESSURE_RADIUS,
    COMM_RANGE,
    COMM_RECOVERY_RANGE,
)

# =========================================================
# 5. 맵 마스크 / 영역 판정
# =========================================================

floor_surface = pygame.Surface(
    (SCREEN_WIDTH, SCREEN_HEIGHT),
    pygame.SRCALPHA,
)
floor_surface.fill((0, 0, 0, 0))
pygame.draw.polygon(
    floor_surface,
    (255, 255, 255, 255),
    cross_points,
)
walkable_mask = pygame.mask.from_surface(floor_surface)


def get_robot_region(position):
    point = (int(position.x), int(position.y))

    if junction_rect.collidepoint(point):
        return "JUNCTION"
    if up_rect.collidepoint(point):
        return "UP"
    if left_rect.collidepoint(point):
        return "LEFT"
    if right_rect.collidepoint(point):
        return "RIGHT"
    if bottom_rect.collidepoint(point):
        return "BOTTOM"

    return "OUTSIDE"


def is_region_allowed(position):
    """현재 DFS branch 외의 통로는 가상 밸브처럼 닫습니다.

    Branch 사이에는 시작 위치까지 돌아가지 않고 Parent Junction에서 바로
    다음 Branch로 전환합니다. 모든 Branch가 끝난 뒤에만 Base 쪽으로 복귀합니다.
    """
    region = get_robot_region(position)

    if phase == SimulationPhase.MOVE_TO_JUNCTION:
        return region in {"BOTTOM", "JUNCTION"}

    if phase in {
        SimulationPhase.EXPLORE_BRANCH,
        SimulationPhase.FORM_SHEPHERD_BOUNDARY,
        SimulationPhase.PRESSURE_PUSH,
        SimulationPhase.FLOW_BACKTRACK,
        SimulationPhase.JUNCTION_SWITCH,
    }:
        return region in {"BOTTOM", "JUNCTION", active_branch}

    if phase in {SimulationPhase.RETURN_TO_BASE, SimulationPhase.DONE}:
        return region in {"BOTTOM", "JUNCTION"}

    return region != "OUTSIDE"


def is_walkable(position, radius):
    x = int(round(position.x))
    y = int(round(position.y))
    diagonal = int(round(radius / math.sqrt(2.0)))

    test_points = [
        (x, y),
        (x + radius, y),
        (x - radius, y),
        (x, y + radius),
        (x, y - radius),
        (x + diagonal, y + diagonal),
        (x + diagonal, y - diagonal),
        (x - diagonal, y + diagonal),
        (x - diagonal, y - diagonal),
    ]

    for px, py in test_points:
        if not (0 <= px < SCREEN_WIDTH and 0 <= py < SCREEN_HEIGHT):
            return False
        if walkable_mask.get_at((px, py)) == 0:
            return False

    return is_region_allowed(position)

# =========================================================
# 6. 공통 함수
# =========================================================


def limit_vector(vector, maximum_length):
    if vector.length_squared() > maximum_length * maximum_length:
        vector.scale_to_length(maximum_length)
    return vector


def spiky_kernel(distance, h):
    if distance < 0.0 or distance > h:
        return 0.0

    q = 1.0 - distance / h
    return 10.0 / (math.pi * h * h) * q**3


def spiky_gradient(r_ij, h):
    distance = r_ij.length()

    if distance <= EPSILON or distance > h:
        return pygame.Vector2(0.0, 0.0)

    q = 1.0 - distance / h
    gradient_magnitude = -30.0 / (math.pi * h**3) * q**2
    return gradient_magnitude * (r_ij / distance)


def interpolate_color(color_a, color_b, ratio):
    ratio = max(0.0, min(1.0, ratio))
    return tuple(
        int(color_a[i] + (color_b[i] - color_a[i]) * ratio)
        for i in range(3)
    )


def density_to_color(density, color_reference_density):
    ratio = density / max(color_reference_density, EPSILON)

    if ratio <= 1.0:
        return interpolate_color(
            (151, 190, 226),
            (142, 204, 190),
            ratio,
        )

    high_ratio = min((ratio - 1.0) / 0.75, 1.0)
    return interpolate_color(
        (142, 204, 190),
        (242, 187, 126),
        high_ratio,
    )


def normalized_direction_toward(source, target):
    direction = target - source
    if direction.length_squared() > EPSILON:
        return direction.normalize()
    return pygame.Vector2(0.0, 0.0)


def get_bottom_hold_point():
    return pygame.Vector2(
        center_x,
        center_y + half_width + normal_length - 18,
    )


def get_branch_tip_target(branch):
    if branch == "UP":
        return pygame.Vector2(
            center_x,
            center_y - half_width - normal_length + 18,
        )
    if branch == "LEFT":
        return pygame.Vector2(
            center_x - half_width - normal_length + 18,
            center_y,
        )
    if branch == "RIGHT":
        return pygame.Vector2(
            center_x + half_width + right_length - 18,
            center_y,
        )
    return pygame.Vector2(center_x, center_y)


def get_backtrack_direction(branch):
    """Dead-end에서 Junction을 향하는 단위벡터."""
    return -BRANCH_DIRECTIONS[branch]


def branch_progress_position(position, branch):
    """값이 클수록 dead-end에 가깝습니다."""
    if branch == "UP":
        return -position.y
    if branch == "LEFT":
        return -position.x
    if branch == "RIGHT":
        return position.x
    return 0.0


def branch_progress(robot, branch):
    return branch_progress_position(robot.position, branch)

# =========================================================
# 7. Robot 클래스
# =========================================================


class Robot:
    def __init__(self, x, y, robot_id):
        self.robot_id = robot_id
        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(0.0, 0.0)
        self.acceleration = pygame.Vector2(0.0, 0.0)
        self.radius = ROBOT_RADIUS

        self.density = 0.0
        self.pressure = 0.0

        self.role = "NORMAL"
        self.shepherd_anchor = None

        # 임시 Relay Chain 상태
        self.relay_anchor = None
        self.relay_index = -1

        # Junction Anchor 상태
        self.anchor_position = None
        self.local_branch_states = None
        self.selected_branch = None
        self.parent_branch = "BOTTOM"

        # Junction Anchor 선발용 진입 기록
        self.anchor_region_entry_time = None
        self.was_in_anchor_region = anchor_election_rect.collidepoint(x, y)

        # -------------------------------------------------
        # 명시적 국소 통신 상태
        # -------------------------------------------------
        self.comm_neighbors = []
        self.connected_to_anchor = False
        self.comm_hop = -1
        self.comm_parent = None

        # Anchor에서 다중 홉으로 수신한 메시지
        self.received_branch = None
        self.received_command = None
        self.received_sequence = -1

    def update(self, dt):
        # -------------------------------------------------
        # Junction Anchor는 선발 후 Junction 가장자리 주차점으로 이동한 뒤 고정됩니다.
        # -------------------------------------------------
        if self.role == "ANCHOR" and self.anchor_position is not None:
            position_error = self.anchor_position - self.position

            if (
                position_error.length_squared()
                > ANCHOR_POSITION_TOLERANCE**2
            ):
                # 선발 직후 Anchor가 군집에서 이탈하지 않도록 현재 직접 통신
                # 이웃과의 거리로 주차 이동 속도를 자동 제한합니다.
                motion_scale = get_anchor_deployment_motion_scale(self)
                maximum_step = ANCHOR_MOVE_SPEED * motion_scale * dt

                if maximum_step > 0.0:
                    if position_error.length() <= maximum_step:
                        next_position = self.anchor_position.copy()
                    else:
                        next_position = (
                            self.position
                            + position_error.normalize() * maximum_step
                        )

                    if is_walkable(next_position, self.radius):
                        self.position = next_position
            else:
                self.position = self.anchor_position.copy()

            self.velocity.update(0.0, 0.0)
            self.acceleration.update(0.0, 0.0)
            return

        # -------------------------------------------------
        # Relay는 Branch 측면 slot으로 이동한 뒤 고정 통신 노드가 됩니다.
        # Backtracking 시 역할이 해제되면 다시 NORMAL로 군집에 합류합니다.
        # -------------------------------------------------
        if self.role == "RELAY" and self.relay_anchor is not None:
            position_error = self.relay_anchor - self.position

            if position_error.length_squared() > RELAY_POSITION_TOLERANCE**2:
                maximum_step = RELAY_MOVE_SPEED * dt

                if position_error.length() <= maximum_step:
                    next_position = self.relay_anchor.copy()
                else:
                    next_position = (
                        self.position
                        + position_error.normalize() * maximum_step
                    )

                if is_walkable(next_position, self.radius):
                    self.position = next_position
            else:
                self.position = self.relay_anchor.copy()

            self.velocity.update(0.0, 0.0)
            self.acceleration.update(0.0, 0.0)
            return

        # -------------------------------------------------
        # Shepherd는 경계 형성 중 anchor로 빠르게 정렬하고,
        # PRESSURE_PUSH 동안에는 고정 경계 입자로 유지합니다.
        # 일반 로봇은 Shepherd의 압력을 받지만 Shepherd 자신은
        # SPH 반작용으로 밀려나지 않습니다.
        # -------------------------------------------------
        if self.role == "SHEPHERD" and self.shepherd_anchor is not None:
            if phase == SimulationPhase.FORM_SHEPHERD_BOUNDARY:
                position_error = self.shepherd_anchor - self.position

                if position_error.length_squared() > EPSILON:
                    maximum_step = SHEPHERD_FORM_SPEED * dt

                    if position_error.length() <= maximum_step:
                        next_position = self.shepherd_anchor.copy()
                    else:
                        next_position = (
                            self.position
                            + position_error.normalize() * maximum_step
                        )

                    if is_walkable(next_position, self.radius):
                        self.position = next_position

                self.velocity.update(0.0, 0.0)
                self.acceleration.update(0.0, 0.0)
                return

            if phase == SimulationPhase.PRESSURE_PUSH:
                self.position = self.shepherd_anchor.copy()
                self.velocity.update(0.0, 0.0)
                self.acceleration.update(0.0, 0.0)
                return

        self.velocity += self.acceleration * dt
        limit_vector(self.velocity, MAX_SPEED)

        x_position = pygame.Vector2(
            self.position.x + self.velocity.x * dt,
            self.position.y,
        )
        if is_walkable(x_position, self.radius):
            self.position.x = x_position.x
        else:
            self.velocity.x = 0.0

        y_position = pygame.Vector2(
            self.position.x,
            self.position.y + self.velocity.y * dt,
        )
        if is_walkable(y_position, self.radius):
            self.position.y = y_position.y
        else:
            self.velocity.y = 0.0

        self.acceleration.update(0.0, 0.0)

    def draw(self, surface, color_reference_density, show_density_color):
        x = round(self.position.x)
        y = round(self.position.y)

        if self.role == "ANCHOR":
            color = ANCHOR_COLOR
        elif self.role == "RELAY":
            color = RELAY_COLOR
        elif self.role == "SHEPHERD":
            color = SHEPHERD_COLOR
        elif show_density_color:
            color = density_to_color(
                self.density,
                color_reference_density,
            )
        else:
            color = ROBOT_BASE_COLOR

        marker_size = self.radius * 2 + 1
        marker_rect = pygame.Rect(
            x - self.radius,
            y - self.radius,
            marker_size,
            marker_size,
        )

        pygame.draw.rect(
            surface,
            color,
            marker_rect,
            border_radius=self.radius,
        )

        if self.role == "RELAY":
            pygame.draw.circle(
                surface,
                RELAY_COLOR,
                (x, y),
                self.radius + 3,
                width=1,
            )

        # Anchor가 존재한 뒤 통신망에서 끊긴 로봇은 붉은 테두리로 표시합니다.
        if (
            junction_anchor is not None
            and self.role != "ANCHOR"
            and not self.connected_to_anchor
        ):
            pygame.draw.rect(
                surface,
                DISCONNECTED_COLOR,
                marker_rect.inflate(2, 2),
                width=1,
                border_radius=self.radius + 1,
            )

# =========================================================
# 8. 로봇 생성
# =========================================================


def create_grid_robots(robot_count):
    robots = []

    entrance_left = center_x - half_width + ROBOT_RADIUS + 4
    entrance_right = center_x + half_width - ROBOT_RADIUS - 4
    entrance_top = center_y + half_width + 12
    entrance_bottom = (
        center_y + half_width + normal_length - ROBOT_RADIUS - 7
    )

    available_width = entrance_right - entrance_left
    robots_per_row = max(1, int(available_width // GRID_SPACING) + 1)

    for robot_id in range(robot_count):
        row = robot_id // robots_per_row
        column = robot_id % robots_per_row

        x = entrance_left + column * GRID_SPACING
        y = entrance_bottom - row * GRID_SPACING

        if y < entrance_top:
            print(
                f"경고: 현재 입구에는 {len(robots)}대만 "
                "격자로 배치할 수 있습니다."
            )
            break

        robots.append(Robot(x, y, robot_id))

    return robots


def create_random_robots(robot_count):
    robots = []
    minimum_distance = ROBOT_RADIUS * 2 + 1

    entrance_left = center_x - half_width + ROBOT_RADIUS + 4
    entrance_right = center_x + half_width - ROBOT_RADIUS - 4
    entrance_top = center_y + half_width + 12
    entrance_bottom = (
        center_y + half_width + normal_length - ROBOT_RADIUS - 7
    )

    attempts = 0
    max_attempts = robot_count * 400

    while len(robots) < robot_count and attempts < max_attempts:
        attempts += 1
        candidate = pygame.Vector2(
            random.uniform(entrance_left, entrance_right),
            random.uniform(entrance_top, entrance_bottom),
        )

        if all(
            candidate.distance_to(robot.position) >= minimum_distance
            for robot in robots
        ):
            robots.append(Robot(candidate.x, candidate.y, len(robots)))

    if len(robots) < robot_count:
        print(
            f"경고: 요청한 {robot_count}대 중 "
            f"{len(robots)}대만 배치했습니다."
        )

    return robots

# =========================================================
# 9. Spatial Hashing
# =========================================================


def cell_key(position):
    return (
        int(position.x // CELL_SIZE),
        int(position.y // CELL_SIZE),
    )


def build_spatial_grid(robots):
    grid = {}
    for robot in robots:
        grid.setdefault(cell_key(robot.position), []).append(robot)
    return grid


def iter_neighbor_candidates(robot, grid):
    cx, cy = cell_key(robot.position)

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for candidate in grid.get((cx + dx, cy + dy), []):
                yield candidate


# =========================================================
# 9-1. 명시적 국소 통신 / 다중 홉 메시지 전달
# =========================================================


def update_communication_neighbors(robots, grid):
    """COMM_RANGE 안에 있는 로봇만 직접 통신 이웃으로 등록합니다."""
    comm_range_squared = COMM_RANGE**2

    for robot in robots:
        robot.comm_neighbors = []

        for other in iter_neighbor_candidates(robot, grid):
            if robot is other:
                continue

            if (
                robot.position.distance_squared_to(other.position)
                <= comm_range_squared
            ):
                robot.comm_neighbors.append(other)


def get_anchor_deployment_motion_scale(anchor):
    """Anchor 주차 이동 중 직접 통신 링크가 끊어지지 않도록 속도를 제한합니다.

    반환값 1.0은 정상 이동, 0.0은 통신 이웃이 따라올 때까지 정지를 뜻합니다.
    MOVE_TO_JUNCTION 이후 주차가 완료된 Anchor에는 이 값이 사용되지 않습니다.
    """
    if anchor is None or anchor.role != "ANCHOR":
        return 0.0

    direct_neighbors = [
        neighbor
        for neighbor in anchor.comm_neighbors
        if neighbor.role != "ANCHOR"
    ]

    if len(direct_neighbors) < ANCHOR_MIN_DIRECT_NEIGHBORS:
        return 0.0

    nearest_distance = min(
        anchor.position.distance_to(neighbor.position)
        for neighbor in direct_neighbors
    )

    if nearest_distance <= ANCHOR_LINK_WARNING_DISTANCE:
        return 1.0

    if nearest_distance >= ANCHOR_LINK_STOP_DISTANCE:
        return 0.0

    # WARNING~STOP 구간에서 선형 감속합니다.
    return max(
        0.0,
        min(
            1.0,
            (ANCHOR_LINK_STOP_DISTANCE - nearest_distance)
            / (ANCHOR_LINK_STOP_DISTANCE - ANCHOR_LINK_WARNING_DISTANCE),
        ),
    )


def anchor_deployment_ready(anchor, robots):
    """Anchor가 주차점에 도착했고 안전한 직접 이웃을 확보했는지 검사합니다."""
    if anchor is None or anchor.anchor_position is None:
        return False

    if (
        anchor.position.distance_to(anchor.anchor_position)
        > ANCHOR_POSITION_TOLERANCE
    ):
        return False

    safe_neighbors = sum(
        robot is not anchor
        and robot.role != "ANCHOR"
        and anchor.position.distance_to(robot.position)
        <= COMM_SAFE_DISTANCE
        for robot in robots
    )

    return safe_neighbors >= ANCHOR_READY_DIRECT_NEIGHBORS


def get_anchor_message(anchor):
    """현재 Anchor가 전파하는 DFS 명령을 반환합니다."""
    if anchor is None:
        return None, None

    return anchor.selected_branch, phase.name


def propagate_anchor_message(robots, anchor):
    """Anchor에서 시작해 통신 그래프를 BFS로 순회합니다.

    직접 통신 범위 밖에 있는 로봇도 중간 로봇을 통하는 경로가 있으면
    selected_branch와 현재 제어 명령을 수신합니다.
    """
    global communication_sequence
    global last_message_signature

    for robot in robots:
        robot.connected_to_anchor = False
        robot.comm_hop = -1
        robot.comm_parent = None
        robot.received_branch = None
        robot.received_command = None
        robot.received_sequence = -1

    if anchor is None:
        last_message_signature = None
        return

    selected_branch, command = get_anchor_message(anchor)
    signature = (selected_branch, command)

    if signature != last_message_signature:
        communication_sequence += 1
        last_message_signature = signature
        print(
            "[Communication] new message: "
            f"seq={communication_sequence}, "
            f"command={command}, branch={selected_branch}"
        )

    anchor.connected_to_anchor = True
    anchor.comm_hop = 0
    anchor.received_branch = selected_branch
    anchor.received_command = command
    anchor.received_sequence = communication_sequence

    queue = deque([anchor])

    while queue:
        current = queue.popleft()

        for neighbor in current.comm_neighbors:
            if neighbor.connected_to_anchor:
                continue

            neighbor.connected_to_anchor = True
            neighbor.comm_hop = current.comm_hop + 1
            neighbor.comm_parent = current
            neighbor.received_branch = selected_branch
            neighbor.received_command = command
            neighbor.received_sequence = communication_sequence
            queue.append(neighbor)


def update_communication_system(robots, grid):
    update_communication_neighbors(robots, grid)
    propagate_anchor_message(robots, junction_anchor)


def find_nearest_connected_robot(robot, grid):
    """통신이 끊긴 로봇 주변에서 Anchor와 연결된 가장 가까운 로봇을 찾습니다."""
    recovery_squared = COMM_RECOVERY_RANGE**2
    nearest = None
    nearest_distance_squared = recovery_squared

    for candidate in iter_neighbor_candidates(robot, grid):
        if candidate is robot or not candidate.connected_to_anchor:
            continue

        distance_squared = robot.position.distance_squared_to(
            candidate.position
        )

        if distance_squared < nearest_distance_squared:
            nearest_distance_squared = distance_squared
            nearest = candidate

    return nearest


def compute_connectivity_force(robot, grid):
    """현재 BFS 통신 트리의 부모 링크를 통신 범위 안에 유지합니다.

    이 항은 명령 전달 자체와 별개의 연결성 유지용 안전 보조항입니다.
    """
    if junction_anchor is None or robot.role == "ANCHOR":
        return pygame.Vector2(0.0, 0.0)

    force = pygame.Vector2(0.0, 0.0)

    if robot.connected_to_anchor and robot.comm_parent is not None:
        delta = robot.comm_parent.position - robot.position
        distance = delta.length()

        if distance > EPSILON:
            direction = delta / distance

            if distance > COMM_SAFE_DISTANCE:
                force += (
                    CONNECTIVITY_GAIN
                    * (distance - COMM_SAFE_DISTANCE)
                    * direction
                )

            if distance > COMM_BARRIER_START:
                denominator = max(
                    COMM_RANGE - COMM_BARRIER_START,
                    EPSILON,
                )
                ratio = min(
                    1.0,
                    (distance - COMM_BARRIER_START) / denominator,
                )
                force += (
                    CONNECTIVITY_BARRIER_GAIN
                    * ratio**2
                    * direction
                )

    elif not robot.connected_to_anchor:
        # 이미 끊어진 경우 가장 가까운 연결 로봇 쪽으로 이동해 재접속을 시도합니다.
        recovery_target = find_nearest_connected_robot(robot, grid)

        if recovery_target is not None:
            delta = recovery_target.position - robot.position
            distance = delta.length()

            if distance > EPSILON:
                force += (
                    COMM_RECOVERY_GAIN
                    * min(distance, COMM_RECOVERY_RANGE)
                    * (delta / distance)
                )

    return force


def get_communication_stats(robots):
    if junction_anchor is None:
        return {
            "connected": 0,
            "disconnected": len(robots),
            "max_hop": 0,
            "direct_anchor_neighbors": 0,
            "minimum_margin": 0.0,
        }

    connected = [robot for robot in robots if robot.connected_to_anchor]
    disconnected_count = len(robots) - len(connected)
    max_hop = max((robot.comm_hop for robot in connected), default=0)

    margins = []
    for robot in connected:
        if robot.comm_parent is None:
            continue
        distance = robot.position.distance_to(robot.comm_parent.position)
        margins.append(COMM_RANGE - distance)

    return {
        "connected": len(connected),
        "disconnected": disconnected_count,
        "max_hop": max_hop,
        "direct_anchor_neighbors": len(junction_anchor.comm_neighbors),
        "minimum_margin": min(margins) if margins else COMM_RANGE,
    }


def draw_communication_links(surface, robots):
    """모든 완전 그래프가 아니라 Anchor BFS 트리 링크만 그립니다."""
    if junction_anchor is None:
        return

    for robot in robots:
        parent = robot.comm_parent
        if not robot.connected_to_anchor or parent is None:
            continue

        distance = robot.position.distance_to(parent.position)

        if distance <= COMM_SAFE_DISTANCE:
            color = COMM_LINK_SAFE_COLOR
        elif distance <= COMM_BARRIER_START:
            color = COMM_LINK_WARNING_COLOR
        else:
            color = COMM_LINK_DANGER_COLOR

        pygame.draw.line(
            surface,
            color,
            (round(robot.position.x), round(robot.position.y)),
            (round(parent.position.x), round(parent.position.y)),
            width=1,
        )


# =========================================================
# 9-2. 임시 Relay Chain 배치 / 회수
# =========================================================


def get_relay_path_endpoint(branch):
    """통행을 막지 않도록 Branch 한쪽 측면의 Relay lane 끝점을 반환합니다."""
    if branch == "UP":
        return pygame.Vector2(
            center_x - half_width + RELAY_LANE_MARGIN,
            center_y - half_width - normal_length + RELAY_END_CLEARANCE,
        )

    if branch == "LEFT":
        return pygame.Vector2(
            center_x - half_width - normal_length + RELAY_END_CLEARANCE,
            center_y - half_width + RELAY_LANE_MARGIN,
        )

    if branch == "RIGHT":
        return pygame.Vector2(
            center_x + half_width + right_length - RELAY_END_CLEARANCE,
            center_y - half_width + RELAY_LANE_MARGIN,
        )

    return ANCHOR_PARK_POSITION.copy()


def initialize_relay_plan(branch):
    """Anchor에서 Branch 끝 방향으로 RELAY_SPACING 간격의 slot을 생성합니다."""
    global relay_slots
    global relay_deploy_cooldown
    global relay_retract_cooldown
    global relay_retract_clear_timer
    global relay_motion_scale

    start = ANCHOR_PARK_POSITION.copy()
    end = get_relay_path_endpoint(branch)
    path_vector = end - start
    path_length = path_vector.length()

    relay_slots = []
    relay_deploy_cooldown = 0.0
    relay_retract_cooldown = 0.0
    relay_retract_clear_timer = 0.0
    relay_motion_scale = 1.0

    if path_length <= EPSILON:
        return

    direction = path_vector / path_length
    distance = RELAY_SPACING
    slot_index = 0

    while distance <= path_length:
        relay_slots.append(
            {
                "index": slot_index,
                "position": start + direction * distance,
                "path_distance": distance,
            }
        )
        slot_index += 1
        distance += RELAY_SPACING

    print(
        f"[Relay] plan initialized: branch={branch}, "
        f"slots={len(relay_slots)}, spacing={RELAY_SPACING:.1f}px"
    )


def relay_path_progress(position, branch):
    """Anchor에서 Branch relay lane 끝 방향으로 투영한 경로 진행거리입니다."""
    start = ANCHOR_PARK_POSITION
    end = get_relay_path_endpoint(branch)
    path_vector = end - start
    path_length = path_vector.length()

    if path_length <= EPSILON:
        return 0.0

    direction = path_vector / path_length
    projected = (position - start).dot(direction)
    return max(0.0, min(path_length, projected))


def get_relays(robots):
    return [robot for robot in robots if robot.role == "RELAY"]


def get_active_branch_relays(robots):
    return sorted(
        [
            robot
            for robot in robots
            if robot.role == "RELAY" and robot.relay_index >= 0
        ],
        key=lambda robot: robot.relay_index,
    )


def get_deployed_relay_indices(robots):
    return {
        robot.relay_index
        for robot in robots
        if robot.role == "RELAY" and robot.relay_index >= 0
    }


def relay_at_slot_is_settled(robot):
    return (
        robot.role == "RELAY"
        and robot.relay_anchor is not None
        and robot.position.distance_to(robot.relay_anchor)
        <= RELAY_POSITION_TOLERANCE
    )


def get_exploration_front_progress(robots, branch):
    candidates = [
        robot
        for robot in robots
        if robot.role in {"NORMAL", "SHEPHERD"}
        and get_robot_region(robot.position) in {"JUNCTION", branch}
    ]

    if not candidates:
        return 0.0

    return max(
        relay_path_progress(robot.position, branch)
        for robot in candidates
    )


def select_relay_candidate(robots, slot):
    """slot 부근의 연결된 NORMAL 로봇을 Relay로 전환합니다."""
    candidates = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and robot.connected_to_anchor
        and get_robot_region(robot.position) in {"JUNCTION", active_branch}
        and robot.position.distance_to(slot["position"])
        <= RELAY_SELECTION_RADIUS
    ]

    if not candidates:
        return None

    # slot에 가깝고 속도가 낮은 로봇을 우선 선택합니다.
    return min(
        candidates,
        key=lambda robot: (
            robot.position.distance_squared_to(slot["position"]),
            robot.velocity.length_squared(),
            robot.robot_id,
        ),
    )


def deploy_relay_for_slot(robots, slot):
    candidate = select_relay_candidate(robots, slot)

    if candidate is None:
        return False

    candidate.role = "RELAY"
    candidate.relay_anchor = slot["position"].copy()
    candidate.relay_index = slot["index"]
    candidate.velocity.update(0.0, 0.0)
    candidate.acceleration.update(0.0, 0.0)

    print(
        f"[Relay] deployed: robot={candidate.robot_id}, "
        f"index={candidate.relay_index}, "
        f"slot=({candidate.relay_anchor.x:.1f}, {candidate.relay_anchor.y:.1f})"
    )
    return True


def get_next_undeployed_slot(robots):
    deployed_indices = get_deployed_relay_indices(robots)

    for slot in relay_slots:
        if slot["index"] not in deployed_indices:
            return slot

    return None


def relay_plan_complete(robots):
    if not relay_slots:
        return True

    relays_by_index = {
        robot.relay_index: robot
        for robot in get_active_branch_relays(robots)
    }

    return all(
        slot["index"] in relays_by_index
        and relay_at_slot_is_settled(relays_by_index[slot["index"]])
        for slot in relay_slots
    )


def compute_relay_front_link_distance(robots):
    """가장 먼 Relay와 그 앞쪽 탐색 군집 사이의 최소 거리를 계산합니다."""
    relays = get_active_branch_relays(robots)
    chain_end = relays[-1] if relays else junction_anchor

    if chain_end is None:
        return 0.0

    chain_progress = relay_path_progress(chain_end.position, active_branch)
    front_robots = [
        robot
        for robot in robots
        if robot.role in {"NORMAL", "SHEPHERD"}
        and get_robot_region(robot.position) in {"JUNCTION", active_branch}
        and relay_path_progress(robot.position, active_branch)
        >= chain_progress - 2.0
    ]

    if not front_robots:
        return 0.0

    return min(
        chain_end.position.distance_to(robot.position)
        for robot in front_robots
    )


def update_relay_deployment(robots, dt):
    """군집이 전진할 때 필요한 slot에 Relay를 순차적으로 남깁니다."""
    global relay_deploy_cooldown
    global relay_motion_scale

    relay_deploy_cooldown = max(0.0, relay_deploy_cooldown - dt)
    relay_motion_scale = 1.0

    if phase != SimulationPhase.EXPLORE_BRANCH or junction_anchor is None:
        return

    front_progress = get_exploration_front_progress(robots, active_branch)
    next_slot = get_next_undeployed_slot(robots)

    if next_slot is not None:
        slot_due = (
            front_progress
            >= next_slot["path_distance"] - RELAY_DEPLOY_LOOKAHEAD
        )

        if slot_due and relay_deploy_cooldown <= 0.0:
            if deploy_relay_for_slot(robots, next_slot):
                relay_deploy_cooldown = RELAY_DEPLOY_COOLDOWN
            else:
                relay_motion_scale = min(
                    relay_motion_scale,
                    RELAY_WAIT_SPEED_SCALE,
                )

        # Relay가 필요한 위치까지 군집이 도착했는데 배치되지 않았다면 감속합니다.
        if front_progress >= next_slot["path_distance"] - 2.0:
            relay_motion_scale = min(
                relay_motion_scale,
                RELAY_WAIT_SPEED_SCALE,
            )

    # 새 Relay가 slot으로 이동 중이면 군집을 잠시 감속합니다.
    unsettled_relays = [
        relay
        for relay in get_active_branch_relays(robots)
        if not relay_at_slot_is_settled(relay)
    ]
    if unsettled_relays:
        relay_motion_scale = min(
            relay_motion_scale,
            RELAY_FORMING_SPEED_SCALE,
        )

    # 마지막 Relay와 군집 사이 링크가 늘어날수록 선제적으로 속도를 줄입니다.
    front_link_distance = compute_relay_front_link_distance(robots)

    if front_link_distance > RELAY_LINK_WARNING_DISTANCE:
        denominator = max(
            RELAY_LINK_STOP_DISTANCE - RELAY_LINK_WARNING_DISTANCE,
            EPSILON,
        )
        scale = (
            RELAY_LINK_STOP_DISTANCE - front_link_distance
        ) / denominator
        relay_motion_scale = min(
            relay_motion_scale,
            max(0.0, min(1.0, scale)),
        )

    if front_link_distance >= RELAY_LINK_STOP_DISTANCE:
        relay_motion_scale = 0.0


def release_relay_into_backtracking(robot):
    backtrack_direction = get_backtrack_direction(active_branch)
    released_index = robot.relay_index

    robot.role = "NORMAL"
    robot.relay_anchor = None
    robot.relay_index = -1
    robot.velocity = backtrack_direction * RELAY_RELEASE_SPEED
    robot.acceleration.update(0.0, 0.0)

    print(
        f"[Relay] retracted into flow: robot={robot.robot_id}, "
        f"index={released_index}"
    )


def update_relay_retraction(robots, dt):
    """복귀 군집의 마지막 로봇이 Relay를 지난 뒤에만 역순으로 회수합니다.

    기존에는 첫 번째 로봇 한 대가 먼 Relay 근처에 도착하는 즉시 Relay를
    해제했기 때문에 Branch 끝에 남아 있던 로봇들의 통신 경로가 끊어질 수
    있었습니다. 이제는 다음 조건을 모두 만족해야 회수합니다.

    1. 현재 가장 먼 Relay보다 Branch 끝쪽에 남은 이동 로봇이 없음
    2. Branch/Junction의 이동 로봇들이 Anchor 통신망에 연결되어 있음
    3. 위 상태가 RELAY_RETRACT_DWELL_TIME 동안 연속 유지됨
    """
    global relay_retract_cooldown
    global relay_retract_clear_timer

    relay_retract_cooldown = max(0.0, relay_retract_cooldown - dt)

    if phase != SimulationPhase.FLOW_BACKTRACK:
        relay_retract_clear_timer = 0.0
        return

    relays = get_active_branch_relays(robots)
    if not relays:
        relay_retract_clear_timer = 0.0
        return

    farthest_relay = relays[-1]
    relay_progress = relay_path_progress(
        farthest_relay.position,
        active_branch,
    )

    # Relay보다 Branch 끝쪽에 남은 NORMAL/SHEPHERD가 한 대라도 있으면
    # 아직 Relay를 회수하면 안 됩니다.
    mobile_robots = [
        robot
        for robot in robots
        if robot.role in {"NORMAL", "SHEPHERD"}
        and get_robot_region(robot.position) in {active_branch, "JUNCTION"}
    ]

    robots_still_beyond_relay = [
        robot
        for robot in mobile_robots
        if get_robot_region(robot.position) == active_branch
        and relay_path_progress(robot.position, active_branch)
        > relay_progress + RELAY_PASS_MARGIN
    ]

    # 회수 직전까지 현재 이동 집단이 Anchor와 연결되어 있는지도 확인합니다.
    all_mobile_connected = all(
        robot.connected_to_anchor
        for robot in mobile_robots
    )

    tail_has_passed = len(robots_still_beyond_relay) == 0

    if tail_has_passed and all_mobile_connected:
        relay_retract_clear_timer += dt
    else:
        relay_retract_clear_timer = 0.0

    if (
        relay_retract_clear_timer >= RELAY_RETRACT_DWELL_TIME
        and relay_retract_cooldown <= 0.0
    ):
        release_relay_into_backtracking(farthest_relay)
        relay_retract_cooldown = RELAY_RETRACT_COOLDOWN
        relay_retract_clear_timer = 0.0

def draw_relay_plan(surface, robots):
    """계획 slot과 현재 Relay chain을 시각화합니다."""
    for slot in relay_slots:
        pygame.draw.circle(
            surface,
            RELAY_SLOT_COLOR,
            (
                round(slot["position"].x),
                round(slot["position"].y),
            ),
            3,
            width=1,
        )

    chain_nodes = []
    if junction_anchor is not None:
        chain_nodes.append(junction_anchor)
    chain_nodes.extend(get_active_branch_relays(robots))

    for first, second in zip(chain_nodes, chain_nodes[1:]):
        pygame.draw.line(
            surface,
            RELAY_COLOR,
            (
                round(first.position.x),
                round(first.position.y),
            ),
            (
                round(second.position.x),
                round(second.position.y),
            ),
            width=2,
        )


# =========================================================
# 10. Junction Anchor 선발 / DFS 관리
# =========================================================


def update_anchor_entry_records(robots, current_time):
    """Anchor Region 경계를 처음 통과한 시간을 프레임 단위로 기록합니다."""
    for robot in robots:
        inside = anchor_election_rect.collidepoint(
            robot.position.x,
            robot.position.y,
        )

        if (
            inside
            and not robot.was_in_anchor_region
            and robot.anchor_region_entry_time is None
            and robot.role == "NORMAL"
        ):
            robot.anchor_region_entry_time = current_time

        robot.was_in_anchor_region = inside


def elect_junction_anchor(robots):
    """Junction Anchor Region에 가장 먼저 진입한 일반 로봇을 선발합니다."""
    global junction_anchor

    if junction_anchor is not None:
        return junction_anchor

    candidates = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and robot.anchor_region_entry_time is not None
    ]

    if not candidates:
        return None

    junction_anchor = min(
        candidates,
        key=lambda robot: (
            robot.anchor_region_entry_time,
            robot.robot_id,
        ),
    )

    junction_anchor.role = "ANCHOR"
    junction_anchor.anchor_position = ANCHOR_PARK_POSITION.copy()
    junction_anchor.local_branch_states = branch_states
    junction_anchor.selected_branch = None

    print(
        "[Anchor] elected: "
        f"robot={junction_anchor.robot_id}, "
        f"entry_time={junction_anchor.anchor_region_entry_time:.3f}"
    )

    return junction_anchor


def anchor_select_next_branch(anchor):
    """Anchor가 DFS 순서에서 첫 번째 UNVISITED Branch를 선택합니다."""
    global active_branch

    if anchor is None or anchor.local_branch_states is None:
        return None

    for branch in DFS_ORDER:
        if anchor.local_branch_states[branch] == "UNVISITED":
            anchor.local_branch_states[branch] = "ACTIVE"
            anchor.selected_branch = branch
            active_branch = branch
            initialize_relay_plan(branch)

            print(f"[Anchor] selected branch: {branch}")
            return branch

    anchor.selected_branch = None
    return None


def anchor_complete_active_branch(anchor, branch):
    """Backtracking이 Junction까지 완료되면 현재 Branch를 VISITED로 기록합니다."""
    if anchor is None or anchor.local_branch_states is None:
        return

    anchor.local_branch_states[branch] = "VISITED"
    anchor.selected_branch = None
    print(f"[Anchor] branch completed: {branch}")


def release_anchor_for_final_return(anchor):
    """모든 Branch 완료 후 Junction Anchor를 일반 로봇으로 복귀시킵니다."""
    if anchor is None:
        return

    anchor.role = "NORMAL"
    anchor.anchor_position = None
    anchor.selected_branch = None
    anchor.relay_anchor = None
    anchor.relay_index = -1
    anchor.velocity = pygame.Vector2(0.0, 0.0)
    anchor.acceleration.update(0.0, 0.0)

    print(
        f"[Anchor] robot={anchor.robot_id} released; "
        "returning to Base with the swarm"
    )


def begin_final_return(anchor):
    """Relay plan을 종료하고 Anchor를 포함한 전체 군집의 최종 복귀를 시작합니다."""
    global phase
    global relay_slots
    global relay_motion_scale

    relay_slots = []
    relay_motion_scale = 1.0
    release_anchor_for_final_return(anchor)
    phase = SimulationPhase.RETURN_TO_BASE
    print("[DFS] 모든 Branch 완료, Anchor 포함 전체 로봇 Base 복귀")


# =========================================================
# 11. Shepherd 선택 / 경계 생성
# =========================================================


def build_shepherd_slots(branch, count):
    """Dead-end 폭을 가로지르는 균일한 Shepherd anchor line."""
    slots = []
    usable_half_width = half_width - SHEPHERD_EDGE_MARGIN

    if count <= 1:
        lateral_values = [0.0]
    else:
        lateral_values = [
            -usable_half_width
            + 2.0 * usable_half_width * index / (count - 1)
            for index in range(count)
        ]

    if branch == "UP":
        y = center_y - half_width - normal_length + 14
        slots = [
            pygame.Vector2(center_x + lateral, y)
            for lateral in lateral_values
        ]

    elif branch == "LEFT":
        x = center_x - half_width - normal_length + 14
        slots = [
            pygame.Vector2(x, center_y + lateral)
            for lateral in lateral_values
        ]

    elif branch == "RIGHT":
        x = center_x + half_width + right_length - 14
        slots = [
            pygame.Vector2(x, center_y + lateral)
            for lateral in lateral_values
        ]

    return slots


def reset_robot_roles(robots):
    for robot in robots:
        if robot.role == "SHEPHERD":
            robot.role = "NORMAL"
        robot.shepherd_anchor = None


def select_shepherds(robots, branch):
    """dead-end 조기 감지 영역에 먼저 들어온 앞쪽 8대를 Shepherd로 선발합니다.

    군집이 완전히 포화된 뒤 선발하지 않으므로 일반 로봇이 Shepherd 뒤,
    즉 벽과 Shepherd 경계 사이에 끼는 현상을 줄입니다.
    """
    reset_robot_roles(robots)

    capture_rect = early_capture_regions[branch]
    candidates = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and get_robot_region(robot.position) == branch
        and capture_rect.collidepoint(robot.position.x, robot.position.y)
    ]

    # 값이 클수록 dead-end에 먼저 도착한 로봇입니다.
    candidates.sort(
        key=lambda robot: branch_progress(robot, branch),
        reverse=True,
    )

    first_arrivals = candidates[:SHEPHERD_COUNT]
    if len(first_arrivals) < SHEPHERD_COUNT:
        return []

    slots = build_shepherd_slots(branch, SHEPHERD_COUNT)
    selected = []
    unused = list(first_arrivals)

    # 첫 도착자 8대만 이용해서 통로 폭 방향 slot에 배치합니다.
    for slot in slots:
        chosen = min(
            unused,
            key=lambda robot: robot.position.distance_squared_to(slot),
        )
        unused.remove(chosen)
        chosen.role = "SHEPHERD"
        chosen.shepherd_anchor = slot.copy()
        selected.append(chosen)

    print(f"[Shepherd] early first-arrival selected: {len(selected)}")
    return selected


def get_shepherds(robots):
    return [robot for robot in robots if robot.role == "SHEPHERD"]


def shepherd_boundary_formed(robots):
    shepherds = get_shepherds(robots)
    if not shepherds:
        return False

    return all(
        robot.shepherd_anchor is not None
        and robot.position.distance_to(robot.shepherd_anchor)
        <= SHEPHERD_FORM_TOLERANCE
        for robot in shepherds
    )


def normal_backtracking_metrics(robots, branch):
    """Branch 안 일반 로봇이 Junction 방향으로 흐르기 시작했는지 계산합니다."""
    normals = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and get_robot_region(robot.position) == branch
    ]

    if not normals:
        return 1.0, 0.0, 0

    backtrack_direction = get_backtrack_direction(branch)
    signed_speeds = [
        robot.velocity.dot(backtrack_direction)
        for robot in normals
    ]

    moving_count = sum(
        speed >= FLOW_SPEED_THRESHOLD
        for speed in signed_speeds
    )
    moving_ratio = moving_count / len(normals)
    average_speed = sum(max(0.0, speed) for speed in signed_speeds) / len(normals)

    return moving_ratio, average_speed, len(normals)


def release_shepherds_into_flow(robots):
    """일반 로봇의 후진 흐름이 형성되면 Shepherd를 즉시 해제합니다.

    고정 상태에서 갑자기 0속도로 풀리면 뒤늦게 출발할 수 있으므로,
    현재 일반 로봇 흐름의 평균 속도 이상으로 Junction 방향 초기속도를 줍니다.
    """
    released = 0
    backtrack_direction = get_backtrack_direction(active_branch)

    normal_signed_speeds = [
        robot.velocity.dot(backtrack_direction)
        for robot in robots
        if robot.role == "NORMAL"
        and get_robot_region(robot.position) == active_branch
    ]

    positive_speeds = [
        max(0.0, speed)
        for speed in normal_signed_speeds
    ]

    if positive_speeds:
        average_flow_speed = max(
            SHEPHERD_RELEASE_SPEED,
            sum(positive_speeds) / len(positive_speeds),
        )
    else:
        average_flow_speed = SHEPHERD_RELEASE_SPEED

    for robot in robots:
        if robot.role != "SHEPHERD":
            continue

        robot.role = "NORMAL"
        robot.shepherd_anchor = None
        robot.velocity = backtrack_direction * average_flow_speed
        robot.acceleration.update(0.0, 0.0)
        released += 1

    print(
        "[Shepherd] released into normal flow: "
        f"{released}, speed={average_flow_speed:.2f}"
    )


# =========================================================
# 11. SPH 계산
# =========================================================


def compute_densities(robots, grid):
    self_contribution = spiky_kernel(0.0, SMOOTHING_LENGTH)
    h_squared = SMOOTHING_LENGTH**2

    for robot_i in robots:
        density = self_contribution

        for robot_j in iter_neighbor_candidates(robot_i, grid):
            if robot_i is robot_j:
                continue

            r_ij = robot_i.position - robot_j.position
            distance_squared = r_ij.length_squared()

            if distance_squared <= h_squared:
                density += spiky_kernel(
                    math.sqrt(distance_squared),
                    SMOOTHING_LENGTH,
                )

        robot_i.density = max(density, EPSILON)


def compute_pressures(robots, reference_density):
    for robot in robots:
        density_ratio = robot.density / max(reference_density, EPSILON)
        robot.pressure = (
            PRESSURE_GAIN
            * robot.density
            * (density_ratio**STIFFNESS_EXPONENT - 1.0)
        )

        # PRESSURE_PUSH 동안 Shepherd를 고압 경계입자로 만듦
        if (
            phase == SimulationPhase.PRESSURE_PUSH
            and robot.role == "SHEPHERD"
        ):
            ramp = min(
                1.0,
                0.25 + pressure_push_timer / max(PRESSURE_RAMP_TIME, EPSILON),
            )
            robot.pressure += (
                PRESSURE_GAIN
                * robot.density
                * SHEPHERD_PRESSURE_FACTOR
                * ramp
            )


def compute_route_force(robot):
    region = get_robot_region(robot.position)
    junction_target = pygame.Vector2(center_x, center_y)
    bottom_hold = get_bottom_hold_point()
    force = pygame.Vector2(0.0, 0.0)

    if robot.role in {"ANCHOR", "RELAY"}:
        return force

    # Anchor 선발 이후에는 실제로 Anchor 통신망에 연결되어 있고,
    # 현재 메시지를 수신한 로봇만 DFS 이동 명령을 수행합니다.
    if phase != SimulationPhase.MOVE_TO_JUNCTION and junction_anchor is not None:
        if (
            not robot.connected_to_anchor
            or robot.received_command != phase.name
        ):
            return force

        if (
            phase in {
                SimulationPhase.EXPLORE_BRANCH,
                SimulationPhase.FORM_SHEPHERD_BOUNDARY,
                SimulationPhase.PRESSURE_PUSH,
                SimulationPhase.FLOW_BACKTRACK,
            }
            and robot.received_branch != active_branch
        ):
            return force

    if phase == SimulationPhase.MOVE_TO_JUNCTION:
        force = (
            normalized_direction_toward(robot.position, junction_target)
            * ROUTE_FORCE
        )

    elif phase == SimulationPhase.EXPLORE_BRANCH:
        force = (
            normalized_direction_toward(
                robot.position,
                get_branch_tip_target(active_branch),
            )
            * ROUTE_FORCE
            * relay_motion_scale
        )

    elif phase == SimulationPhase.FORM_SHEPHERD_BOUNDARY:
        if robot.role == "SHEPHERD":
            # 실제 anchor 정렬은 Robot.update()가 직접 처리합니다.
            force = pygame.Vector2(0.0, 0.0)
        elif region == active_branch:
            # Shepherd가 경계를 만드는 동안 Branch 내부 일반 로봇은 대기합니다.
            force = pygame.Vector2(0.0, 0.0)
        else:
            # Junction 및 아래쪽 로봇은 Junction에서 대기합니다.
            force = (
                normalized_direction_toward(
                    robot.position,
                    JUNCTION_STAGING_POSITION,
                )
                * OUTLET_FORCE
            )

    elif phase == SimulationPhase.PRESSURE_PUSH:
        if robot.role == "SHEPHERD":
            force = pygame.Vector2(0.0, 0.0)
        elif region != active_branch:
            force = (
                normalized_direction_toward(
                    robot.position,
                    JUNCTION_STAGING_POSITION,
                )
                * OUTLET_FORCE
            )
        # Branch 안 NORMAL은 Shepherd의 가상 압력으로 먼저 후진합니다.

    elif phase == SimulationPhase.FLOW_BACKTRACK:
        # Branch 로봇은 Parent Junction까지만 복귀합니다.
        target = (
            junction_target
            if region == active_branch
            else JUNCTION_STAGING_POSITION
        )
        force = (
            normalized_direction_toward(robot.position, target)
            * FLOW_BACKTRACK_FORCE
        )

    elif phase == SimulationPhase.JUNCTION_SWITCH:
        # 다음 Branch 결정 전 짧게 Junction 안에서 안정화합니다.
        force = (
            normalized_direction_toward(
                robot.position,
                JUNCTION_STAGING_POSITION,
            )
            * OUTLET_FORCE
        )

    elif phase == SimulationPhase.RETURN_TO_BASE:
        # 모든 Branch가 VISITED 된 후에만 시작 위치로 최종 복귀합니다.
        force = (
            normalized_direction_toward(robot.position, bottom_hold)
            * OUTLET_FORCE
        )

    elif phase == SimulationPhase.DONE:
        force = pygame.Vector2(0.0, 0.0)

    # 복도 중심선 구속
    if region in {"UP", "BOTTOM"}:
        force.x += CENTERING_GAIN * (center_x - robot.position.x)
    elif region in {"LEFT", "RIGHT"}:
        force.y += CENTERING_GAIN * (center_y - robot.position.y)

    return force


def compute_sph_forces(robots, grid):
    h_squared = SMOOTHING_LENGTH**2
    virtual_radius_squared = VIRTUAL_PRESSURE_RADIUS**2
    backtrack_direction = get_backtrack_direction(active_branch)

    for robot_i in robots:
        if robot_i.role in {"ANCHOR", "RELAY"}:
            robot_i.acceleration.update(0.0, 0.0)
            continue

        # Shepherd는 FORM/PRESSURE_PUSH 중 움직이는 입자가 아니라
        # 고정된 SPH 경계 입자입니다. 자신의 이동 힘 계산은 생략하지만,
        # 다른 일반 로봇의 이웃 robot_j로는 계속 포함됩니다.
        if (
            robot_i.role == "SHEPHERD"
            and phase in {
                SimulationPhase.FORM_SHEPHERD_BOUNDARY,
                SimulationPhase.PRESSURE_PUSH,
            }
        ):
            robot_i.acceleration.update(0.0, 0.0)
            continue

        pressure_force = pygame.Vector2(0.0, 0.0)
        viscosity_force = pygame.Vector2(0.0, 0.0)
        repulsion_force = pygame.Vector2(0.0, 0.0)
        virtual_pressure_force = pygame.Vector2(0.0, 0.0)
        cohesion_force = pygame.Vector2(0.0, 0.0)

        neighbor_count = 0
        neighbor_center_sum = pygame.Vector2(0.0, 0.0)

        for robot_j in iter_neighbor_candidates(robot_i, grid):
            if robot_i is robot_j:
                continue

            r_ij = robot_i.position - robot_j.position
            distance_squared = r_ij.length_squared()

            # 가상 압력은 SPH support보다 넓게 쓸 수 있으므로 먼저 검사
            if (
                phase == SimulationPhase.PRESSURE_PUSH
                and robot_i.role == "NORMAL"
                and robot_j.role == "SHEPHERD"
                and distance_squared <= virtual_radius_squared
                and branch_progress(robot_i, active_branch)
                <= branch_progress(robot_j, active_branch) + 2.0
            ):
                distance = math.sqrt(max(distance_squared, EPSILON))
                ratio = max(0.0, 1.0 - distance / VIRTUAL_PRESSURE_RADIUS)
                ramp = min(
                    1.0,
                    0.25 + pressure_push_timer / max(PRESSURE_RAMP_TIME, EPSILON),
                )
                virtual_pressure_force += (
                    backtrack_direction
                    * VIRTUAL_PRESSURE_FORCE
                    * ratio**2
                    * ramp
                )

            if distance_squared <= EPSILON or distance_squared > h_squared:
                continue

            neighbor_count += 1
            neighbor_center_sum += robot_j.position

            distance = math.sqrt(distance_squared)
            gradient = spiky_gradient(r_ij, SMOOTHING_LENGTH)

            # 대칭형 SPH 압력력
            pressure_coefficient = (
                robot_i.pressure / max(robot_i.density**2, EPSILON)
                + robot_j.pressure / max(robot_j.density**2, EPSILON)
            )
            pressure_force += -pressure_coefficient * gradient

            # Monaghan 점성: 접근 중일 때만 적용
            v_ij = robot_i.velocity - robot_j.velocity
            approach_value = v_ij.dot(r_ij)

            if approach_value < 0.0:
                mu_ij = (
                    SMOOTHING_LENGTH
                    * approach_value
                    / (distance_squared + 0.01 * SMOOTHING_LENGTH**2)
                )

                c_i_squared = (
                    robot_i.pressure + PRESSURE_GAIN * robot_i.density
                ) / max(robot_i.density, EPSILON)
                c_j_squared = (
                    robot_j.pressure + PRESSURE_GAIN * robot_j.density
                ) / max(robot_j.density, EPSILON)

                c_i = math.sqrt(max(c_i_squared, 0.0))
                c_j = math.sqrt(max(c_j_squared, 0.0))
                c_ij = 0.5 * (c_i + c_j)
                mean_density = 0.5 * (
                    robot_i.density + robot_j.density
                )

                pi_ij = (
                    -VISCOSITY_XI1 * c_ij * mu_ij
                    + VISCOSITY_XI2 * mu_ij**2
                ) / max(mean_density, EPSILON)

                viscosity_force += -pi_ij * gradient

            if distance < SAFE_RADIUS:
                direction_away = r_ij / distance
                penetration_ratio = (
                    SAFE_RADIUS - distance
                ) / SAFE_RADIUS
                repulsion_force += (
                    REPULSION_GAIN
                    * penetration_ratio
                    * direction_away
                )

        route_force = compute_route_force(robot_i)
        connectivity_force = compute_connectivity_force(robot_i, grid)

        # 뒤처진 로봇 보정. 단 PRESSURE_PUSH 중 branch 안 NORMAL에는
        # direct route boost를 주지 않아 압력에 의한 복귀를 유지합니다.
        direct_pressure_phase_normal = (
            phase == SimulationPhase.PRESSURE_PUSH
            and robot_i.role == "NORMAL"
            and get_robot_region(robot_i.position) == active_branch
        )

        if (
            0 < neighbor_count < ISOLATION_NEIGHBOR_THRESHOLD
            and not direct_pressure_phase_normal
        ):
            local_center = neighbor_center_sum / neighbor_count
            cohesion_direction = local_center - robot_i.position

            if cohesion_direction.length_squared() > EPSILON:
                ratio = (
                    ISOLATION_NEIGHBOR_THRESHOLD - neighbor_count
                ) / ISOLATION_NEIGHBOR_THRESHOLD
                cohesion_force = (
                    cohesion_direction.normalize()
                    * LOCAL_COHESION_GAIN
                    * ratio
                )

        if (
            neighbor_count < ISOLATION_NEIGHBOR_THRESHOLD
            and not direct_pressure_phase_normal
        ):
            boost_ratio = (
                ISOLATION_NEIGHBOR_THRESHOLD - neighbor_count
            ) / ISOLATION_NEIGHBOR_THRESHOLD
            route_force *= 1.0 + ISOLATION_ROUTE_BOOST * boost_ratio

        total_acceleration = (
            pressure_force
            + viscosity_force
            + repulsion_force
            + virtual_pressure_force
            + cohesion_force
            + route_force
            + connectivity_force
            - DAMPING * robot_i.velocity
        )

        robot_i.acceleration = limit_vector(
            total_acceleration,
            MAX_ACCELERATION,
        )

# =========================================================
# 12. 포화 감지 / 상태 전이
# =========================================================


def is_dead_end_saturated(robots, branch, reference_density):
    end_rect = dead_end_regions[branch]
    robots_at_end = [
        robot
        for robot in robots
        if end_rect.collidepoint(
            robot.position.x,
            robot.position.y,
        )
    ]

    if len(robots_at_end) < MIN_ROBOTS_AT_END:
        return False

    average_speed = sum(
        robot.velocity.length() for robot in robots_at_end
    ) / len(robots_at_end)

    average_density = sum(
        robot.density for robot in robots_at_end
    ) / len(robots_at_end)

    return (
        average_speed < SATURATION_SPEED_THRESHOLD
        and average_density
        > reference_density * SATURATION_DENSITY_RATIO
    )


def count_branch_roles(robots, branch):
    normal_count = 0
    shepherd_count = 0
    relay_count = 0

    for robot in robots:
        if get_robot_region(robot.position) != branch:
            continue

        if robot.role == "SHEPHERD":
            shepherd_count += 1
        elif robot.role == "RELAY":
            relay_count += 1
        else:
            normal_count += 1

    return normal_count, shepherd_count, relay_count


def update_simulation_state(robots, dt, reference_density):
    global phase
    global active_branch
    global saturation_timer
    global shepherd_form_timer
    global pressure_push_timer
    global flow_establish_timer
    global junction_switch_timer

    # Junction 최초 진입자를 Anchor로 선발합니다.
    anchor = elect_junction_anchor(robots)

    if phase == SimulationPhase.MOVE_TO_JUNCTION:
        robots_in_junction = sum(
            get_robot_region(robot.position) == "JUNCTION"
            and robot.role != "ANCHOR"
            for robot in robots
        )

        # Anchor가 주차점에 도착하고 안전한 직접 통신 이웃을 확보한 뒤에만
        # 첫 Branch 명령을 전파합니다. 배치 중 링크가 불안정하면 군집을 기다립니다.
        anchor_ready = anchor_deployment_ready(anchor, robots)

        if (
            anchor is not None
            and anchor_ready
            and robots_in_junction >= JUNCTION_ENTRY_COUNT
        ):
            selected = anchor_select_next_branch(anchor)

            if selected is None:
                begin_final_return(anchor)
            else:
                phase = SimulationPhase.EXPLORE_BRANCH
                print(f"[DFS] Branch 탐색 시작: {active_branch}")

    elif phase == SimulationPhase.EXPLORE_BRANCH:
        # 군집이 전진하는 동안 필요한 일부 로봇만 Relay로 순차 배치합니다.
        update_relay_deployment(robots, dt)

        # Dead-end 조기 감지 영역에 먼저 도착한 8대를 즉시 Shepherd로 선발합니다.
        capture_count = sum(
            get_robot_region(robot.position) == active_branch
            and robot.role == "NORMAL"
            and early_capture_regions[active_branch].collidepoint(
                robot.position.x,
                robot.position.y,
            )
            for robot in robots
        )

        if (
            capture_count >= SHEPHERD_COUNT
            and relay_plan_complete(robots)
        ):
            selected = select_shepherds(robots, active_branch)
            if len(selected) == SHEPHERD_COUNT:
                phase = SimulationPhase.FORM_SHEPHERD_BOUNDARY
                shepherd_form_timer = 0.0
                print(f"[DFS] dead-end first arrivals detected: {active_branch}")
                print("[Shepherd] 조기 경계층 형성 시작")

    elif phase == SimulationPhase.FORM_SHEPHERD_BOUNDARY:
        shepherd_form_timer += dt

        if (
            shepherd_boundary_formed(robots)
            or shepherd_form_timer >= SHEPHERD_FORM_TIMEOUT
        ):
            phase = SimulationPhase.PRESSURE_PUSH
            pressure_push_timer = 0.0
            flow_establish_timer = 0.0
            print("[Shepherd] 고압 경계 형성 완료")
            print("[Pressure] 일반 로봇 backtracking 흐름 생성 시작")

    elif phase == SimulationPhase.PRESSURE_PUSH:
        pressure_push_timer += dt

        moving_ratio, average_speed, normal_count = normal_backtracking_metrics(
            robots,
            active_branch,
        )

        flow_is_established = (
            normal_count >= FLOW_MIN_NORMAL_COUNT
            and moving_ratio >= FLOW_RATIO_THRESHOLD
            and average_speed >= FLOW_AVERAGE_SPEED_THRESHOLD
        )

        if flow_is_established:
            flow_establish_timer += dt
        else:
            flow_establish_timer = 0.0

        if (
            flow_establish_timer >= FLOW_ESTABLISH_DWELL_TIME
            or pressure_push_timer >= FLOW_FALLBACK_TIME
            or normal_count == 0
        ):
            release_shepherds_into_flow(robots)
            phase = SimulationPhase.FLOW_BACKTRACK
            flow_establish_timer = 0.0
            print(
                "[Pressure] 흐름 형성 완료 "
                f"(ratio={moving_ratio:.2f}, avg={average_speed:.2f})"
            )
            print("[Shepherd] 즉시 일반 로봇으로 합류")

    elif phase == SimulationPhase.FLOW_BACKTRACK:
        # 복귀 군집이 Relay에 닿을 때마다 가장 먼 Relay부터 NORMAL로 합류합니다.
        update_relay_retraction(robots, dt)

        robots_remaining_in_branch = sum(
            get_robot_region(robot.position) == active_branch
            for robot in robots
        )
        robots_in_junction = sum(
            get_robot_region(robot.position) == "JUNCTION"
            and robot.role != "ANCHOR"
            for robot in robots
        )

        remaining_relays = len(get_active_branch_relays(robots))

        if (
            robots_remaining_in_branch <= BRANCH_CLEAR_LIMIT
            and robots_in_junction >= JUNCTION_SWITCH_COUNT
            and remaining_relays == 0
        ):
            anchor_complete_active_branch(anchor, active_branch)
            phase = SimulationPhase.JUNCTION_SWITCH
            junction_switch_timer = 0.0
            print("[DFS] Parent Junction 도착, 다음 Branch 전환 준비")

    elif phase == SimulationPhase.JUNCTION_SWITCH:
        junction_switch_timer += dt

        if junction_switch_timer >= JUNCTION_SWITCH_DWELL_TIME:
            next_branch = anchor_select_next_branch(anchor)

            if next_branch is None:
                begin_final_return(anchor)
            else:
                saturation_timer = 0.0
                shepherd_form_timer = 0.0
                pressure_push_timer = 0.0
                flow_establish_timer = 0.0
                phase = SimulationPhase.EXPLORE_BRANCH
                print(f"[DFS] Junction에서 바로 다음 Branch 탐색: {active_branch}")

    elif phase == SimulationPhase.RETURN_TO_BASE:
        robots_in_bottom = sum(
            get_robot_region(robot.position) == "BOTTOM"
            for robot in robots
        )
        remaining_special_roles = sum(
            robot.role in {"ANCHOR", "RELAY", "SHEPHERD"}
            for robot in robots
        )

        if (
            robots_in_bottom >= RETURN_BOTTOM_TARGET_COUNT
            and remaining_special_roles == 0
        ):
            phase = SimulationPhase.DONE
            print(
                "[DFS] 탐색 및 최종 Base 복귀 완료: "
                f"{robots_in_bottom}/{len(robots)} robots"
            )


# =========================================================
# 14. 초기화
# =========================================================


def reset_dfs_state():
    global phase
    global active_branch
    global branch_states
    global junction_anchor
    global simulation_time
    global junction_switch_timer
    global saturation_timer
    global shepherd_form_timer
    global pressure_push_timer
    global flow_establish_timer
    global communication_sequence
    global last_message_signature
    global relay_slots
    global relay_deploy_cooldown
    global relay_retract_cooldown
    global relay_retract_clear_timer
    global relay_motion_scale

    phase = SimulationPhase.MOVE_TO_JUNCTION
    active_branch = DFS_ORDER[0]
    branch_states = {
        "UP": "UNVISITED",
        "LEFT": "UNVISITED",
        "RIGHT": "UNVISITED",
    }

    junction_anchor = None
    simulation_time = 0.0
    junction_switch_timer = 0.0
    saturation_timer = 0.0
    shepherd_form_timer = 0.0
    pressure_push_timer = 0.0
    flow_establish_timer = 0.0
    communication_sequence = 0
    last_message_signature = None
    relay_slots = []
    relay_deploy_cooldown = 0.0
    relay_retract_cooldown = 0.0
    relay_retract_clear_timer = 0.0
    relay_motion_scale = 1.0


def initialize_simulation():
    reset_dfs_state()

    if SPAWN_MODE == "grid":
        robots = create_grid_robots(ROBOT_COUNT)
    else:
        robots = create_random_robots(ROBOT_COUNT)

    if not robots:
        raise RuntimeError("로봇을 생성하지 못했습니다.")

    grid = build_spatial_grid(robots)
    compute_densities(robots, grid)

    initial_mean_density = sum(
        robot.density for robot in robots
    ) / len(robots)

    reference_density = initial_mean_density * 0.70
    color_reference_density = initial_mean_density * 0.68

    print(f"생성된 로봇 수: {len(robots)}")
    print(f"초기 평균 밀도: {initial_mean_density:.6f}")
    print(f"물리 기준 밀도 rho_0: {reference_density:.6f}")

    update_communication_system(robots, grid)

    return robots, reference_density, color_reference_density


robots, reference_density, color_reference_density = initialize_simulation()

# =========================================================
# 14. 실행 루프
# =========================================================

running = True
paused = False
show_density_color = False
show_regions = True
show_comm_links = SHOW_COMM_LINKS_DEFAULT

while running:
    frame_dt = min(clock.tick(FPS) / 1000.0, 0.033)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:
                robots, reference_density, color_reference_density = (
                    initialize_simulation()
                )
            elif event.key == pygame.K_d:
                show_density_color = not show_density_color
            elif event.key == pygame.K_v:
                show_regions = not show_regions
            elif event.key == pygame.K_c:
                show_comm_links = not show_comm_links
            elif event.key == pygame.K_ESCAPE:
                running = False

    if not paused:
        simulation_time += frame_dt
        substep_dt = frame_dt / SUBSTEPS

        for _ in range(SUBSTEPS):
            spatial_grid = build_spatial_grid(robots)
            update_communication_system(robots, spatial_grid)
            compute_densities(robots, spatial_grid)
            compute_pressures(robots, reference_density)
            compute_sph_forces(robots, spatial_grid)

            for robot in robots:
                robot.update(substep_dt)

        update_anchor_entry_records(robots, simulation_time)

        update_simulation_state(
            robots,
            frame_dt,
            reference_density,
        )

        # 상태 전이로 Anchor 명령이 바뀌었을 수 있으므로 즉시 다시 전파합니다.
        spatial_grid = build_spatial_grid(robots)
        update_communication_system(robots, spatial_grid)
    else:
        spatial_grid = build_spatial_grid(robots)
        update_communication_system(robots, spatial_grid)
        compute_densities(robots, spatial_grid)
        compute_pressures(robots, reference_density)

    # -----------------------------------------------------
    # 그리기
    # -----------------------------------------------------

    screen.fill(BACKGROUND_COLOR)
    pygame.draw.polygon(screen, FLOOR_COLOR, cross_points)
    pygame.draw.polygon(screen, WALL_COLOR, cross_points, width=5)

    if show_regions:
        pygame.draw.rect(screen, JUNCTION_COLOR, junction_rect, width=2)
        pygame.draw.rect(screen, ANCHOR_COLOR, anchor_election_rect, width=1)
        pygame.draw.circle(
            screen,
            ANCHOR_COLOR,
            (round(ANCHOR_PARK_POSITION.x), round(ANCHOR_PARK_POSITION.y)),
            4,
            width=1,
        )

        for branch, rect in dead_end_regions.items():
            border_color = (
                END_REGION_COLOR
                if branch == active_branch
                else (175, 175, 175)
            )
            pygame.draw.rect(screen, border_color, rect, width=2)

        # 조기 Shepherd 선발 영역을 얇은 보라색 선으로 표시
        pygame.draw.rect(
            screen,
            SHEPHERD_COLOR,
            early_capture_regions[active_branch],
            width=1,
        )

        # Shepherd anchor 표시
        for robot in get_shepherds(robots):
            if robot.shepherd_anchor is not None:
                pygame.draw.circle(
                    screen,
                    SHEPHERD_COLOR,
                    (
                        round(robot.shepherd_anchor.x),
                        round(robot.shepherd_anchor.y),
                    ),
                    3,
                    width=1,
                )

        # 현재 Branch의 Relay slot과 실제 Relay chain 표시
        draw_relay_plan(screen, robots)

    pygame.draw.circle(
        screen,
        JUNCTION_COLOR,
        (center_x, center_y),
        5,
    )

    if show_comm_links:
        draw_communication_links(screen, robots)

    for robot in robots:
        robot.draw(
            screen,
            color_reference_density,
            show_density_color,
        )

    normal_remaining, shepherd_remaining, relay_remaining = count_branch_roles(
        robots,
        active_branch,
    )

    phase_text = phase.name
    active_text = (
        active_branch
        if phase not in {SimulationPhase.MOVE_TO_JUNCTION, SimulationPhase.DONE}
        else "-"
    )
    anchor_id = junction_anchor.robot_id if junction_anchor is not None else "-"
    anchor_selected = (
        junction_anchor.selected_branch
        if junction_anchor is not None
        and junction_anchor.selected_branch is not None
        else "-"
    )

    anchor_parked = (
        junction_anchor is not None
        and junction_anchor.anchor_position is not None
        and junction_anchor.position.distance_to(
            junction_anchor.anchor_position
        ) <= ANCHOR_POSITION_TOLERANCE
    )
    anchor_direct_safe = (
        sum(
            robot is not junction_anchor
            and junction_anchor is not None
            and junction_anchor.position.distance_to(robot.position)
            <= COMM_SAFE_DISTANCE
            for robot in robots
        )
        if junction_anchor is not None
        else 0
    )
    communication_stats = get_communication_stats(robots)

    hud_lines = [
        f"FPS: {clock.get_fps():.1f}",
        f"Robots: {len(robots)}",
        f"Motion speed: {MOTION_SPEED_MULTIPLIER:.1f}x",
        f"Phase: {phase_text}",
        (
            f"Anchor ID: {anchor_id} | role="
            f"{junction_anchor.role if junction_anchor is not None else '-'} | "
            f"selected: {anchor_selected}"
        ),
        f"Anchor deploy: parked={anchor_parked} | safe neighbors={anchor_direct_safe}",
        (
            "Comm: "
            f"{communication_stats['connected']}/{len(robots)} connected | "
            f"max hop={communication_stats['max_hop']}"
        ),
        (
            "Comm link: "
            f"direct={communication_stats['direct_anchor_neighbors']} | "
            f"min margin={communication_stats['minimum_margin']:.1f}px"
        ),
        f"Active branch: {active_text}",
        (
            f"Relays: {len(get_relays(robots))}/{len(relay_slots)} | "
            f"motion scale={relay_motion_scale:.2f}"
        ),
        (
            "Return to Base: "
            f"{sum(get_robot_region(robot.position) == 'BOTTOM' for robot in robots)}"
            f"/{len(robots)}"
        ),
        f"Shepherds: {len(get_shepherds(robots))}",
        (
            "In branch: "
            f"normal={normal_remaining}, "
            f"relay={relay_remaining}, "
            f"shepherd={shepherd_remaining}"
        ),
        (
            "Branch state: "
            f"UP={branch_states['UP']} | "
            f"LEFT={branch_states['LEFT']} | "
            f"RIGHT={branch_states['RIGHT']}"
        ),
    ]

    for index, text in enumerate(hud_lines):
        rendered = small_font.render(text, True, TEXT_COLOR)
        screen.blit(rendered, (15, 14 + index * 22))

    control_text = font.render(
        "SPACE pause | R reset | D density | V regions | C comm links | ESC quit",
        True,
        TEXT_COLOR,
    )
    screen.blit(control_text, (15, SCREEN_HEIGHT - 30))

    pygame.display.flip()

pygame.quit()
sys.exit()
