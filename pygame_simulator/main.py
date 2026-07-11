import math
import random
import sys
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
pygame.display.set_caption("SPH + DFS Shepherd Pressure Boundary")
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
JUNCTION_COLOR = (153, 164, 181)
END_REGION_COLOR = (224, 171, 115)

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
    REGROUP = auto()
    SELECT_NEXT_BRANCH = auto()
    DONE = auto()


DFS_ORDER = ["UP", "LEFT", "RIGHT"]
BRANCH_DIRECTIONS = {
    "UP": pygame.Vector2(0.0, -1.0),
    "LEFT": pygame.Vector2(-1.0, 0.0),
    "RIGHT": pygame.Vector2(1.0, 0.0),
}

phase = SimulationPhase.MOVE_TO_JUNCTION
dfs_index = 0
active_branch = DFS_ORDER[0]
branch_states = {
    "UP": "UNVISITED",
    "LEFT": "UNVISITED",
    "RIGHT": "UNVISITED",
}
branch_states[active_branch] = "ACTIVE"

saturation_timer = 0.0
shepherd_form_timer = 0.0
pressure_push_timer = 0.0
flow_establish_timer = 0.0

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
DAMPING = 2.8

SAFE_RADIUS = 7.5
REPULSION_GAIN = 260.0

ROUTE_FORCE = 52.0
OUTLET_FORCE = 44.0
SHEPHERD_BACKTRACK_FORCE = 50.0
CENTERING_GAIN = 1.2

MAX_SPEED = 78.0
MAX_ACCELERATION = 520.0
EPSILON = 1e-8

# 고립 로봇이 무리를 따라오도록 하는 보조항
ISOLATION_NEIGHBOR_THRESHOLD = 4
ISOLATION_ROUTE_BOOST = 1.1
LOCAL_COHESION_GAIN = 20.0

# Junction 진입 후 첫 branch 탐색 시작 조건
JUNCTION_ENTRY_COUNT = 18

# Shepherd boundary
# dead-end에 가장 먼저 도착한 8대를 즉시 선발하기 위한 조기 감지 깊이
SHEPHERD_COUNT = 8
EARLY_CAPTURE_DEPTH = 34
SHEPHERD_EDGE_MARGIN = 12
SHEPHERD_FORM_KP = 9.0
SHEPHERD_HOLD_KP = 9.5
SHEPHERD_HOLD_KD = 3.8
SHEPHERD_FORM_TOLERANCE = 5.0
SHEPHERD_FORM_TIMEOUT = 1.4

# Shepherd의 가상 압력
SHEPHERD_PRESSURE_FACTOR = 5.2
VIRTUAL_PRESSURE_RADIUS = 60.0
VIRTUAL_PRESSURE_FORCE = 105.0
PRESSURE_RAMP_TIME = 1.2

# 일반 로봇의 backtracking 흐름이 형성되면 Shepherd를 즉시 해제
FLOW_SPEED_THRESHOLD = 2.5
FLOW_RATIO_THRESHOLD = 0.72
FLOW_AVERAGE_SPEED_THRESHOLD = 3.0
FLOW_ESTABLISH_DWELL_TIME = 0.40
FLOW_MIN_NORMAL_COUNT = 8
FLOW_FALLBACK_TIME = 5.0

# 조기 해제 뒤 전체가 함께 빠져나가는 단계
FLOW_BACKTRACK_FORCE = 46.0
BRANCH_CLEAR_LIMIT = 1

# Regroup 조건
REGROUP_BOTTOM_TARGET_COUNT = int(ROBOT_COUNT * 0.94)
REGROUP_SPREAD_THRESHOLD = 38.0

CELL_SIZE = max(SMOOTHING_LENGTH, VIRTUAL_PRESSURE_RADIUS)

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

    Shepherd가 일반 로봇을 밀어내고 마지막으로 복귀할 때까지
    현재 branch는 계속 열어 둡니다.
    """
    region = get_robot_region(position)

    if phase == SimulationPhase.MOVE_TO_JUNCTION:
        return region in {"BOTTOM", "JUNCTION"}

    if phase in {
        SimulationPhase.EXPLORE_BRANCH,
        SimulationPhase.FORM_SHEPHERD_BOUNDARY,
        SimulationPhase.PRESSURE_PUSH,
        SimulationPhase.FLOW_BACKTRACK,
        SimulationPhase.REGROUP,
    }:
        return region in {"BOTTOM", "JUNCTION", active_branch}

    if phase == SimulationPhase.DONE:
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

    def update(self, dt):
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

        if self.role == "SHEPHERD":
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
# 10. Shepherd 선택 / 경계 생성
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
        if get_robot_region(robot.position) == branch
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
    """압력으로 일반 로봇의 backtracking 흐름이 형성되면 즉시 합류시킵니다."""
    released = 0
    for robot in robots:
        if robot.role == "SHEPHERD":
            robot.role = "NORMAL"
            robot.shepherd_anchor = None
            released += 1

    print(f"[Shepherd] released into normal flow: {released}")


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

    if phase == SimulationPhase.MOVE_TO_JUNCTION:
        force = (
            normalized_direction_toward(robot.position, junction_target)
            * ROUTE_FORCE
        )

    elif phase == SimulationPhase.EXPLORE_BRANCH:
        target = (
            junction_target
            if region == "BOTTOM"
            else get_branch_tip_target(active_branch)
        )
        force = normalized_direction_toward(robot.position, target) * ROUTE_FORCE

    elif phase == SimulationPhase.FORM_SHEPHERD_BOUNDARY:
        if robot.role == "SHEPHERD" and robot.shepherd_anchor is not None:
            position_error = robot.shepherd_anchor - robot.position
            force = (
                SHEPHERD_FORM_KP * position_error
                - SHEPHERD_HOLD_KD * robot.velocity
            )
        elif region in {"JUNCTION", "BOTTOM"}:
            # 뒤에서 더 많은 일반 로봇이 branch로 들어오지 않게 아래 대기지점으로 보냄
            force = (
                normalized_direction_toward(robot.position, bottom_hold)
                * OUTLET_FORCE
            )
        else:
            # Branch 안 NORMAL은 전진력을 즉시 제거하고 감쇠만 받게 함
            force = pygame.Vector2(0.0, 0.0)

    elif phase == SimulationPhase.PRESSURE_PUSH:
        if robot.role == "SHEPHERD" and robot.shepherd_anchor is not None:
            position_error = robot.shepherd_anchor - robot.position
            force = (
                SHEPHERD_HOLD_KP * position_error
                - SHEPHERD_HOLD_KD * robot.velocity
            )
        elif region in {"JUNCTION", "BOTTOM"}:
            force = (
                normalized_direction_toward(robot.position, bottom_hold)
                * OUTLET_FORCE
            )
        # Branch 안 NORMAL에는 직접 후진력을 주지 않고 Shepherd 압력으로 흐름을 시작시킴

    elif phase == SimulationPhase.FLOW_BACKTRACK:
        # 흐름이 만들어진 뒤에는 Shepherd 역할을 해제하고 전원이 같은 일반 로봇으로 합류
        target = bottom_hold if region in {"JUNCTION", "BOTTOM"} else junction_target
        force = (
            normalized_direction_toward(robot.position, target)
            * FLOW_BACKTRACK_FORCE
        )

    elif phase in {SimulationPhase.REGROUP, SimulationPhase.DONE}:
        force = (
            normalized_direction_toward(robot.position, bottom_hold)
            * OUTLET_FORCE
        )

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

    for robot in robots:
        if get_robot_region(robot.position) != branch:
            continue

        if robot.role == "SHEPHERD":
            shepherd_count += 1
        else:
            normal_count += 1

    return normal_count, shepherd_count


def update_simulation_state(robots, dt, reference_density):
    global phase
    global dfs_index
    global active_branch
    global saturation_timer
    global shepherd_form_timer
    global pressure_push_timer
    global flow_establish_timer

    bottom_hold = get_bottom_hold_point()

    if phase == SimulationPhase.MOVE_TO_JUNCTION:
        robots_in_junction = sum(
            get_robot_region(robot.position) == "JUNCTION"
            for robot in robots
        )

        if robots_in_junction >= JUNCTION_ENTRY_COUNT:
            phase = SimulationPhase.EXPLORE_BRANCH
            print(f"[DFS] Branch 탐색 시작: {active_branch}")

    elif phase == SimulationPhase.EXPLORE_BRANCH:
        # 포화될 때까지 기다리지 않고 조기 감지 영역에 먼저 도착한 8대를 즉시 선발
        capture_count = sum(
            get_robot_region(robot.position) == active_branch
            and early_capture_regions[active_branch].collidepoint(
                robot.position.x,
                robot.position.y,
            )
            for robot in robots
        )

        if capture_count >= SHEPHERD_COUNT:
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

        # 일반 로봇 대다수가 Junction 방향으로 움직이는 상태가 잠깐 유지되면 즉시 해제
        if flow_is_established:
            flow_establish_timer += dt
        else:
            flow_establish_timer = 0.0

        # 정상 조건 또는 너무 오래 걸리는 경우의 fallback
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
        robots_remaining_in_branch = sum(
            get_robot_region(robot.position) == active_branch
            for robot in robots
        )

        if robots_remaining_in_branch <= BRANCH_CLEAR_LIMIT:
            phase = SimulationPhase.REGROUP
            print("[DFS] 전체 backtracking 완료, regroup 시작")

    elif phase == SimulationPhase.REGROUP:
        robots_in_bottom = sum(
            get_robot_region(robot.position) == "BOTTOM"
            for robot in robots
        )
        robots_in_junction = sum(
            get_robot_region(robot.position) == "JUNCTION"
            for robot in robots
        )
        robots_remaining_in_branch = sum(
            get_robot_region(robot.position) == active_branch
            for robot in robots
        )

        average_bottom_distance = sum(
            robot.position.distance_to(bottom_hold)
            for robot in robots
        ) / len(robots)

        if (
            robots_in_bottom >= REGROUP_BOTTOM_TARGET_COUNT
            and robots_in_junction <= 8
            and robots_remaining_in_branch <= 1
            and average_bottom_distance <= REGROUP_SPREAD_THRESHOLD
        ):
            branch_states[active_branch] = "VISITED"
            phase = SimulationPhase.SELECT_NEXT_BRANCH

    elif phase == SimulationPhase.SELECT_NEXT_BRANCH:
        dfs_index += 1

        if dfs_index >= len(DFS_ORDER):
            phase = SimulationPhase.DONE
            print("[DFS] 모든 Branch 탐색 완료")
            return

        active_branch = DFS_ORDER[dfs_index]
        branch_states[active_branch] = "ACTIVE"
        saturation_timer = 0.0
        shepherd_form_timer = 0.0
        pressure_push_timer = 0.0
        flow_establish_timer = 0.0
        phase = SimulationPhase.EXPLORE_BRANCH
        print(f"[DFS] 다음 Branch 탐색: {active_branch}")

# =========================================================
# 13. 초기화
# =========================================================


def reset_dfs_state():
    global phase
    global dfs_index
    global active_branch
    global branch_states
    global saturation_timer
    global shepherd_form_timer
    global pressure_push_timer
    global flow_establish_timer

    phase = SimulationPhase.MOVE_TO_JUNCTION
    dfs_index = 0
    active_branch = DFS_ORDER[0]
    branch_states = {
        "UP": "UNVISITED",
        "LEFT": "UNVISITED",
        "RIGHT": "UNVISITED",
    }
    branch_states[active_branch] = "ACTIVE"

    saturation_timer = 0.0
    shepherd_form_timer = 0.0
    pressure_push_timer = 0.0
    flow_establish_timer = 0.0


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

    return robots, reference_density, color_reference_density


robots, reference_density, color_reference_density = initialize_simulation()

# =========================================================
# 14. 실행 루프
# =========================================================

running = True
paused = False
show_density_color = False
show_regions = True

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
            elif event.key == pygame.K_ESCAPE:
                running = False

    if not paused:
        substep_dt = frame_dt / SUBSTEPS

        for _ in range(SUBSTEPS):
            spatial_grid = build_spatial_grid(robots)
            compute_densities(robots, spatial_grid)
            compute_pressures(robots, reference_density)
            compute_sph_forces(robots, spatial_grid)

            for robot in robots:
                robot.update(substep_dt)

        update_simulation_state(
            robots,
            frame_dt,
            reference_density,
        )
    else:
        spatial_grid = build_spatial_grid(robots)
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

    pygame.draw.circle(
        screen,
        JUNCTION_COLOR,
        (center_x, center_y),
        5,
    )

    for robot in robots:
        robot.draw(
            screen,
            color_reference_density,
            show_density_color,
        )

    normal_remaining, shepherd_remaining = count_branch_roles(
        robots,
        active_branch,
    )

    phase_text = phase.name
    active_text = active_branch if phase != SimulationPhase.DONE else "-"

    hud_lines = [
        f"FPS: {clock.get_fps():.1f}",
        f"Robots: {len(robots)}",
        f"Phase: {phase_text}",
        f"Active branch: {active_text}",
        f"Shepherds: {len(get_shepherds(robots))}",
        f"In branch: normal={normal_remaining}, shepherd={shepherd_remaining}",
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
        "SPACE pause | R reset | D soft density | V regions | ESC quit",
        True,
        TEXT_COLOR,
    )
    screen.blit(control_text, (15, SCREEN_HEIGHT - 30))

    pygame.display.flip()

pygame.quit()
sys.exit()