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
pygame.display.set_caption("SPH + DFS Cross-Corridor Simulation")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)
small_font = pygame.font.SysFont(None, 20)

# 색상
BACKGROUND_COLOR = (245, 245, 240)
FLOOR_COLOR = (215, 220, 228)
WALL_COLOR = (44, 55, 72)
TEXT_COLOR = (35, 42, 54)
ROBOT_OUTLINE_COLOR = (15, 70, 120)
JUNCTION_COLOR = (135, 145, 160)
END_REGION_COLOR = (215, 120, 80)

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

# =========================================================
# 3. DFS 상태
# =========================================================


class SimulationPhase(Enum):
    MOVE_TO_JUNCTION = auto()
    EXPLORE_BRANCH = auto()
    BACKTRACK = auto()
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

# =========================================================
# 4. SPH 및 이동 파라미터
# =========================================================

ROBOT_COUNT = 220
SPAWN_MODE = "grid"  # "grid" 또는 "random"
ROBOT_RADIUS = 3
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
BACKTRACK_FORCE = 48.0
CENTERING_GAIN = 1.2

MAX_SPEED = 78.0
MAX_ACCELERATION = 520.0
EPSILON = 1e-8

MIN_ROBOTS_AT_END = 18
SATURATION_SPEED_THRESHOLD = 7.0
SATURATION_DENSITY_RATIO = 1.22
SATURATION_DWELL_TIME = 1.4

JUNCTION_ENTRY_COUNT = 18
BACKTRACK_REMAINING_LIMIT = 4

# 공간 해싱 셀 크기: SPH support radius와 동일하게 둠
CELL_SIZE = SMOOTHING_LENGTH

# =========================================================
# 5. 맵 마스크 및 영역 판정
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
    """DFS가 선택하지 않은 branch를 가상 밸브처럼 닫습니다."""
    region = get_robot_region(position)

    if phase == SimulationPhase.MOVE_TO_JUNCTION:
        return region in {"BOTTOM", "JUNCTION"}

    if phase in {
        SimulationPhase.EXPLORE_BRANCH,
        SimulationPhase.BACKTRACK,
    }:
        return region in {"BOTTOM", "JUNCTION", active_branch}

    return region != "OUTSIDE"


def is_walkable(position, radius):
    """로봇 원이 복도 내부와 현재 허용된 DFS 영역 안에 있는지 확인합니다."""
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
    """초기에도 고밀도가 눈에 잘 보이도록 시각화 기준을 낮게 둡니다."""
    ratio = density / max(color_reference_density, EPSILON)

    if ratio <= 1.0:
        return interpolate_color(
            (35, 120, 230),
            (55, 195, 120),
            ratio,
        )

    # ratio=1.0부터 1.6 사이를 빠르게 초록→빨강으로 변환
    high_ratio = min((ratio - 1.0) / 0.60, 1.0)
    return interpolate_color(
        (55, 195, 120),
        (235, 65, 50),
        high_ratio,
    )


# =========================================================
# 7. 로봇 클래스
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

    def update(self, dt):
        self.velocity += self.acceleration * dt
        limit_vector(self.velocity, MAX_SPEED)

        # x, y를 따로 적분해서 벽을 따라 미끄러질 수 있게 함
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
        position = (round(self.position.x), round(self.position.y))
        color = (
            density_to_color(self.density, color_reference_density)
            if show_density_color
            else (30, 136, 229)
        )

        pygame.draw.circle(surface, color, position, self.radius)
        pygame.draw.circle(
            surface,
            ROBOT_OUTLINE_COLOR,
            position,
            self.radius,
            width=1,
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
            robots.append(
                Robot(candidate.x, candidate.y, len(robots))
            )

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
# 10. SPH 계산
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


def compute_route_force(robot):
    region = get_robot_region(robot.position)
    force = pygame.Vector2(0.0, 0.0)

    if phase == SimulationPhase.MOVE_TO_JUNCTION:
        target = pygame.Vector2(center_x, center_y)
        direction = target - robot.position
        if direction.length_squared() > EPSILON:
            force = direction.normalize() * ROUTE_FORCE

    elif phase == SimulationPhase.EXPLORE_BRANCH:
        # 아래쪽 로봇은 먼저 Junction으로 들어옴
        if region == "BOTTOM":
            target = pygame.Vector2(center_x, center_y)
            direction = target - robot.position
            if direction.length_squared() > EPSILON:
                force = direction.normalize() * ROUTE_FORCE
        else:
            force = BRANCH_DIRECTIONS[active_branch] * ROUTE_FORCE

    elif phase == SimulationPhase.BACKTRACK:
        # 1차 구현에서는 branch 전체 로봇을 아래 대기지점까지 복귀시킴
        hold_point = pygame.Vector2(
            center_x,
            center_y + half_width + normal_length - 18,
        )
        direction = hold_point - robot.position
        if direction.length_squared() > EPSILON:
            force = direction.normalize() * BACKTRACK_FORCE

    # 경로 중심선 쪽으로 약한 구속력 추가
    region = get_robot_region(robot.position)
    if region in {"UP", "BOTTOM"}:
        force.x += CENTERING_GAIN * (center_x - robot.position.x)
    elif region in {"LEFT", "RIGHT"}:
        force.y += CENTERING_GAIN * (center_y - robot.position.y)

    return force


def compute_sph_forces(robots, grid):
    h_squared = SMOOTHING_LENGTH**2

    for robot_i in robots:
        pressure_force = pygame.Vector2(0.0, 0.0)
        viscosity_force = pygame.Vector2(0.0, 0.0)
        repulsion_force = pygame.Vector2(0.0, 0.0)

        for robot_j in iter_neighbor_candidates(robot_i, grid):
            if robot_i is robot_j:
                continue

            r_ij = robot_i.position - robot_j.position
            distance_squared = r_ij.length_squared()

            if distance_squared <= EPSILON or distance_squared > h_squared:
                continue

            distance = math.sqrt(distance_squared)
            gradient = spiky_gradient(r_ij, SMOOTHING_LENGTH)

            # 대칭형 SPH 압력력
            pressure_coefficient = (
                robot_i.pressure / max(robot_i.density**2, EPSILON)
                + robot_j.pressure / max(robot_j.density**2, EPSILON)
            )
            pressure_force += -pressure_coefficient * gradient

            # Monaghan 인공점성: 두 로봇이 접근할 때만 작동
            v_ij = robot_i.velocity - robot_j.velocity
            approach_value = v_ij.dot(r_ij)

            if approach_value < 0.0:
                mu_ij = (
                    SMOOTHING_LENGTH
                    * approach_value
                    / (
                        distance_squared
                        + 0.01 * SMOOTHING_LENGTH**2
                    )
                )

                # P + kappa*rho = kappa*rho*(rho/rho0)^lambda > 0
                c_i_squared = (
                    robot_i.pressure
                    + PRESSURE_GAIN * robot_i.density
                ) / max(robot_i.density, EPSILON)
                c_j_squared = (
                    robot_j.pressure
                    + PRESSURE_GAIN * robot_j.density
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

            # 최소거리 충돌 반발력은 SAFE_RADIUS 안에서만 적용
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

        total_acceleration = (
            pressure_force
            + viscosity_force
            + repulsion_force
            + route_force
            - DAMPING * robot_i.velocity
        )

        robot_i.acceleration = limit_vector(
            total_acceleration,
            MAX_ACCELERATION,
        )


# =========================================================
# 11. DFS 포화 감지 및 상태 전이
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


def update_simulation_state(robots, dt, reference_density):
    global phase
    global dfs_index
    global active_branch
    global saturation_timer

    if phase == SimulationPhase.MOVE_TO_JUNCTION:
        robots_in_junction = sum(
            get_robot_region(robot.position) == "JUNCTION"
            for robot in robots
        )

        if robots_in_junction >= JUNCTION_ENTRY_COUNT:
            phase = SimulationPhase.EXPLORE_BRANCH
            print(f"[DFS] Branch 탐색 시작: {active_branch}")

    elif phase == SimulationPhase.EXPLORE_BRANCH:
        if is_dead_end_saturated(
            robots,
            active_branch,
            reference_density,
        ):
            saturation_timer += dt
        else:
            saturation_timer = 0.0

        if saturation_timer >= SATURATION_DWELL_TIME:
            phase = SimulationPhase.BACKTRACK
            saturation_timer = 0.0
            print(f"[DFS] Branch 포화 감지: {active_branch}")
            print("[DFS] Parent Junction 방향 Backtracking 시작")

    elif phase == SimulationPhase.BACKTRACK:
        robots_remaining = sum(
            get_robot_region(robot.position) == active_branch
            for robot in robots
        )

        if robots_remaining <= BACKTRACK_REMAINING_LIMIT:
            branch_states[active_branch] = "VISITED"
            phase = SimulationPhase.SELECT_NEXT_BRANCH

    elif phase == SimulationPhase.SELECT_NEXT_BRANCH:
        dfs_index += 1

        if dfs_index >= len(DFS_ORDER):
            phase = SimulationPhase.DONE
            print("[DFS] 모든 Branch 탐색 완료")
            for robot in robots:
                robot.velocity.update(0.0, 0.0)
                robot.acceleration.update(0.0, 0.0)
            return

        active_branch = DFS_ORDER[dfs_index]
        branch_states[active_branch] = "ACTIVE"
        phase = SimulationPhase.EXPLORE_BRANCH
        print(f"[DFS] 다음 Branch 탐색: {active_branch}")


# =========================================================
# 12. 초기화 / 리셋
# =========================================================


def reset_dfs_state():
    global phase
    global dfs_index
    global active_branch
    global branch_states
    global saturation_timer

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

    # 물리 계산 기준: 초기 밀도보다 낮게 두어 집단 팽창 유도
    reference_density = initial_mean_density * 0.70

    # 색 표시 기준: 더 낮게 두어 초기 고밀도도 빨갛게 보이게 함
    color_reference_density = initial_mean_density * 0.68

    print(f"생성된 로봇 수: {len(robots)}")
    print(f"초기 평균 밀도: {initial_mean_density:.6f}")
    print(f"물리 기준 밀도 rho_0: {reference_density:.6f}")

    return robots, reference_density, color_reference_density


robots, reference_density, color_reference_density = (
    initialize_simulation()
)

# =========================================================
# 13. 실행 루프
# =========================================================

running = True
paused = False
show_density_color = True
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

    if not paused and phase != SimulationPhase.DONE:
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
    # 화면 그리기
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

    # HUD
    phase_text = phase.name
    active_text = active_branch if phase != SimulationPhase.DONE else "-"

    hud_lines = [
        f"FPS: {clock.get_fps():.1f}",
        f"Robots: {len(robots)}",
        f"Phase: {phase_text}",
        f"Active branch: {active_text}",
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
        "SPACE pause | R reset | D density color | V regions | ESC quit",
        True,
        TEXT_COLOR,
    )
    screen.blit(control_text, (15, SCREEN_HEIGHT - 30))

    pygame.display.flip()

pygame.quit()
sys.exit()