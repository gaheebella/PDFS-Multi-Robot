import math
import random
import sys

import pygame


pygame.init()

# =========================================================
# 1. 화면 설정
# =========================================================

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
FPS = 60
SUBSTEPS = 2

screen = pygame.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT)
)
pygame.display.set_caption(
    "HydroSwarm SPH Multi-Robot Simulation"
)

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)


# =========================================================
# 2. 색상
# =========================================================

BACKGROUND_COLOR = (245, 245, 240)
FLOOR_COLOR = (215, 220, 228)
WALL_COLOR = (44, 55, 72)

ROBOT_COLOR = (30, 136, 229)
ROBOT_OUTLINE_COLOR = (15, 70, 120)

TEXT_COLOR = (35, 42, 54)


# =========================================================
# 3. 십자가 맵 설정
# =========================================================

center_x = 400
center_y = 350

# 모든 통로의 폭
corridor_width = 120
half_width = corridor_width // 2

# 위·왼쪽·아래쪽 길이
normal_length = 180

# 오른쪽 통로는 2배
right_length = normal_length * 2

cross_points = [
    # 위쪽 끝
    (
        center_x - half_width,
        center_y - half_width - normal_length,
    ),
    (
        center_x + half_width,
        center_y - half_width - normal_length,
    ),

    # 위쪽에서 오른쪽 방향
    (
        center_x + half_width,
        center_y - half_width,
    ),
    (
        center_x + half_width + right_length,
        center_y - half_width,
    ),

    # 오른쪽 끝
    (
        center_x + half_width + right_length,
        center_y + half_width,
    ),
    (
        center_x + half_width,
        center_y + half_width,
    ),

    # 아래쪽 끝
    (
        center_x + half_width,
        center_y + half_width + normal_length,
    ),
    (
        center_x - half_width,
        center_y + half_width + normal_length,
    ),

    # 아래쪽에서 왼쪽 방향
    (
        center_x - half_width,
        center_y + half_width,
    ),
    (
        center_x - half_width - normal_length,
        center_y + half_width,
    ),

    # 왼쪽 끝
    (
        center_x - half_width - normal_length,
        center_y - half_width,
    ),
    (
        center_x - half_width,
        center_y - half_width,
    ),
]


# =========================================================
# 4. SPH 파라미터
# =========================================================

# 로봇 표시 및 충돌 크기
ROBOT_RADIUS = 3

# SPH 상호작용 반경 h
SMOOTHING_LENGTH = 30.0

# 압력 상태방정식
PRESSURE_GAIN = 1500.0
STIFFNESS_EXPONENT = 0.5

# Monaghan 인공점성
VISCOSITY_XI1 = 0.8
VISCOSITY_XI2 = 1.0

# 전체 속도 감쇠
DAMPING = 2.5

# 별도 충돌 반발력
SAFE_RADIUS = 8.0
REPULSION_GAIN = 5000.0

# 아래쪽 통로에서 Junction으로 주입하는 힘
INJECTION_FORCE = 45.0

# 속도와 가속도 제한
MAX_SPEED = 90.0
MAX_ACCELERATION = 500.0

EPSILON = 1e-8


# =========================================================
# 5. 이동 가능 영역 Mask
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


def is_walkable(position, radius):
    """
    로봇 원이 십자가 복도 내부에 완전히 들어가는지 확인합니다.
    중심점과 원 주변 8개 지점을 검사합니다.
    """
    x = int(round(position.x))
    y = int(round(position.y))

    diagonal = int(
        round(radius / math.sqrt(2.0))
    )

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
        if not (
            0 <= px < SCREEN_WIDTH
            and 0 <= py < SCREEN_HEIGHT
        ):
            return False

        if walkable_mask.get_at((px, py)) == 0:
            return False

    return True


# =========================================================
# 6. 벡터 제한 함수
# =========================================================

def limit_vector(vector, maximum_length):
    """
    벡터 길이가 maximum_length를 넘으면 크기를 제한합니다.
    """
    if vector.length_squared() > maximum_length ** 2:
        vector.scale_to_length(maximum_length)

    return vector


# =========================================================
# 7. SPH Spiky Kernel
# =========================================================

def spiky_kernel(distance, h):
    """
    HydroSwarm 식 (5)

    W(r, h) = 10/(pi*h^2) * (1-r/h)^3
    """
    if distance < 0.0 or distance > h:
        return 0.0

    q = 1.0 - distance / h

    return (
        10.0
        / (math.pi * h * h)
        * q ** 3
    )


def spiky_gradient(r_ij, h):
    """
    Spiky kernel의 위치 기울기입니다.

    r_ij = r_i - r_j
    """
    distance = r_ij.length()

    if distance <= EPSILON or distance > h:
        return pygame.Vector2(0.0, 0.0)

    q = 1.0 - distance / h

    gradient_magnitude = (
        -30.0
        / (math.pi * h ** 3)
        * q ** 2
    )

    direction = r_ij / distance

    return gradient_magnitude * direction


# =========================================================
# 8. 로봇 클래스
# =========================================================

class Robot:
    def __init__(self, x, y, robot_id):
        self.robot_id = robot_id

        # 2차 동역학 상태
        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(0.0, 0.0)
        self.acceleration = pygame.Vector2(0.0, 0.0)

        self.radius = ROBOT_RADIUS

        # SPH 상태
        self.density = 0.0
        self.pressure = 0.0

    def update(self, dt):
        """
        가속도 → 속도 → 위치 순서로 적분합니다.
        """

        # 속도 갱신
        self.velocity += self.acceleration * dt

        # 최대속도 제한
        limit_vector(
            self.velocity,
            MAX_SPEED,
        )

        # x축과 y축을 따로 이동시켜 벽을 따라 미끄러지게 함
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

        # 다음 계산을 위해 가속도 초기화
        self.acceleration.update(0.0, 0.0)

    def draw(
        self,
        surface,
        reference_density,
        show_density_color,
    ):
        position = (
            round(self.position.x),
            round(self.position.y),
        )

        if show_density_color:
            color = density_to_color(
                self.density,
                reference_density,
            )
        else:
            color = ROBOT_COLOR

        pygame.draw.circle(
            surface,
            color,
            position,
            self.radius,
        )

        pygame.draw.circle(
            surface,
            ROBOT_OUTLINE_COLOR,
            position,
            self.radius,
            width=1,
        )


# =========================================================
# 9. 밀도 시각화
# =========================================================

def interpolate_color(color_a, color_b, ratio):
    ratio = max(0.0, min(1.0, ratio))

    return tuple(
        int(
            color_a[index]
            + (
                color_b[index]
                - color_a[index]
            )
            * ratio
        )
        for index in range(3)
    )


def density_to_color(density, reference_density):
    """
    낮은 밀도: 파랑
    기준 밀도: 초록
    높은 밀도: 주황/빨강
    """
    ratio = density / max(
        reference_density,
        EPSILON,
    )

    if ratio <= 1.0:
        return interpolate_color(
            (35, 120, 230),
            (45, 190, 120),
            ratio,
        )

    high_ratio = min(
        (ratio - 1.0) / 1.5,
        1.0,
    )

    return interpolate_color(
        (45, 190, 120),
        (230, 75, 55),
        high_ratio,
    )


# =========================================================
# 10. 격자 형태 로봇 생성
# =========================================================

def create_grid_robots(robot_count):
    robots = []

    spacing = 7

    entrance_left = (
        center_x
        - half_width
        + ROBOT_RADIUS
        + 5
    )

    entrance_right = (
        center_x
        + half_width
        - ROBOT_RADIUS
        - 5
    )

    entrance_top = (
        center_y
        + half_width
        + 20
    )

    entrance_bottom = (
        center_y
        + half_width
        + normal_length
        - ROBOT_RADIUS
        - 10
    )

    available_width = (
        entrance_right
        - entrance_left
    )

    robots_per_row = max(
        1,
        int(available_width // spacing) + 1,
    )

    for robot_id in range(robot_count):
        row = robot_id // robots_per_row
        column = robot_id % robots_per_row

        x = entrance_left + column * spacing
        y = entrance_bottom - row * spacing

        if y < entrance_top:
            print(
                f"경고: 현재 입구에는 "
                f"{len(robots)}대만 배치할 수 있습니다."
            )
            break

        robots.append(
            Robot(
                x=x,
                y=y,
                robot_id=robot_id,
            )
        )

    return robots


# =========================================================
# 11. 무작위 형태 로봇 생성
# =========================================================

def create_random_robots(robot_count):
    robots = []

    minimum_distance = (
        ROBOT_RADIUS * 2 + 2
    )

    entrance_left = (
        center_x
        - half_width
        + ROBOT_RADIUS
        + 5
    )

    entrance_right = (
        center_x
        + half_width
        - ROBOT_RADIUS
        - 5
    )

    entrance_top = (
        center_y
        + half_width
        + 20
    )

    entrance_bottom = (
        center_y
        + half_width
        + normal_length
        - ROBOT_RADIUS
        - 10
    )

    max_attempts = robot_count * 200
    attempts = 0

    while (
        len(robots) < robot_count
        and attempts < max_attempts
    ):
        attempts += 1

        x = random.uniform(
            entrance_left,
            entrance_right,
        )

        y = random.uniform(
            entrance_top,
            entrance_bottom,
        )

        candidate = pygame.Vector2(x, y)

        overlaps = False

        for robot in robots:
            if (
                candidate.distance_to(robot.position)
                < minimum_distance
            ):
                overlaps = True
                break

        if overlaps:
            continue

        robots.append(
            Robot(
                x=x,
                y=y,
                robot_id=len(robots),
            )
        )

    if len(robots) < robot_count:
        print(
            f"경고: 요청한 {robot_count}대 중 "
            f"{len(robots)}대만 배치했습니다."
        )

    return robots


# =========================================================
# 12. 밀도 계산
# =========================================================

def compute_densities(robots):
    """
    HydroSwarm 식 (6)

    rho_i = sum_j W_ij
    """
    self_contribution = spiky_kernel(
        0.0,
        SMOOTHING_LENGTH,
    )

    for robot_i in robots:
        density = self_contribution

        for robot_j in robots:
            if robot_i is robot_j:
                continue

            r_ij = (
                robot_i.position
                - robot_j.position
            )

            distance_squared = (
                r_ij.length_squared()
            )

            if (
                distance_squared
                > SMOOTHING_LENGTH ** 2
            ):
                continue

            distance = math.sqrt(
                distance_squared
            )

            density += spiky_kernel(
                distance,
                SMOOTHING_LENGTH,
            )

        robot_i.density = max(
            density,
            EPSILON,
        )


# =========================================================
# 13. 압력 계산
# =========================================================

def compute_pressures(
    robots,
    reference_density,
):
    """
    HydroSwarm 식 (8), 식 (9)

    P_i = kappa * rho_i
          * ((rho_i/rho_0)^lambda - 1)
    """
    for robot in robots:
        density_ratio = (
            robot.density
            / max(reference_density, EPSILON)
        )

        robot.pressure = (
            PRESSURE_GAIN
            * robot.density
            * (
                density_ratio
                ** STIFFNESS_EXPONENT
                - 1.0
            )
        )


# =========================================================
# 14. SPH 힘 계산
# =========================================================

def compute_sph_forces(robots):
    """
    압력력 + Monaghan 점성력 + 충돌 반발력
    + Junction 방향 유도력 + 전체 감쇠
    """
    for robot_i in robots:
        pressure_force = pygame.Vector2(
            0.0,
            0.0,
        )

        viscosity_force = pygame.Vector2(
            0.0,
            0.0,
        )

        repulsion_force = pygame.Vector2(
            0.0,
            0.0,
        )

        for robot_j in robots:
            if robot_i is robot_j:
                continue

            # 상대 위치
            r_ij = (
                robot_i.position
                - robot_j.position
            )

            distance_squared = (
                r_ij.length_squared()
            )

            if distance_squared <= EPSILON:
                continue

            if (
                distance_squared
                > SMOOTHING_LENGTH ** 2
            ):
                continue

            distance = math.sqrt(
                distance_squared
            )

            gradient = spiky_gradient(
                r_ij,
                SMOOTHING_LENGTH,
            )

            # -------------------------------------------------
            # 압력력: HydroSwarm 식 (13)
            # -------------------------------------------------

            pressure_coefficient = (
                robot_i.pressure
                / max(
                    robot_i.density ** 2,
                    EPSILON,
                )
                +
                robot_j.pressure
                / max(
                    robot_j.density ** 2,
                    EPSILON,
                )
            )

            pressure_force += (
                -pressure_coefficient
                * gradient
            )

            # -------------------------------------------------
            # Monaghan 인공점성: 식 (10), 식 (11), 식 (14)
            # -------------------------------------------------

            v_ij = (
                robot_i.velocity
                - robot_j.velocity
            )

            approach_value = v_ij.dot(
                r_ij
            )

            # 두 로봇 사이 거리가 줄어들 때만 적용
            if approach_value < 0.0:
                mu_ij = (
                    SMOOTHING_LENGTH
                    * approach_value
                    /
                    (
                        distance_squared
                        + 0.01
                        * SMOOTHING_LENGTH ** 2
                    )
                )

                b_i = (
                    PRESSURE_GAIN
                    * robot_i.density
                )

                b_j = (
                    PRESSURE_GAIN
                    * robot_j.density
                )

                c_i_squared = (
                    robot_i.pressure + b_i
                ) / max(
                    robot_i.density,
                    EPSILON,
                )

                c_j_squared = (
                    robot_j.pressure + b_j
                ) / max(
                    robot_j.density,
                    EPSILON,
                )

                c_i = math.sqrt(
                    max(c_i_squared, 0.0)
                )

                c_j = math.sqrt(
                    max(c_j_squared, 0.0)
                )

                c_ij = 0.5 * (
                    c_i + c_j
                )

                mean_density = 0.5 * (
                    robot_i.density
                    + robot_j.density
                )

                pi_ij = (
                    -VISCOSITY_XI1
                    * c_ij
                    * mu_ij
                    +
                    VISCOSITY_XI2
                    * mu_ij ** 2
                ) / max(
                    mean_density,
                    EPSILON,
                )

                viscosity_force += (
                    -pi_ij * gradient
                )

            # -------------------------------------------------
            # 충돌 방지용 별도 반발력: 식 (16)
            # -------------------------------------------------

            direction_away = (
                r_ij / distance
            )

            repulsion_force += (
                REPULSION_GAIN
                / max(
                    distance,
                    SAFE_RADIUS,
                ) ** 2
                * direction_away
            )

        # -----------------------------------------------------
        # 아래쪽 입구에서 Junction 방향으로 로봇을 주입
        # -----------------------------------------------------

        guidance_force = pygame.Vector2(
            0.0,
            0.0,
        )

        junction_bottom = (
            center_y + half_width
        )

        if robot_i.position.y > junction_bottom:
            guidance_force.y = (
                -INJECTION_FORCE
            )

        # -----------------------------------------------------
        # 전체 제어 입력: HydroSwarm 식 (17)
        # -----------------------------------------------------

        total_acceleration = (
            pressure_force
            + viscosity_force
            + repulsion_force
            + guidance_force
            - DAMPING * robot_i.velocity
        )

        limit_vector(
            total_acceleration,
            MAX_ACCELERATION,
        )

        robot_i.acceleration = (
            total_acceleration
        )


# =========================================================
# 15. 초기화
# =========================================================

ROBOT_COUNT = 250

# "grid" 또는 "random"
SPAWN_MODE = "grid"


def initialize_simulation():
    if SPAWN_MODE == "grid":
        new_robots = create_grid_robots(
            ROBOT_COUNT
        )
    else:
        new_robots = create_random_robots(
            ROBOT_COUNT
        )

    if not new_robots:
        raise RuntimeError(
            "로봇을 한 대도 생성하지 못했습니다."
        )

    # 초기 밀도 계산
    compute_densities(new_robots)

    initial_mean_density = (
        sum(
            robot.density
            for robot in new_robots
        )
        / len(new_robots)
    )

    # 초기 평균보다 낮게 두어 자연 팽창 유도
    reference_density = (
        initial_mean_density * 0.65
    )

    print(
        f"생성된 로봇 수: {len(new_robots)}"
    )

    print(
        "초기 평균 밀도: "
        f"{initial_mean_density:.6f}"
    )

    print(
        "기준 밀도 rho_0: "
        f"{reference_density:.6f}"
    )

    return (
        new_robots,
        reference_density,
    )


robots, reference_density = (
    initialize_simulation()
)


# =========================================================
# 16. 실행 루프
# =========================================================

running = True
paused = False
show_density_color = True

while running:
    frame_dt = min(
        clock.tick(FPS) / 1000.0,
        0.033,
    )

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            # Space: 일시정지
            if event.key == pygame.K_SPACE:
                paused = not paused

            # R: 초기화
            elif event.key == pygame.K_r:
                robots, reference_density = (
                    initialize_simulation()
                )

            # D: 밀도 색상 표시 전환
            elif event.key == pygame.K_d:
                show_density_color = (
                    not show_density_color
                )

            # ESC: 종료
            elif event.key == pygame.K_ESCAPE:
                running = False

    # -----------------------------------------------------
    # 시뮬레이션 계산
    # -----------------------------------------------------

    if not paused:
        substep_dt = (
            frame_dt / SUBSTEPS
        )

        for _ in range(SUBSTEPS):
            # 1. 국소 밀도
            compute_densities(robots)

            # 2. Tait 압력
            compute_pressures(
                robots,
                reference_density,
            )

            # 3. 압력·점성·반발력 계산
            compute_sph_forces(robots)

            # 4. 가속도·속도·위치 갱신
            for robot in robots:
                robot.update(substep_dt)

    # 정지 중에도 밀도 색상은 갱신
    else:
        compute_densities(robots)
        compute_pressures(
            robots,
            reference_density,
        )

    # -----------------------------------------------------
    # 화면 출력
    # -----------------------------------------------------

    screen.fill(BACKGROUND_COLOR)

    # 복도 내부
    pygame.draw.polygon(
        screen,
        FLOOR_COLOR,
        cross_points,
    )

    # 벽
    pygame.draw.polygon(
        screen,
        WALL_COLOR,
        cross_points,
        width=5,
    )

    # Junction 중심 표시
    pygame.draw.circle(
        screen,
        (120, 130, 145),
        (center_x, center_y),
        5,
    )

    # 로봇 출력
    for robot in robots:
        robot.draw(
            screen,
            reference_density,
            show_density_color,
        )

    # HUD 정보
    fps_text = font.render(
        f"FPS: {clock.get_fps():.1f}",
        True,
        TEXT_COLOR,
    )

    robot_text = font.render(
        f"Robots: {len(robots)}",
        True,
        TEXT_COLOR,
    )

    status_text = font.render(
        (
            "PAUSED"
            if paused
            else "RUNNING"
        ),
        True,
        TEXT_COLOR,
    )

    control_text = font.render(
        "SPACE: pause | R: reset | D: density color",
        True,
        TEXT_COLOR,
    )

    screen.blit(
        fps_text,
        (15, 15),
    )

    screen.blit(
        robot_text,
        (15, 40),
    )

    screen.blit(
        status_text,
        (15, 65),
    )

    screen.blit(
        control_text,
        (15, SCREEN_HEIGHT - 30),
    )

    pygame.display.flip()


pygame.quit()
sys.exit()