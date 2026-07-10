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

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("SPH Multi-Robot Simulation")

clock = pygame.time.Clock()

# 색상
BACKGROUND_COLOR = (245, 245, 240)
FLOOR_COLOR = (215, 220, 228)
WALL_COLOR = (44, 55, 72)
ROBOT_COLOR = (30, 136, 229)
ROBOT_OUTLINE_COLOR = (15, 70, 120)


# =========================================================
# 2. 십자가 맵 설정
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


# =========================================================
# 3. 로봇 클래스
# =========================================================

class Robot:
    def __init__(self, x, y, robot_id):
        self.robot_id = robot_id

        # 위치
        self.position = pygame.Vector2(x, y)

        # 추후 이동 구현에 사용할 속도와 가속도
        self.velocity = pygame.Vector2(0, 0)
        self.acceleration = pygame.Vector2(0, 0)

        # 화면에 표시할 로봇 크기
        self.radius = 3

    def update(self, dt):
        """
        현재는 로봇을 움직이지 않습니다.
        이후 SPH 힘과 속도를 추가할 때 사용합니다.
        """
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

        # 다음 프레임 전에 가속도 초기화
        self.acceleration.update(0, 0)

    def draw(self, surface):
        position = (
            round(self.position.x),
            round(self.position.y),
        )

        pygame.draw.circle(
            surface,
            ROBOT_COLOR,
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
# 4. 격자 형태 로봇 생성
# =========================================================

def create_grid_robots(robot_count):
    robots = []

    robot_radius = 3

    # 로봇 중심 사이 간격
    spacing = 10

    # 아래쪽 통로 내부 범위
    entrance_left = center_x - half_width + robot_radius + 5
    entrance_right = center_x + half_width - robot_radius - 5

    entrance_top = center_y + half_width + 20
    entrance_bottom = (
        center_y
        + half_width
        + normal_length
        - robot_radius
        - 10
    )

    available_width = entrance_right - entrance_left

    # 한 줄에 들어갈 로봇 수
    robots_per_row = max(
        1,
        int(available_width // spacing) + 1,
    )

    for robot_id in range(robot_count):
        row = robot_id // robots_per_row
        column = robot_id % robots_per_row

        x = entrance_left + column * spacing

        # 아래에서 위쪽 방향으로 배치
        y = entrance_bottom - row * spacing

        # 아래쪽 통로를 넘어갈 정도로 로봇 수가 많으면 중단
        if y < entrance_top:
            print(
                f"경고: 현재 입구에는 로봇 {len(robots)}대만 "
                f"격자로 배치할 수 있습니다."
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
# 5. 무작위 형태 로봇 생성
# =========================================================

def create_random_robots(robot_count):
    robots = []

    robot_radius = 3
    minimum_distance = robot_radius * 2 + 2

    entrance_left = center_x - half_width + robot_radius + 5
    entrance_right = center_x + half_width - robot_radius - 5

    entrance_top = center_y + half_width + 20
    entrance_bottom = (
        center_y
        + half_width
        + normal_length
        - robot_radius
        - 10
    )

    max_attempts = robot_count * 200
    attempts = 0

    while len(robots) < robot_count and attempts < max_attempts:
        attempts += 1

        x = random.uniform(entrance_left, entrance_right)
        y = random.uniform(entrance_top, entrance_bottom)

        candidate_position = pygame.Vector2(x, y)

        # 기존 로봇과 겹치는지 검사
        overlaps = False

        for robot in robots:
            distance = candidate_position.distance_to(
                robot.position
            )

            if distance < minimum_distance:
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
# 6. 로봇 생성
# =========================================================

ROBOT_COUNT = 100

# "grid" 또는 "random" 중 선택
SPAWN_MODE = "grid"

if SPAWN_MODE == "grid":
    robots = create_grid_robots(ROBOT_COUNT)
else:
    robots = create_random_robots(ROBOT_COUNT)

print(f"생성된 로봇 수: {len(robots)}")


# =========================================================
# 7. 실행 루프
# =========================================================

running = True

while running:
    # 초 단위 프레임 시간
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 배경
    screen.fill(BACKGROUND_COLOR)

    # 십자가 방 내부
    pygame.draw.polygon(
        screen,
        FLOOR_COLOR,
        cross_points,
    )

    # 십자가 방 외곽선
    pygame.draw.polygon(
        screen,
        WALL_COLOR,
        cross_points,
        width=5,
    )

    # 로봇 업데이트 및 그리기
    for robot in robots:
        robot.update(dt)
        robot.draw(screen)

    pygame.display.flip()

pygame.quit()
sys.exit()