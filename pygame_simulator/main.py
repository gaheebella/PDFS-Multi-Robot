import pygame

pygame.init()

# 화면 크기
WIDTH = 800
HEIGHT = 600

# 창 생성
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("My Robot Simulator")

clock = pygame.time.Clock()

# 로봇 초기 위치
robot_x = 400
robot_y = 300
robot_speed = 5

running = True

while running:
    # 창 닫기 이벤트
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 현재 누르고 있는 키 확인
    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        robot_x -= robot_speed

    if keys[pygame.K_RIGHT]:
        robot_x += robot_speed

    if keys[pygame.K_UP]:
        robot_y -= robot_speed

    if keys[pygame.K_DOWN]:
        robot_y += robot_speed

    # 화면 밖으로 못 나가게 제한
    robot_x = max(20, min(WIDTH - 20, robot_x))
    robot_y = max(20, min(HEIGHT - 20, robot_y))

    # 배경
    screen.fill((240, 240, 240))

    # 통신 범위
    pygame.draw.circle(
        screen,
        (150, 150, 150),
        (robot_x, robot_y),
        100,
        2
    )

    # 로봇
    pygame.draw.circle(
        screen,
        (0, 100, 255),
        (robot_x, robot_y),
        20
    )

    pygame.display.flip()
    clock.tick(60)

pygame.quit()