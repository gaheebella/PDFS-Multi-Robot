#!/usr/bin/env python3
import math
import collections
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


class FollowerController(Node):
    def __init__(self):
        super().__init__('follower_controller')

        self.leader_pose = None
        self.follower_pose = None
        self.scan_msg = None
        self.use_breadcrumb = False

        # Leader 우선권 양보 상태머신
        # 상태: 'follow' → 'yielding' → 'follow'
        #
        # [follow]
        #   leader가 follower 쪽으로 접근 중 + 거리 < approach_detect_dist
        #   → 'yielding' 진입
        #
        # [yielding]
        #   follower 후진 유지
        #   leader가 follower를 지나쳐서 멀어지기 시작하면
        #   (거리가 다시 증가 + 거리 > approach_clear_dist)
        #   → 'follow' 복귀
        self._yield_state = 'follow'          # 'follow' | 'yielding'
        self._prev_leader_dist = None         # 직전 루프의 leader 거리 (거리 증가 감지용)
        self.approach_detect_dist = 0.40      # 이 거리 이내 + 접근 중이면 yielding 진입
        self.approach_clear_dist  = 0.55      # 이 거리 이상 + 멀어지면 follow 복귀
        self.backoff_speed = -0.10            # 후진 속도 (m/s)

        # odom 안정화 대기 (시작 직후 쓰레기 값으로 인한 오회전 방지)
        # 10Hz 타이머 기준 약 1초 (10 tick) 대기
        self._init_ticks = 0
        self._init_done  = False

        # 제어 파라미터
        # burger 반경 0.105m 기준
        self.comm_range = 2.0
        self.follow_distance = 0.01        # 거의 붙어서 따라감
        self.k_linear = 0.8               # 속도 반응 높임
        self.k_angular = 2.5
        self.max_linear = 0.20            # burger 정격 속도 수준으로 복구
        self.max_angular = 1.5
        self.search_angular_speed = 1.0

        # 히스테리시스 임계값
        self.ROTATE_ENTER = 0.4
        self.ROTATE_EXIT = 0.15
        self._rotating = False

        # Breadcrumb path 파라미터
        self.path_record_min_dist = 0.03   # 촘촘하게 (burger 반경의 1/3)
        self.max_path_length = 3000
        self.waypoint_reach_dist = 0.12    # 0.08 → 0.12 (경계값 문제 해결)
        self.lookahead_steps = 4           # 2 → 4 (leader 속도 따라잡기)

        # 장애물 회피 파라미터
        self.obstacle_stop_dist = 0.35
        self.obstacle_clear_dist = 0.42
        self.obstacle_turn_speed = 1.0
        # 회피 상태 플래그
        self._avoiding = False

        # Leader 경로 저장 덱
        self.leader_path = collections.deque(maxlen=self.max_path_length)
        self.waypoint_idx = 0

        self.leader_sub = self.create_subscription(
            Odometry, '/leader_odom', self.leader_callback, 10)
        self.follower_sub = self.create_subscription(
            Odometry, '/follower_odom', self.follower_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/follower_scan', self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(
            Twist, '/follower_cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.control_loop)

    # ------------------------------------------------------------------ #
    #  콜백
    # ------------------------------------------------------------------ #

    def leader_callback(self, msg):
        self.leader_pose = msg.pose.pose
        lx = msg.pose.pose.position.x
        ly = msg.pose.pose.position.y

        if self.leader_path:
            last_x, last_y = self.leader_path[-1]
            if math.hypot(lx - last_x, ly - last_y) <= self.path_record_min_dist:
                return

        self.leader_path.append((lx, ly))

    def follower_callback(self, msg):
        self.follower_pose = msg.pose.pose

    def scan_callback(self, msg):
        self.scan_msg = msg

    # ------------------------------------------------------------------ #
    #  유틸리티
    # ------------------------------------------------------------------ #

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def get_front_min(self):
        """전방 ±30° 내 최소 거리 반환. scan 없으면 10.0"""
        if self.scan_msg is None:
            return 10.0
        ranges = self.scan_msg.ranges
        n = len(ranges)
        front = list(ranges[0:30]) + list(ranges[n - 30:])
        return min((r for r in front if math.isfinite(r)), default=10.0)

    # ------------------------------------------------------------------ #
    #  Waypoint 선택
    # ------------------------------------------------------------------ #

    def select_waypoint(self, fx, fy):
        path = list(self.leader_path)
        n = len(path)

        if n == 0:
            return None

        self.waypoint_idx = min(self.waypoint_idx, n - 1)

        # leader가 멀리 가버린 경우 catch-up:
        # path 끝(leader 현재위치)부터 거꾸로 탐색해서
        # follower가 waypoint_reach_dist 이내로 접근 가능한
        # 가장 앞쪽 idx로 점프
        for i in range(n - 1, self.waypoint_idx, -1):
            wx, wy = path[i]
            if math.hypot(fx - wx, fy - wy) < self.waypoint_reach_dist * 4:
                # 이 점에 접근 가능하면 여기서부터 시작
                self.waypoint_idx = i
                break

        # 현재 waypoint 도달 시 다음으로 전진
        while self.waypoint_idx < n - 1:
            wx, wy = path[self.waypoint_idx]
            if math.hypot(fx - wx, fy - wy) < self.waypoint_reach_dist:
                self.waypoint_idx += 1
            else:
                break

        target_idx = min(self.waypoint_idx + self.lookahead_steps, n - 1)
        return path[target_idx]

    # ------------------------------------------------------------------ #
    #  장애물 회피
    #  핵심 원칙:
    #    - 전방 장애물 감지 시 _avoiding = True
    #    - 전방이 clear_dist 이상 뚫릴 때까지 e_theta 방향으로 회전 유지
    #    - 뚫리면 _avoiding = False → 추종 복귀
    # ------------------------------------------------------------------ #

    def handle_obstacle(self, cmd, e_theta):
        """
        반환값: (회피 중 여부, cmd)
        회피 중이면 True 반환 → 호출자는 즉시 publish 후 return
        """
        front_min = self.get_front_min()

        # 히스테리시스: 진입은 stop_dist, 탈출은 clear_dist
        if not self._avoiding:
            if front_min < self.obstacle_stop_dist:
                self._avoiding = True
        else:
            if front_min >= self.obstacle_clear_dist:
                self._avoiding = False

        if self._avoiding:
            cmd.linear.x = 0.0
            # 추종 목표(e_theta) 방향으로 회전 → 뚫리는 순간 자연스럽게 전진 복귀
            cmd.angular.z = (
                self.obstacle_turn_speed if e_theta >= 0
                else -self.obstacle_turn_speed
            )
            self.get_logger().info(
                f'AVOIDING: front={front_min:.2f}, '
                f'turn={"L" if e_theta >= 0 else "R"}, '
                f'clear_at={self.obstacle_clear_dist:.2f}',
                throttle_duration_sec=0.5
            )
            return True, cmd

        return False, cmd

    # ------------------------------------------------------------------ #
    #  메인 제어 루프
    # ------------------------------------------------------------------ #

    def control_loop(self):
        cmd = Twist()

        if self.leader_pose is None or self.follower_pose is None:
            self.cmd_pub.publish(cmd)
            return

        # odom 안정화 대기: 양쪽 odom 수신 후 10 tick 동안 정지
        if not self._init_done:
            self._init_ticks += 1
            if self._init_ticks >= 10:
                self._init_done = True
                self.get_logger().info('odom stabilized, starting control')
            else:
                self.cmd_pub.publish(cmd)  # 정지 유지
                return

        fx = self.follower_pose.position.x
        fy = self.follower_pose.position.y
        fyaw = self.quaternion_to_yaw(self.follower_pose.orientation)

        lx = self.leader_pose.position.x
        ly = self.leader_pose.position.y
        leader_dist = math.hypot(lx - fx, ly - fy)

        # 통신 범위 밖: 탐색 회전
        if leader_dist > self.comm_range:
            cmd.linear.x = 0.0
            cmd.angular.z = self.search_angular_speed
            self._rotating = False
            self._avoiding = False
            self._yield_state = 'follow'
            self._prev_leader_dist = None
            self.use_breadcrumb = False
            self.waypoint_idx = 0
            self.get_logger().info(
                f'OUT OF RANGE: d={leader_dist:.2f} -> searching',
                throttle_duration_sec=1.0
            )
            self.cmd_pub.publish(cmd)
            return

        # ── Leader 우선권 양보 상태머신 ───────────────────────────────────
        is_approaching = (
            self._prev_leader_dist is not None
            and leader_dist < self.approach_detect_dist  # 거리 임계값 이내
            and leader_dist < self._prev_leader_dist     # 가까워지는 중
        )
        # 거리가 멀어지고 + 충분히 벌어지면 복귀
        is_clearing = (
            self._prev_leader_dist is not None
            and leader_dist > self._prev_leader_dist
            and leader_dist > self.approach_clear_dist
        )
        # 거리 임계값 이내면 무조건 yielding 유지 (노이즈로 is_approaching 놓쳐도 유지)
        force_yield = leader_dist < self.approach_detect_dist * 0.8

        self._prev_leader_dist = leader_dist

        if self._yield_state == 'follow':
            if is_approaching or force_yield:
                self._yield_state = 'yielding'
                self._avoiding = False
                self.get_logger().info(
                    f'YIELD START: d={leader_dist:.2f}'
                )
        elif self._yield_state == 'yielding':
            if is_clearing and not force_yield:
                self._yield_state = 'follow'
                self.get_logger().info(
                    f'YIELD END: d={leader_dist:.2f}, resuming follow'
                )

        if self._yield_state == 'yielding':
            cmd.linear.x = self.backoff_speed
            cmd.angular.z = 0.0
            self._avoiding = False  # yielding 중 장애물 회피 재진입 방지
            self.get_logger().info(
                f'YIELDING: d={leader_dist:.2f}, backing off',
                throttle_duration_sec=0.5
            )
            self.cmd_pub.publish(cmd)
            return

        # ── 초기 직접 추종 모드 ────────────────────────────────────────────
        if not self.use_breadcrumb:
            dx = lx - fx
            dy = ly - fy
            distance = leader_dist

            if len(self.leader_path) >= 10:
                # 전환 시 follower에서 가장 가까운 경로 점부터 시작
                path_list = list(self.leader_path)
                closest_idx = min(
                    range(len(path_list)),
                    key=lambda i: math.hypot(fx - path_list[i][0], fy - path_list[i][1])
                )
                self.waypoint_idx = closest_idx
                self.use_breadcrumb = True
                self.get_logger().info(
                    f'Switching to breadcrumb mode, wp_idx={self.waypoint_idx}'
                )

            # distance가 너무 작으면 atan2가 무의미 → 정지
            if distance < 0.05:
                self.cmd_pub.publish(cmd)
                return

            theta_target = math.atan2(dy, dx)
            e_theta = self.normalize_angle(theta_target - fyaw)

            cmd.angular.z = self.k_angular * e_theta
            cmd.angular.z = max(min(cmd.angular.z, self.max_angular), -self.max_angular)

            if distance > self.follow_distance:
                cmd.linear.x = min(
                    self.k_linear * distance,
                    self.max_linear
                )
            elif distance > 0.01:
                cmd.linear.x = min(self.k_linear * distance * 0.5, self.max_linear)
            else:
                cmd.linear.x = 0.0

            avoided, cmd = self.handle_obstacle(cmd, e_theta)
            if avoided:
                self.cmd_pub.publish(cmd)
                return

            self.get_logger().info(
                f'[DIRECT] d={distance:.2f}, e_theta={e_theta:.2f}, '
                f'v={cmd.linear.x:.2f}, w={cmd.angular.z:.2f}',
                throttle_duration_sec=1.0
            )
            self.cmd_pub.publish(cmd)
            return

        # ── 1. Waypoint 선택 및 e_theta 계산 ──────────────────────────────
        waypoint = self.select_waypoint(fx, fy)
        if waypoint is None:
            self.cmd_pub.publish(cmd)
            return

        wx, wy = waypoint
        dx = wx - fx
        dy = wy - fy
        distance = math.hypot(dx, dy)

        theta_target = math.atan2(dy, dx)
        e_theta = self.normalize_angle(theta_target - fyaw)

        # ── 2. 장애물 회피 (e_theta 계산 후 적용) ─────────────────────────
        avoided, cmd = self.handle_obstacle(cmd, e_theta)
        if avoided:
            self.cmd_pub.publish(cmd)
            return

        # ── 3. 히스테리시스 회전 모드 ─────────────────────────────────────
        if self._rotating:
            if abs(e_theta) < self.ROTATE_EXIT:
                self._rotating = False
        else:
            if abs(e_theta) > self.ROTATE_ENTER:
                self._rotating = True

        # ── 4. 각속도 ──────────────────────────────────────────────────────
        cmd.angular.z = self.k_angular * e_theta
        cmd.angular.z = max(min(cmd.angular.z, self.max_angular), -self.max_angular)

        # ── 5. 선속도 ──────────────────────────────────────────────────────
        if distance <= self.waypoint_reach_dist:
            cmd.linear.x = 0.0

        elif self._rotating:
            cmd.linear.x = 0.0

        else:
            cmd.linear.x = min(self.k_linear * distance, self.max_linear)

        self.get_logger().info(
            f'[BREAD] wp=({wx:.2f},{wy:.2f}), d={distance:.2f}, '
            f'e_theta={e_theta:.2f}, rotating={self._rotating}, '
            f'path_len={len(self.leader_path)}, wp_idx={self.waypoint_idx}, '
            f'v={cmd.linear.x:.2f}, w={cmd.angular.z:.2f}',
            throttle_duration_sec=1.0
        )
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = FollowerController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()