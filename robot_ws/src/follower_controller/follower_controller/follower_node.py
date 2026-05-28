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
        self._yield_state = 'follow'
        self._prev_leader_dist = None
        self._prev_leader_pos = None          # leader 이전 위치 (이동 방향 감지용)
        self.approach_detect_dist = 0.50      # 이 거리 이내에서 접근 감지
        self.approach_clear_dist  = 0.65      # 이 거리 이상 벌어지면 복귀
        self.backoff_speed = -0.10

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
        self.max_linear = 0.30            # burger 정격 속도 수준으로 복구
        self.max_angular = 1.5
        self.search_angular_speed = 1.0

        # 히스테리시스 임계값
        self.ROTATE_ENTER = 0.4
        self.ROTATE_EXIT = 0.15
        self._rotating = False

        # Breadcrumb path 파라미터
        self.path_record_min_dist = 0.03   # 촘촘하게 (burger 반경의 1/3)
        self.max_path_length = 3000
        self.waypoint_reach_dist = 0.08    # 도달 판정 거리
        self.lookahead_steps = 4

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

    def get_rear_min(self):
        """후방 ±30° 내 최소 거리 반환. scan 없으면 10.0"""
        if self.scan_msg is None:
            return 10.0
        ranges = self.scan_msg.ranges
        n = len(ranges)
        rear = list(ranges[n // 2 - 30: n // 2 + 30])
        return min((r for r in rear if math.isfinite(r)), default=10.0)

    def get_side_space(self):
        """좌우 공간 중 더 넓은 쪽 반환: ('L', dist) or ('R', dist)"""
        if self.scan_msg is None:
            return ('L', 10.0)
        ranges = self.scan_msg.ranges
        n = len(ranges)
        left  = list(ranges[60:120])
        right = list(ranges[n - 120: n - 60])
        left_min  = min((r for r in left  if math.isfinite(r)), default=10.0)
        right_min = min((r for r in right if math.isfinite(r)), default=10.0)
        if left_min >= right_min:
            return ('L', left_min)
        else:
            return ('R', right_min)

    def select_waypoint(self, fx, fy):
        path = list(self.leader_path)
        n = len(path)

        if n == 0:
            return None

        self.waypoint_idx = min(self.waypoint_idx, n - 1)

        # 현재 waypoint 도달 시 다음으로 전진 (단방향)
        while self.waypoint_idx < n - 1:
            wx, wy = path[self.waypoint_idx]
            if math.hypot(fx - wx, fy - wy) < self.waypoint_reach_dist:
                self.waypoint_idx += 1
            else:
                break

        # leader가 너무 멀리 가서 뒤처진 경우 강제 따라잡기
        # path 끝에서 lookahead_steps 이내로 맞춤
        lag = n - 1 - self.waypoint_idx
        if lag > 10:  # 10개 점(약 0.3m) 이상 뒤처지면 점프
            self.waypoint_idx = n - 1 - self.lookahead_steps

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

        # 통신 범위 밖: leader 방향으로 적극 추적
        if leader_dist > self.comm_range:
            self._rotating = False
            self._avoiding = False
            self._yield_state = 'follow'
            self._prev_leader_dist = None
            self._prev_leader_pos = None
            self.use_breadcrumb = False
            self.waypoint_idx = 0

            # odom으로 leader 방향은 알 수 있으므로
            # 방향 정렬 후 전진해서 통신 범위 안으로 복귀
            theta_target = math.atan2(ly - fy, lx - fx)
            e_theta = self.normalize_angle(theta_target - fyaw)

            cmd.angular.z = self.k_angular * e_theta
            cmd.angular.z = max(min(cmd.angular.z, self.max_angular), -self.max_angular)

            if abs(e_theta) < 0.3:
                # 방향 정렬됐으면 전진해서 따라잡기
                cmd.linear.x = self.max_linear
            else:
                # 아직 방향 안 맞으면 제자리 회전
                cmd.linear.x = 0.0

            # 전방에 장애물 있으면 전진 금지
            front_min = self.get_front_min()
            if front_min < self.obstacle_stop_dist:
                cmd.linear.x = 0.0

            self.get_logger().info(
                f'OUT OF RANGE: d={leader_dist:.2f}, '
                f'e_theta={e_theta:.2f}, '
                f'v={cmd.linear.x:.2f}, w={cmd.angular.z:.2f}',
                throttle_duration_sec=1.0
            )
            self.cmd_pub.publish(cmd)
            return

        # ── Leader 우선권 양보 상태머신 ───────────────────────────────────
        # leader가 follower 쪽으로 이동 중일 때만 yielding 진입
        # (정상 추종 중 가까워지는 건 제외)
        #
        # 판단 방법:
        # leader 이동 방향 벡터와 "leader → follower" 방향의 내적
        # 내적 > 0 : leader가 follower 쪽으로 오는 중
        # 내적 < 0 : leader가 follower에서 멀어지는 중 (정상 추종)

        # leader 이동 벡터 계산 (이전 위치 대비)
        if self._prev_leader_pos is not None:
            leader_vel_x = lx - self._prev_leader_pos[0]
            leader_vel_y = ly - self._prev_leader_pos[1]
            # follower 방향 단위벡터 (leader → follower)
            to_f_x = fx - lx
            to_f_y = fy - ly
            dist_lf = math.hypot(to_f_x, to_f_y)
            if dist_lf > 1e-6:
                to_f_x /= dist_lf
                to_f_y /= dist_lf
            dot = leader_vel_x * to_f_x + leader_vel_y * to_f_y
            leader_coming = (dot > 0.001 and leader_dist < self.approach_detect_dist)
        else:
            leader_coming = False
        self._prev_leader_pos = (lx, ly)

        if self._yield_state == 'follow':
            if leader_coming:
                self._yield_state = 'yielding'
                self._avoiding = False
                self.get_logger().info(
                    f'YIELD START: d={leader_dist:.2f}, dot={dot:.4f}'
                )
        elif self._yield_state == 'yielding':
            # leader가 멀어지고 충분히 벌어지면 복귀
            if leader_dist > self.approach_clear_dist:
                self._yield_state = 'follow'
                self.get_logger().info(
                    f'YIELD END: d={leader_dist:.2f}, resuming follow'
                )

        if self._yield_state == 'yielding':
            self._avoiding = False
            rear_min = self.get_rear_min()

            if rear_min > 0.25:
                # 뒤에 공간 있으면 후진
                cmd.linear.x = self.backoff_speed
                cmd.angular.z = 0.0
                self.get_logger().info(
                    f'YIELDING backoff: d={leader_dist:.2f}, rear={rear_min:.2f}',
                    throttle_duration_sec=0.5
                )
            else:
                # 뒤에 벽 있으면 옆으로 비켜서 leader가 지나갈 공간 확보
                side, side_dist = self.get_side_space()
                turn_dir = 1.0 if side == 'L' else -1.0
                cmd.linear.x = 0.0
                cmd.angular.z = turn_dir * self.obstacle_turn_speed
                self.get_logger().info(
                    f'YIELDING dodge: d={leader_dist:.2f}, rear={rear_min:.2f}, '
                    f'dodge={side}({side_dist:.2f})',
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
                #self.use_breadcrumb = True
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

            if self._yield_state != 'yielding':
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

        # ── 2. 장애물 회피 (yielding 중에는 건너뜀) ──────────────────────
        if self._yield_state != 'yielding':
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
            f'leader_d={leader_dist:.2f}, yield={self._yield_state}, '
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
