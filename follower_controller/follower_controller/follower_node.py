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

        # 제어 파라미터
        self.comm_range = 2.0
        self.follow_distance = 0.45
        self.k_linear = 0.5
        self.k_angular = 2.0
        self.max_linear = 0.15
        self.max_angular = 1.5
        self.search_angular_speed = 1.8

        # 히스테리시스 임계값 (chattering 방지)
        self.ROTATE_ENTER = 0.5
        self.ROTATE_EXIT = 0.25
        self._rotating = False

        # Breadcrumb path 파라미터
        self.path_record_min_dist = 0.10
        self.max_path_length = 2000
        self.waypoint_reach_dist = 0.15
        self.lookahead_steps = 6

        # 장애물 회피 파라미터
        self.obstacle_stop_dist = 0.55
        self.obstacle_turn_speed = 1.2

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
            if math.hypot(lx - last_x, ly - last_y) < self.path_record_min_dist:
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

    # ------------------------------------------------------------------ #
    #  Waypoint 선택 (단방향 전진 + lookahead_steps)
    # ------------------------------------------------------------------ #

    def select_waypoint(self, fx, fy):
        path = list(self.leader_path)
        n = len(path)

        if n == 0:
            return None

        self.waypoint_idx = min(self.waypoint_idx, n - 1)

        while self.waypoint_idx < n - 1:
            wx, wy = path[self.waypoint_idx]
            if math.hypot(fx - wx, fy - wy) < self.waypoint_reach_dist:
                self.waypoint_idx += 1
            else:
                break

        target_idx = min(self.waypoint_idx + self.lookahead_steps, n - 1)
        return path[target_idx]

    # ------------------------------------------------------------------ #
    #  메인 제어 루프
    # ------------------------------------------------------------------ #

    def control_loop(self):
        cmd = Twist()

        if self.leader_pose is None or self.follower_pose is None:
            self.cmd_pub.publish(cmd)
            return

        fx = self.follower_pose.position.x
        fy = self.follower_pose.position.y
        fyaw = self.quaternion_to_yaw(self.follower_pose.orientation)

        # 통신 범위 판단 (Leader 현재 위치 기준)
        lx = self.leader_pose.position.x
        ly = self.leader_pose.position.y
        leader_dist = math.hypot(lx - fx, ly - fy)

        if leader_dist > self.comm_range:
            cmd.linear.x = 0.0
            cmd.angular.z = self.search_angular_speed
            self._rotating = False
            self.get_logger().info(
                f'OUT OF RANGE: d={leader_dist:.2f} -> searching',
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

        # ── 2. 장애물 회피 (e_theta 기반 회전 방향) ───────────────────────
        if self.scan_msg is not None:
            ranges = list(self.scan_msg.ranges)
            n = len(ranges)

            front = ranges[0:45] + ranges[n - 45:]
            left  = ranges[45:120]
            right = ranges[n - 120: n - 45]

            front_min = min((r for r in front if math.isfinite(r)), default=10.0)
            left_min  = min((r for r in left  if math.isfinite(r)), default=10.0)
            right_min = min((r for r in right if math.isfinite(r)), default=10.0)

            side_stop_dist = 0.25

            if left_min < side_stop_dist:
                cmd.linear.x = 0.0
                cmd.angular.z = -self.obstacle_turn_speed

                self.get_logger().info(
                    f'TOO CLOSE LEFT: left={left_min:.2f}'
                )

                self.cmd_pub.publish(cmd)
                return

            if right_min < side_stop_dist:
                cmd.linear.x = 0.0
                cmd.angular.z = self.obstacle_turn_speed

                self.get_logger().info(
                    f'TOO CLOSE RIGHT: right={right_min:.2f}'
                )

                self.cmd_pub.publish(cmd)
                return

            if front_min < self.obstacle_stop_dist:
                cmd.linear.x = 0.0
                # 추종 목표 방향(e_theta)으로 회전 → 회피 후 자연스럽게 추종으로 복귀
                cmd.angular.z = (
                    self.obstacle_turn_speed if e_theta >= 0
                    else -self.obstacle_turn_speed
                )
                self.get_logger().info(
                    f'OBSTACLE: front={front_min:.2f}, '
                    f'left={left_min:.2f}, right={right_min:.2f}, '
                    f'turn={"L" if e_theta >= 0 else "R"}',
                    throttle_duration_sec=0.5
                )
                self.cmd_pub.publish(cmd)
                return

        # ── 3. 히스테리시스 회전 모드 전환 ────────────────────────────────
        if self._rotating:
            if abs(e_theta) < self.ROTATE_EXIT:
                self._rotating = False
        else:
            if abs(e_theta) > self.ROTATE_ENTER:
                self._rotating = True

        # ── 4. 각속도 (항상 waypoint 방향 추적) ───────────────────────────
        cmd.angular.z = self.k_angular * e_theta
        cmd.angular.z = max(min(cmd.angular.z, self.max_angular), -self.max_angular)

        # ── 5. 선속도 결정 ─────────────────────────────────────────────────
        if distance <= self.waypoint_reach_dist:
            cmd.linear.x = 0.0                          # 도달: 방향 추적만 유지

        elif self._rotating:
            cmd.linear.x = 0.0                          # 큰 각도 오차: 제자리 회전

        else:
            cmd.linear.x = min(self.k_linear * distance, self.max_linear)

        self.get_logger().info(
            f'wp=({wx:.2f},{wy:.2f}), d={distance:.2f}, '
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