#!/usr/bin/env python3
import math
import collections
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class FollowerController(Node):
    def __init__(self):
        super().__init__('follower_controller')

        self.leader_pose = None
        self.follower_pose = None

        # 제어 파라미터
        self.comm_range = 2.0
        self.follow_distance = 0.45
        self.k_linear = 0.6
        self.k_angular = 3.0
        self.max_linear = 0.18
        self.max_angular = 2.5
        self.search_angular_speed = 1.8

        # 히스테리시스 임계값 (chattering 방지)
        self.ROTATE_ENTER = 0.5
        self.ROTATE_EXIT = 0.25
        self._rotating = False

        # Breadcrumb path 파라미터
        # Leader가 최소 이 거리 이상 이동했을 때만 경로에 점 추가
        self.path_record_min_dist = 0.05
        # 최대 저장 waypoint 수 (메모리 제한)
        self.max_path_length = 2000
        # Follower가 waypoint에 이 거리 이내로 접근하면 다음 waypoint로 넘어감
        self.waypoint_reach_dist = 0.10
        # Lookahead: follower 전방으로 이 거리만큼 앞선 waypoint를 목표로 삼음
        self.lookahead_dist = 0.30

        # Leader 경로 저장 덱 (오래된 점은 앞에서 제거)
        self.leader_path = collections.deque(maxlen=self.max_path_length)
        # Follower가 현재 추종 중인 waypoint 인덱스
        self.waypoint_idx = 0

        self.leader_sub = self.create_subscription(
            Odometry,
            '/leader_odom',
            self.leader_callback,
            10
        )
        self.follower_sub = self.create_subscription(
            Odometry,
            '/follower_odom',
            self.follower_callback,
            10
        )
        self.cmd_pub = self.create_publisher(
            Twist,
            '/follower_cmd_vel',
            10
        )
        self.timer = self.create_timer(0.1, self.control_loop)

    # ------------------------------------------------------------------ #
    #  콜백
    # ------------------------------------------------------------------ #

    def leader_callback(self, msg):
        self.leader_pose = msg.pose.pose
        lx = msg.pose.pose.position.x
        ly = msg.pose.pose.position.y

        # 마지막 기록 점과 충분히 멀어졌을 때만 추가
        if self.leader_path:
            last_x, last_y = self.leader_path[-1]
            if math.hypot(lx - last_x, ly - last_y) < self.path_record_min_dist:
                return

        self.leader_path.append((lx, ly))

    def follower_callback(self, msg):
        self.follower_pose = msg.pose.pose

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
    #  Waypoint 선택 (lookahead 기반)
    # ------------------------------------------------------------------ #

    def select_waypoint(self, fx, fy):
        """
        Leader_path 중에서 follower가 추종할 waypoint를 선택한다.

        전략:
        1. 이미 지나친 waypoint는 건너뜀 (waypoint_idx 전진).
        2. waypoint_idx 이후 점들 중 lookahead_dist 이상 앞에 있는
           가장 가까운 점을 목표로 삼는다.
        3. 그런 점이 없으면 path 끝점을 목표로 삼는다.
        """
        path = self.leader_path
        n = len(path)

        if n == 0:
            return None

        # deque를 리스트로 슬라이싱 (인덱스 접근 편의)
        path_list = list(path)

        # waypoint_idx 범위 보정
        self.waypoint_idx = min(self.waypoint_idx, n - 1)

        # 이미 도달한 waypoint 건너뜀
        while self.waypoint_idx < n - 1:
            wx, wy = path_list[self.waypoint_idx]
            if math.hypot(fx - wx, fy - wy) < self.waypoint_reach_dist:
                self.waypoint_idx += 1
            else:
                break

        # lookahead: waypoint_idx 이후에서 lookahead_dist 이상 앞에 있는 첫 점
        for i in range(self.waypoint_idx, n):
            wx, wy = path_list[i]
            if math.hypot(fx - wx, fy - wy) >= self.lookahead_dist:
                return (wx, wy)

        # lookahead 점이 없으면 경로 마지막 점
        return path_list[-1]

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

        # Leader 현재 위치 (통신 범위 판단용)
        lx = self.leader_pose.position.x
        ly = self.leader_pose.position.y
        leader_dist = math.hypot(lx - fx, ly - fy)

        # 통신 범위 밖이면 제자리 회전 탐색
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

        # 따라갈 waypoint 선택
        waypoint = self.select_waypoint(fx, fy)

        if waypoint is None:
            # 경로가 아직 없으면 정지 대기
            self.cmd_pub.publish(cmd)
            return

        wx, wy = waypoint
        dx = wx - fx
        dy = wy - fy
        distance = math.hypot(dx, dy)

        # Waypoint 방향 및 각도 오차
        theta_target = math.atan2(dy, dx)
        e_theta = self.normalize_angle(theta_target - fyaw)

        # 히스테리시스 회전 모드 전환
        if self._rotating:
            if abs(e_theta) < self.ROTATE_EXIT:
                self._rotating = False
        else:
            if abs(e_theta) > self.ROTATE_ENTER:
                self._rotating = True

        # 각속도 계산 (항상 waypoint 방향 추적)
        cmd.angular.z = self.k_angular * e_theta
        cmd.angular.z = max(min(cmd.angular.z, self.max_angular), -self.max_angular)

        # 선속도 결정
        if distance <= self.waypoint_reach_dist:
            # Waypoint 도달: 선속도 정지, 방향 추적은 유지
            cmd.linear.x = 0.0

        elif self._rotating:
            # 방향 오차가 클 때: 제자리 회전
            cmd.linear.x = 0.0

        else:
            # 전진: 방향 오차에 따라 속도 감쇄
            angle_factor = max(0.3, math.cos(e_theta))
            cmd.linear.x = self.k_linear * distance * angle_factor
            cmd.linear.x = min(cmd.linear.x, self.max_linear)

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