#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class FollowerController(Node):
    def __init__(self):
        super().__init__('follower_controller')

        self.leader_pose = None
        self.follower_pose = None

        # 파라미터
        self.comm_range = 2.0
        self.follow_distance = 0.45
        self.k_linear = 0.8
        self.k_angular = 3.0
        self.max_linear = 0.20
        self.max_angular = 2.5
        self.search_angular_speed = 1.8

        # 히스테리시스 임계값 (chattering 방지)
        self.ROTATE_ENTER = 0.5   # 이 각도 오차 초과 시 회전 모드 진입
        self.ROTATE_EXIT = 0.25   # 이 각도 오차 미만 시 회전 모드 탈출
        self._rotating = False

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

    def leader_callback(self, msg):
        self.leader_pose = msg.pose.pose

    def follower_callback(self, msg):
        self.follower_pose = msg.pose.pose

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

    def control_loop(self):
        cmd = Twist()

        if self.leader_pose is None or self.follower_pose is None:
            self.cmd_pub.publish(cmd)
            return

        lx = self.leader_pose.position.x
        ly = self.leader_pose.position.y
        fx = self.follower_pose.position.x
        fy = self.follower_pose.position.y
        fyaw = self.quaternion_to_yaw(self.follower_pose.orientation)

        dx = lx - fx
        dy = ly - fy
        distance = math.sqrt(dx * dx + dy * dy)

        # 통신 범위 밖이면 제자리 회전 탐색
        if distance > self.comm_range:
            cmd.linear.x = 0.0
            cmd.angular.z = self.search_angular_speed
            self.get_logger().info(
                f'OUT OF RANGE: d={distance:.2f} -> searching',
                throttle_duration_sec=1.0
            )
            self._rotating = False  # 탐색 모드 진입 시 회전 상태 초기화
            self.cmd_pub.publish(cmd)
            return

        # Leader 방향 계산
        theta_target = math.atan2(dy, dx)
        e_theta = self.normalize_angle(theta_target - fyaw)

        # 히스테리시스로 회전 모드 전환 (chattering 방지)
        if self._rotating:
            if abs(e_theta) < self.ROTATE_EXIT:
                self._rotating = False
        else:
            if abs(e_theta) > self.ROTATE_ENTER:
                self._rotating = True

        # 항상 leader 방향으로 각속도 계산
        cmd.angular.z = self.k_angular * e_theta
        cmd.angular.z = max(min(cmd.angular.z, self.max_angular), -self.max_angular)

        # 목표 거리 이하: 선속도만 정지, leader 방향은 계속 추적
        if distance <= self.follow_distance:
            cmd.linear.x = 0.0

        # 회전 모드: 제자리 회전만
        elif self._rotating:
            cmd.linear.x = 0.0

        # 전진 모드: 방향 오차에 따라 속도 감쇄
        else:
            angle_factor = max(0.3, math.cos(e_theta))
            cmd.linear.x = self.k_linear * max(0.0, distance - self.follow_distance) * angle_factor
            cmd.linear.x = min(cmd.linear.x, self.max_linear)

        self.get_logger().info(
            f'd={distance:.2f}, bearing={theta_target:.2f}, '
            f'follower_yaw={fyaw:.2f}, e_theta={e_theta:.2f}, '
            f'rotating={self._rotating}, '
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