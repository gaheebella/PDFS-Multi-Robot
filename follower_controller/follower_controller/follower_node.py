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

        self.comm_range = 2.0
        self.follow_distance = 0.25      # 목표 거리 살짝 줄여 여유 확보

        self.k_linear = 2.0             # 선속도 게인 증가
        self.k_angular = 3.0            # 각도 게인 감소 → 진동 방지

        self.max_linear = 0.60          # 실내 테스트용 안전 속도
        self.max_angular = 3.0

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

        if distance > self.comm_range:
            self.get_logger().info(
                f'OUT OF RANGE: distance={distance:.2f} m'
            )
            self.cmd_pub.publish(cmd)
            return

        theta_target = math.atan2(dy, dx)
        e_theta = self.normalize_angle(theta_target - fyaw)

        cmd.angular.z = self.k_angular * e_theta
        cmd.angular.z = max(min(cmd.angular.z, self.max_angular), -self.max_angular)

        # [수정] 각도 오차에 따라 선속도를 연속적으로 감쇠
        # 기존: abs(e_theta) < 0.4 일 때만 전진 → 조건 미충족 시 영원히 제자리 회전
        # 수정: angle_factor로 부드럽게 감쇠 (정면일수록 빠르게, 옆면일수록 느리게)
        if distance > self.follow_distance:
            angle_factor = max(0.4, math.cos(e_theta))
            cmd.linear.x = self.k_linear * (distance - self.follow_distance) * angle_factor
            cmd.linear.x = min(cmd.linear.x, self.max_linear)
        else:
            cmd.linear.x = 0.0

        self.get_logger().info(
            f'd={distance:.2f}, bearing={theta_target:.2f}, '
            f'follower_yaw={fyaw:.2f}, e_theta={e_theta:.2f}, '
            f'v={cmd.linear.x:.2f}, w={cmd.angular.z:.2f}'
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