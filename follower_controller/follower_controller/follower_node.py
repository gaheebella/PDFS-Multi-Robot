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

        # Communication / following parameters
        self.comm_range = 2.0
        self.follow_distance = 0.25

        # Control gains
        self.k_linear = 2.0
        self.k_angular = 3.0

        # Velocity limits
        self.max_linear = 0.60
        self.max_angular = 3.0

        # Lost target search behavior
        self.search_angular_speed = 1.8

        # Hysteresis thresholds for rotation mode
        self.ROTATE_ENTER = 1.0
        self.ROTATE_EXIT  = 0.6
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

        # [수정] Leader 중심이 아닌 Leader 뒤쪽 목표점을 추종
        leader_yaw = self.quaternion_to_yaw(self.leader_pose.orientation)

        target_x = lx - self.follow_distance * math.cos(leader_yaw)
        target_y = ly - self.follow_distance * math.sin(leader_yaw)

        dx = target_x - fx
        dy = target_y - fy

        distance = math.sqrt(dx * dx + dy * dy)

        # 통신 범위는 Leader 중심 기준으로 판단
        dx_comm = lx - fx
        dy_comm = ly - fy
        comm_distance = math.sqrt(dx_comm * dx_comm + dy_comm * dy_comm)

        # Communication lost: rotate in place to search for leader
        if comm_distance > self.comm_range:
            cmd.linear.x = 0.0
            cmd.angular.z = self.search_angular_speed

            self.get_logger().info(
                f'OUT OF RANGE: comm_d={comm_distance:.2f} m -> searching leader'
            )

            self.cmd_pub.publish(cmd)
            return
        
        # 목표점에 너무 가까우면 bearing이 불안정하므로
        # leader 방향과 같은 방향으로만 정렬
        if distance < 0.12:
            heading_error = self.normalize_angle(leader_yaw - fyaw)

            cmd.linear.x = 0.0
            cmd.angular.z = self.k_angular * heading_error
            cmd.angular.z = max(min(cmd.angular.z, self.max_angular), -self.max_angular)

            self.get_logger().info(
                f'AT TARGET: align with leader, heading_error={heading_error:.2f}, '
                f'w={cmd.angular.z:.2f}'
            )

            self.cmd_pub.publish(cmd)
            return


        theta_target = math.atan2(dy, dx)
        e_theta = self.normalize_angle(theta_target - fyaw)

        # [수정] 히스테리시스로 chattering 방지
        if self._rotating:
            if abs(e_theta) < self.ROTATE_EXIT:
                self._rotating = False
        else:
            if abs(e_theta) > self.ROTATE_ENTER:
                self._rotating = True

        if self._rotating:
            cmd.linear.x = 0.0
            cmd.angular.z = self.search_angular_speed if e_theta > 0 else -self.search_angular_speed

            self.get_logger().info(
                f'ROTATING: d={distance:.2f}, e_theta={e_theta:.2f}'
            )

            self.cmd_pub.publish(cmd)
            return

        cmd.angular.z = self.k_angular * e_theta
        cmd.angular.z = max(min(cmd.angular.z, self.max_angular), -self.max_angular)

        # [수정] 목표점까지 거리 기준으로 선속도 계산 (임계값 0.05m)
        if distance > 0.05:
            angle_factor = max(0.2, math.cos(e_theta))

            cmd.linear.x = self.k_linear * distance * angle_factor
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