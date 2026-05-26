#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class TopicRelay(Node):
    def __init__(self):
        super().__init__('topic_relay_node')

        self.leader_odom_pub = self.create_publisher(Odometry, '/leader_odom', 10)
        self.follower_odom_pub = self.create_publisher(Odometry, '/follower_odom', 10)
        self.follower_scan_pub = self.create_publisher(LaserScan, '/follower_scan', 10)

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.get_logger().info('Topic relay node started.')

    def odom_callback(self, msg):
        self.leader_odom_pub.publish(msg)
        self.follower_odom_pub.publish(msg)

    def scan_callback(self, msg):
        self.follower_scan_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TopicRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
