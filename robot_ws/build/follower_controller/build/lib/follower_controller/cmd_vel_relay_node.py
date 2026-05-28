#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdVelRelay(Node):
    def __init__(self):
        super().__init__('cmd_vel_relay_node')

        self.pub = self.create_publisher(Twist, '/follower/cmd_vel', 10)

        self.create_subscription(
            Twist,
            '/follower_cmd_vel',
            self.cmd_callback,
            10
        )

        self.get_logger().info('CmdVel relay node started: /follower_cmd_vel -> /follower/cmd_vel')

    def cmd_callback(self, msg):
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
