#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class PathPlannerNode(Node):

    def __init__(self):
        super().__init__('path_planner')
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.5,self.timer_callback)


    def timer_callback(self):
        velocity_msg = Twist()
        velocity_msg.linear.x = 1.0
        self.velocity_publisher.publish(velocity_msg)




def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()