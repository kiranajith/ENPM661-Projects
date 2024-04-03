#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

def timer_callback():
    global velocity_publisher
    velocity_msg = Twist()
    velocity_msg.linear.x = 1.0
    velocity_publisher.publish(velocity_msg)

def main():
    rclpy.init(args=None)
    node = Node('path_planner')
    global velocity_publisher
    velocity_publisher = node.create_publisher(Twist, 'cmd_vel', 10)
    timer = node.create_timer(0.5, timer_callback)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
