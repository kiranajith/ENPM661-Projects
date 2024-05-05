import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
  planner_pkg = get_package_share_directory('project5_pkg')
  navigation_pkg = get_package_share_directory('project5_navigation')

  node = Node(
      package="project5_navigation",
      executable="project5_navigation",
      name="project5_node",
      parameters=[]
  )

  ld = LaunchDescription()
  ld.add_action(node)
  return ld
