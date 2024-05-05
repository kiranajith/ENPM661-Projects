#include "rclcpp/rclcpp.hpp"
#include "behaviortree_cpp_v3/bt_factory.h"
#include "navigation_behaviors.h"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <queue>
#include <mutex>

#include <tf2/LinearMath/Quaternion.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "tf2/LinearMath/Matrix3x3.h"  


class NavigationNode : public rclcpp::Node
{
public:
  explicit NavigationNode(const std::string &node_name);
  void setup();
  void create_behavior_tree();
  void update_behavior_tree();
  void path_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);  

private:
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr path_subscriber_;  
  BT::Tree tree_;
  float location_x_;
  float location_y_;
  float location_theta_;
  std::queue<geometry_msgs::msg::PoseStamped> path_queue_;
  std::mutex queue_mutex_;
};