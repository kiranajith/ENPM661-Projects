#include "navigation_node.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"

using namespace std::chrono_literals;

const std::string bt_xml_dir =
    ament_index_cpp::get_package_share_directory("project5_navigation") + "/bt_xml";

NavigationNode::NavigationNode(const std::string &nodeName) : Node(nodeName)
{
  this->declare_parameter("location_x", 0.0f);
  this->declare_parameter("location_y", 0.0f);
  this->declare_parameter("location_theta", 0.0f);

  this->get_parameter("location_x", location_x_);
  this->get_parameter("location_y", location_y_);
  this->get_parameter("location_theta", location_theta_);
  path_subscriber_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/optimal_path", 10, std::bind(&NavigationNode::path_callback, this, std::placeholders::_1));
      

  RCLCPP_INFO(get_logger(), "Init done");
}

void NavigationNode::setup()
{
  RCLCPP_INFO(get_logger(), "Setting up");
  create_behavior_tree();
  RCLCPP_INFO(get_logger(), "BT created");

  const auto timer_period = 500ms;
  timer_ = this->create_wall_timer(
      timer_period,
      std::bind(&NavigationNode::update_behavior_tree, this));

  rclcpp::spin(shared_from_this());
  rclcpp::shutdown();
}

void NavigationNode::create_behavior_tree()
{
  BT::BehaviorTreeFactory factory;

  // register bt node

  BT::NodeBuilder builder =
      [=](const std::string &name, const BT::NodeConfiguration &config)
  {
    return std::make_unique<GoToPose>(name, config, shared_from_this());
  };

  factory.registerBuilder<GoToPose>("GoToPose", builder);

  RCLCPP_INFO(get_logger(), bt_xml_dir.c_str());

  tree_ = factory.createTreeFromFile(bt_xml_dir + "/tree.xml");
  RCLCPP_INFO(get_logger(), "3");
}

void NavigationNode::update_behavior_tree()
{
  if (!path_queue_.empty()) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    auto path_point = path_queue_.front();
    path_queue_.pop();

    // Convert geometry_msgs::msg::Quaternion to tf2::Quaternion
    tf2::Quaternion quat;
    tf2::fromMsg(path_point.pose.orientation, quat);

    // Use tf2::Matrix3x3 to get the yaw from the quaternion
    tf2::Matrix3x3 m(quat);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    this->set_parameter(rclcpp::Parameter("location_x", path_point.pose.position.x));
    this->set_parameter(rclcpp::Parameter("location_y", path_point.pose.position.y));
    this->set_parameter(rclcpp::Parameter("location_theta", yaw));

    location_x_ = path_point.pose.position.x;
    location_y_ = path_point.pose.position.y;
    location_theta_ = yaw;
  }

  BT::NodeStatus tree_status = tree_.tickRoot();

  if (tree_status == BT::NodeStatus::RUNNING)
  {
    return;
  }
  else if (tree_status == BT::NodeStatus::SUCCESS)
  {
    RCLCPP_INFO(this->get_logger(), "Finished Navigation");
  }
  else if (tree_status == BT::NodeStatus::FAILURE)
  {
    RCLCPP_INFO(this->get_logger(), "Navigation Failed");
    timer_->cancel();
  }
}

void NavigationNode::path_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(queue_mutex_);
  path_queue_.push(*msg);
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<NavigationNode>("navigation_node");
  node->setup();

  return 0;
}

