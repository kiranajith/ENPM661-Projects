# ENPM661 Project 3 
## Phase 2 : Implementation of A* algorithm on a turtlebot3 

### Team Members
* Wei-Li Su (weisu77@umd.edu)\
UID: 117525298
* Kiran Ajith (kiran99@umd.edu)\
UID: 119112397 

### Problem 
Given a turtlebot3 environment in Gazebo, the task is implement A* algorithm to have the turtlebot navigate to a given goal position


## Libraries used  
* numpy
* heapq
* cv2
* time
* math 
* imutils 

## Part 2
## Instructions to run the simulation 
1. Create a ROS2 workspace
* Go to your home directory and create a workspace to add the ROS2 packages
```
mkdir -p project3ws/src
cd ~/project3_ws/src
```

2. clone the  trurtlebot repository 
```
git clone https://github.com/shantanuparabumd/turtlebot3_project3.git
```
3. clone the project3 repository 
```
https://github.com/kiranajith/ENPM661-Projects/tree/Project-3-Phase-2
```
4. Source ROS 
```
source /opt/ros/galactic/setup.bash
```
5. Build the workspace 
```
cd ~\project3_ws
colcon build 
```
6. Source the workspace
``` 
source install/setup.bash
```
7. Launch the environment 
```
ros2 launch turtlebot3_project3 competition_world.launch
```
8. Run the path palnner node 
```
ros2 run project3_pkg project3
```
