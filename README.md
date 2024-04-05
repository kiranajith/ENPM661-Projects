# ENPM661 Project 3 
## Phase 2 : Implementation of A* algorithm on a turtlebot3 

### Team Members
* Wei-Li Su (weisu77@umd.edu)\
UID: 117525298
* Kiran Ajith (kiran99@umd.edu)\
UID: 119112397 

### Problem 
Given a 2D map with obstacles, the task is to perform A* algorithm on a mobile robot to find an optimal path between the start node and the goal node.

### Map
* Given map 
<img width="1005" alt="Screenshot 2024-03-09 at 12 41 08 PM" src="https://github.com/kiranajith/ENPM661/assets/63303617/2a6285c5-2b25-41c2-8151-a67b505667f8">

* using half-planes and semi-algebraic models, the following map with obstacles has been modelled
<img width="1191" alt="Screenshot 2024-03-09 at 1 20 33 PM" src="https://github.com/kiranajith/ENPM661/assets/63303617/f468db5c-0c36-43d7-b0dd-88ee546a5269">

* here, in the canvas, the region with black pixels is the free space, the region with white pixels is the obstacles and the region surrounding the obstacles with red pixels is the clearance added to the obstacle. In this scenario, a clearance of 5mm is given.


## Libraries used  
* numpy
* heapq
* cv2
* time
* math 

## Part 2
## Instructions to run the simulation 
1. Create a ROS2 workspace
* Go to your home directory and create a workspace to add the ROS2 packages
```
mkdir -p projeect3ws>/src
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
