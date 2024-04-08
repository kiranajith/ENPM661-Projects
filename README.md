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
mkdir -p project3_ws/src
cd ~/project3_ws/src
```

2. clone the  trurtlebot repository 
```
git clone https://github.com/shantanuparabumd/turtlebot3_project3.git
```
3. clone the project3 repository 
```
git clone https://github.com/kiranajith/ENPM661-Projects.git
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
ros2 launch turtlebot3_project3 competition_world.launch.py x_pose:=0.0 y_pose:=-0.50
```
This command launches the turtlebot with it's spawn loctaion at (0.0,-0.5) wrt to the origin of the gazebo coordinate frame.

8. Run the path planner node 
```
ros2 run project3_pkg project3
```
Please note that the start loctaion in the node is (500,500), corresponding to (0,-0.5) according to the gazebo frame.Hence the path to goal state would be with respect to this point. 
If you want to change the spawn location of the turtebot, make sure to change the *initial_state* value in the main function of the node   


### Test case 1
```
Map Size: height * width: 2000mm * 6000mm
Goal x coordinate in mm: 3200
Goal y coordinate in mm: 1000
Node exploration started
Finding the path...
Solved!!
Total parent:  236
Total Number of Nodes Explored =  235
```

### Test case 2
```
Map Size: height * width: 2000mm * 6000mm
Goal x coordinate in mm: 5750
Goal y coordinate in mm: 900
Node exploration started
Finding the path...
Solved!!
Total parent:  358
Total Number of Nodes Explored =  357
```

