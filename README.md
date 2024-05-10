# ENPM661 Project 5
## Implementation of RRT* algorithm on a Turtlebot3 

### Team Members
* Wei-Li Su (weisu77@umd.edu)\
UID: 117525298
* Kiran Ajith (kiran99@umd.edu)\
UID: 119112397 

## Libraries used  
* numpy
* heapq
* cv2
* time
* math 
* imutils
* mutex
* queue
* transforms3d

## Instructions to run the code
1. clone this repository to the src folder in your ros2 workspace 
```
cd ~/your_ws
git clone https://github.com/kiranajith/ENPM661-Projects.git 
```
2. switch to the project 5 branch 
```
git checkout Project5
```

## Part A - visualization node exploration using python
1. run the rrt_star.py file to visualize RRT* 
```
python3 rrt_star.py
```
2. run the rrt.py file to visualize RRT.
```
python3 rrt.py
```
## Part B - Turtlebot3 simualtion in Gazebo 

1. import the turtlebot3 package with vsc
```
cd ~/your_ws/src
vcs import . < turtlebot3.repos
```
2. build all the packages 
```
cd ~/your_ws
colcon build
```
4. source the workspace 
```
source install/setup.bash
```
5. export the turtlebot3 model 
```
export TURTLEBOT3_MODEL=waffle
```
6. Launch the gazebo simulation
```
ros2 launch project5_pkg turtlebot3_world.launch.py 
```
7. Launch the localization node
```
ros2 launch project5_pkg localization.launch.py     
```
8.  Run the navigation node 
```
ros2 launch project5_navigation navigation.launch.py
```
9. Run the RRT* node
```
ros2 run project5_pkg rrt_star                      
```

