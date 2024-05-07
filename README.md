# ENPM661 Project 5
## Implementation of RRT* algorithm on a Turtlebot3 

### Team Members
* Wei-Li Su (weisu77@umd.edu)\
UID: 117525298
* Kiran Ajith (kiran99@umd.edu)\
UID: 119112397 

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
3. import the turtlebot3 package with vsc
```
cd ~/your_ws/src
vcs import . < turtlebot3.repos
```
4. build all the packages 
```
cd ~/your_ws
colcon build
```
5. source the workspace 
```
source install/setup.bash
```
6. export the turtlebot3 model 
```
export TURTLEBOT3_MODEL=waffle
```
7. Launch the gazebo simulation
```
ros2 launch project5_pkg turtlebot3_world.launch.py 
```
8. Launch the localization node
```
ros2 launch project5_pkg localization.launch.py     
```
9.  Run the navigation node 
```
ros2 launch project5_navigation navigation.launch.py
```
10. Run the RRT* node
```
ros2 run project5_pkg rrt_star                      
```

