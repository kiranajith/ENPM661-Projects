# ENPM661 Project 3
## Implementation of A* algorithm on a mobile robot 

### Problem 
Given a 2D map with obstacles, the task is to perform A* algorithm on a mobile robot to find an optimal path between the start node and the goal node.

### Map
* Given map 
<img width="1005" alt="Screenshot 2024-03-09 at 12 41 08 PM" src="https://github.com/kiranajith/ENPM661/assets/63303617/2a6285c5-2b25-41c2-8151-a67b505667f8">



## Libraries used  
* numpy
* heapq
* cv2
* time
* math 

## Instructions to run the program 
run the following command in the terminal 
```
python3 proj3_weili_kiran.py
```
Alternatively, run the proj3_weili_kiran.py in the IDE of your choice

## Input Instructions
* when the program is run, the user will be prompted to input the x and y coordinates of the start and goal nodes and the initial and final orientation of the robot(permitted angles of orientation are: 0,-30,-60,+30,+60)

## Visualizing the output
A window pops up showing the explored nodes and finally the generated path
