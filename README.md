# ENPM661 Project 3
## Implementation of A* algorithm on a mobile robot 

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
