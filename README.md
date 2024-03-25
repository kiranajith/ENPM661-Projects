# ENPM661 Project 3
## Phase 1 : Implementation of A* algorithm on a mobile robot 

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

## Instructions to run the program 
run the following command in the terminal 
```
python3 a_star_weili_kiran.py
```
Alternatively, run the a_star_weili_kiran.py in the IDE of your choice

## Input Instructions
* when the program is run, the user will be prompted to input the x and y coordinates of the start and goal nodes and the initial and final orientation of the robot(permitted angles of orientation are: 0,-30,-60,+30,+60)

## Visualizing the output
A window pops up showing the explored nodes and finally the generated path. A video file will also be created which shows the same.
![Screenshot 2024-03-24 at 10 54 30 PM](https://github.com/kiranajith/ENPM661-Projects/assets/63303617/00957b11-dd82-4daf-87ab-43a588e32e32)

### Sample test case
#### input
Enter the clearance:5\
Enter the radius:5\
 Start node X : 10\
 Start node Y : 10\
Goal node X : 200\
Goal node Y : 200\
Enter the initial angle of orientation in degree(+-30)0\
Enter the goal angle of orientation in degree(+-30)60\
Enter the step size (1-10): 10\
[20, 479, 0] [400, 99, 60] [20, 479, 0] [400, 99, 60]\
dis:  537\
Node exploration started\
Finding the path...\
Solved!!\
Total Number of Nodes Explored =  3733\
Total run time : 79.78 sec

## Node exploration for the sample test case
https://github.com/kiranajith/ENPM661-Projects/assets/63303617/d6d4c453-dba5-4e37-8a42-dd5add62fa53
