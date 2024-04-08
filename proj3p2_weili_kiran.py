import cv2
import numpy as np
import heapq as hq
import time
import math
import imutils


# Global variable
g_angle = 30
g_action_set_number = 5
g_total_degree = 360
g_matrix_threshold = 0.1
g_scaling = int(1/g_matrix_threshold)
g_goal_threshold = 10 * g_scaling
g_canvas_height = 200
g_canvas_width = 600
g_scaling_canvas_height = g_canvas_height * g_scaling
g_scaling_canvas_width = g_canvas_width * g_scaling
g_initial_parent = -1
g_weighted_a_star = 4
g_dt = 0.2

# Turtlebot3 waffle spec with unit mm
g_robot_radius = 220
g_wheel_radius = 33
g_wheel_distance = 287

def is_within_obstacle(canvas, new_height, new_width):
    """
    Checks if the given position is within an obstacle or the clearance zone.
    """
    # Check if within obstacle or clearance zone (red pixels indicating clearance)
    pixel = canvas[int(round(new_height))][int(round(new_width))]
    if pixel[0] == 255 or pixel[2] == 255:  # Obstacle (white) or clearance (red)
        return False
    else:
        return True

def draw_obstacles(canvas, robot_radius, clearance):   
    """ this function is used to pbstacles in the map
    the obstacles are marked in white pixels

    Args:
        canvas : the empty/map on which obstacles are to be drwan 

    Returns:
        : canvas with obstacles
    """
    offset = robot_radius + clearance  # Total enlargement of obstacles
    height, width, _ = canvas.shape 
    height,width,_ = canvas.shape 
    # print('Dimensions of the canvas')
    # print('------------------------')
    # print(f"height * width: {height} * {width}")
    for i in range(width): # traverse through the width of the canvas 
        for j in range(height): # traverse through the height of the canvas
            # model the left-most rectangle
            # ----- offset -----------
            if(i-1500+offset>=0 and i-1750-offset<=0 and height-j-1000+offset>=0 and height-j-2000-offset<0):
                canvas[j][i] = [0,0,255] 
            # model the 2nd rectangle
            if(i-2500+offset>=0 and i-2750-offset<=0 and height-j+offset>=0 and height-j-1000-offset<=0):
                canvas[j][i] = [0,0,255]

            # model the circle
            if (i - 4200 ) ** 2 + (j - 800) ** 2 - (600 + offset)**2 <= 0:
                canvas[j][i] = [0,0,255]
            # model the hexagon 
            # if(i+offset>=500 and i-offset<=800) and (j-offset<=(0.5*i)+75) and (j+offset>=(0.5*i)-225) and  (j-offset<=(-0.5*i)+725) and (j+offset>=(-0.5*i)+425): 
            #     canvas[j][i] = [0,0,255] 

            # --------- obstacle space --------
            if(i-1500>=0 and i-1750<=0 and height-j-1000>=0 and height-j-2000<0):
                canvas[j][i] = [255,255,255] 
            # model the 2nd rectangle
            if(i-2500>=0 and i-2750<=0 and height-j>=0 and height-j-1000<=0):
                canvas[j][i] = [255,255,255]

            # model the circle
            if (i - 4200) ** 2 + (j - 800) ** 2 - 600**2 <= 0:
                canvas[j][i] = [255,255,255]
            # model the hexagon 
            # if(i>=500 and i<=800) and (j<=(0.5*i)+75) and (j>=(0.5*i)-225) and  (j<=(-0.5*i)+725) and (j>=(-0.5*i)+425): 
            #     canvas[j][i] = [255,255,255]           
    return canvas


def draw_obstacles_with_clearance(canvas, clearance):
    """ this function is used to draw a clearance area around the obstacles
    it is shown as a red pixalated area around each object 

    Args:
        canvas : the map on which obstacles are to be drawn 
        clearance : the amount by which the obstacles are bloated 

    Returns:
         canvas with bloated obstacles 
    """
    canvas = draw_obstacles(canvas)  
    kernel = np.ones((clearance*2+1, clearance*2+1), np.uint8)
    red_zone = cv2.dilate(canvas[:, :, 0], kernel, iterations=1)  
    for i in range(canvas.shape[1]):
        for j in range(canvas.shape[0]):
            if red_zone[j, i] == 255 and canvas[j, i, 0] != 255:
                canvas[j, i] = [0, 0, 255]  # Red color

    return canvas


#------------- ACTION FUNCTIONS ---------------

def threshold(n):
    res = round(n*g_scaling)
    return res
'''
def action_rotate_zero_degrees(node, canvas, visited, step):    
    """ rotates the robot 30 degrees counterclockwise 

    Args:
        node : current node 
        canvas : canvas 
        visited : visited nodes
        step : step size

    Returns:
        bool: feasibility of the action and the new node 
    """
    new_node = node.copy() # copy the current node avoid modifying the original
    new_angle = new_node[2] + 0    # add angle of rotation, 0 here 

    if new_angle < 0: # check for valid rotation angles 
        new_angle += 360 
    new_angle %= 360

    new_width = new_node[0] + threshold(step*np.cos(np.deg2rad(new_angle)))
    new_height = new_node[1] + threshold(step*np.sin(np.deg2rad(new_angle)))
    # print('h, w:',new_height,new_width)

    # if (round(new_height)>0 and round(new_height)<canvas.shape[0]) and (round(new_width)>0 and round(new_width)<canvas.shape[1]) and  (canvas[int(round(new_height))][int(round(new_width))][0]!=255)   :
    if (0 < round(new_height) < canvas.shape[0]) and \
        (0 < round(new_width) < canvas.shape[1]) and \
        is_within_obstacle(canvas,new_height,new_width):
        new_node[0] = new_width
        new_node[1] = new_height
        new_node[2] = new_angle
        # print(new_node)

        if(visited[new_height][new_width][new_angle//g_angle] == 1):
            return True, new_node, True
        elif(visited[new_height][new_width][new_angle//g_angle] == 2):
            return False, new_node, False
        else: # mark the node as visited 
            visited[new_height][new_width][new_angle//g_angle] = 1
            return True, new_node, False
    else: # invalid action, new position is either in the obstacle space or outside the map 
        return False, new_node, False
    


def action_rotate_negative_thirty_degrees(node, canvas, visited, step):   
    """ rotates the robot 30 degrees clockwise 

    Args:
        node : current node 
        canvas : canvas 
        visited : visited nodes
        step : step size

    Returns:
        bool: feasibility of the action and the new node 
    """
    new_node= node.copy()
    new_angle = new_node[2] + 30    
    
    if new_angle < 0:
        new_angle += 360 
    new_angle %= 360
    new_width = new_node[0] + threshold(step * np.cos(np.deg2rad(new_angle)))
    new_height = new_node[1] + threshold(step * np.sin(np.deg2rad(new_angle)))

    if (round(new_height)>0 and round(new_height)<canvas.shape[0]) and \
        (round(new_width)>0 and round(new_width)<canvas.shape[1]) and \
        is_within_obstacle(canvas,new_height,new_width) :
        new_node[0] = new_width
        new_node[1] = new_height
        new_node[2] = new_angle
        
        if(visited[new_height][new_width][new_angle//g_angle] == 1):
            return True, new_node, True
        elif(visited[new_height][new_width][new_angle//g_angle] == 2):
            return False, new_node, False
        else:
            visited[new_height][new_width][new_angle//g_angle] = 1
            return True, new_node, False
    else:
        return False, new_node, False
    
def action_rotate_negative_sixty_degrees(node, canvas, visited, step): 
    """ rotates the robot 60 degrees clockwise 

    Args:
        node : current node 
        canvas : canvas 
        visited : visited nodes
        step : step size

    Returns:
        bool: feasibility of the action and the new node 
    """   
    new_node = node.copy()
    new_angle = new_node[2] + 60    
    
    if new_angle < 0:
        new_angle += 360
    
    new_angle %= 360 
    new_width = new_node[0] + threshold(step*np.cos(np.deg2rad(new_angle)))
    new_height = new_node[1] + threshold(step*np.sin(np.deg2rad(new_angle)))

    if (round(new_height)>0 and round(new_height)<canvas.shape[0]) and \
       (round(new_width)>0 and round(new_width)<canvas.shape[1]) and \
       (is_within_obstacle(canvas,new_height,new_width)) :
        new_node[0] = new_width
        new_node[1] = new_height
        new_node[2] = new_angle

        if(visited[new_height][new_width][new_angle//g_angle] == 1):
            return True, new_node,True
        elif(visited[new_height][new_width][new_angle//g_angle] == 2):
            return False, new_node, False
        else:
            visited[new_height][new_width][new_angle//g_angle] = 1
            return True, new_node, False
    else:
        return False, new_node, False


def action_rotate_positive_thirty_degrees(node, canvas, visited, step): 
    """ rotates the robot 30 degrees counterclockwise 

    Args:
        node : current node 
        canvas : canvas 
        visited : visited nodes
        step : step size

    Returns:
        bool: feasibility of the action and the new node 
    """   
    new_node = node.copy()
    new_angle = new_node[2] - 30    

    if new_angle < 0:
        new_angle += 360 
    new_angle %= 360
    new_width = new_node[0] + threshold(step*np.cos(np.deg2rad(new_angle)))
    new_height = new_node[1] + threshold(step*np.sin(np.deg2rad(new_angle)))

    if (round(new_height)>0 and round(new_height)<canvas.shape[0]) and \
        (round(new_width)>0 and round(new_width)<canvas.shape[1]) and \
        (is_within_obstacle(canvas,new_height,new_width)) :
        new_node[0] = new_width
        new_node[1] = new_height
        new_node[2] = new_angle

        if(visited[new_height][new_width][new_angle//g_angle] == 1):
            return True, new_node, True
        elif(visited[new_height][new_width][new_angle//g_angle] == 2):
            return False, new_node, False
        else:
            visited[new_height][new_width][new_angle//g_angle] = 1
            return True, new_node, False
    else:
        return False, new_node, False


def action_rotate_positive_sixty_degrees(node, canvas, visited, step):    
    """ rotates the robot 60 degrees counterclockwise 

    Args:
        node : current node 
        canvas : canvas 
        visited : visited nodes
        step : step size

    Returns:
        bool: feasibility of the action and the new node 
    """
    new_node = node.copy()
    new_angle = new_node[2] - 60    

    if new_angle < 0:
        new_angle += 360
    new_angle %= 360
    new_width = new_node[0] + threshold(step*np.cos(np.deg2rad(new_angle)))
    new_height = new_node[1] + threshold(step*np.sin(np.deg2rad(new_angle)))

    if (round(new_height)>0 and round(new_height)<canvas.shape[0]) and \
        (round(new_width)>0 and round(new_width)<canvas.shape[1]) and \
        (is_within_obstacle(canvas,new_height,new_width)) :
        new_node[0] = new_width
        new_node[1] = new_height
        new_node[2] = new_angle
            
        if(visited[new_height][new_width][new_angle//g_angle] == 1):
            return True, new_node,True
        elif(visited[new_height][new_width][new_angle//g_angle] == 2):
            return False, new_node, False
        else:
            visited[new_height][new_width][new_angle//g_angle] = 1
            return True, new_node,False
    else:
        return False, new_node,False
'''    

def user_inputs():
    """ this function sets rpm1, rpm2, and clearance by user inputs 

    Returns:
        Two wheel rpm(revolutions per minute) and clearance
    """
    while True:
        rpm1 = input("Enter the first RPM: ")
        if int(rpm1)<0 and int(rpm1)>360: # rpm too large would cause dx pass through the obstacles 
            print("Invalid RPM1,try again..")
        else:
            break
    while True:
        rpm2 = input("Enter the second RPM: ")
        if int(rpm2) < 0 and int(rpm2)>360: # rpm too large would cause dx pass through the obstacles 
            print("Invalid RPM2,try again..")
        else:
            break
    while True:
        clearance = input("Enter the clearance in mm: ")
        if int(clearance) < 0: # check if the step size entered is valid 
            print("Invalid clearance,try again..")
        else:
            break

    return int(rpm1), int(rpm2), int(clearance)


def validate_points(canvas):
    """ this function checks the validity of start and goal nodes 

    Args:
        canvas : the map under consideration 

    Returns:
        the initial and goal points after validation
        if the user inputs an invalid coordinate, ie within the obstacle space or outside the map, 
        user is re-prompted to enter a valid coordinate 
    """
    initial_state = []
    goal_state = []
    print(f"Map Size: height * width: {canvas.shape[0]}mm * {canvas.shape[1]}mm")
    while True:
        # check if each entered point is within the free space of the map 
        while True:
            state = input(" Start node X in mm: ")
            state = int(state)
            if(int(state)<0 or int(state)>canvas.shape[1]-1):
                print("Retry with a different X :")
                continue
            else:
                initial_state.append(int(state))
                break
        while True:
            state = input(" Start node Y in mm: ")
            state = int(state)
            if(int(state)<0 or int(state)>canvas.shape[0]-1):
                print("Retry with a different Y :")
                continue
            else:
                initial_state.append(int(state))
                break
        if(canvas[canvas.shape[0]-1 - initial_state[1]][initial_state[0]][0]==255):
            print("Invalid start node, inside the obstacle space!")
            initial_state.clear()
        else:
            break
    while True:
        while True:
            state = input("Goal node X in mm: ")
            state = int(state)
            if(int(state)<0 or int(state)>canvas.shape[1]-1):
                print("Retry with a different X :")
                continue
            else:
                goal_state.append(int(state))
                break
        while True:
            state = input("Goal node Y in mm: ")
            state = int(state)
            if(int(state)<0 or int(state)>canvas.shape[0]-1):
                print("Retry with a different Y :")
                continue
            else:
                goal_state.append(int(state))
            break
        if(canvas[canvas.shape[0]-1 - goal_state[1]][goal_state[0]][0]==255):
            print("Invalid goal node, inside the obstacle space!")
            goal_state.clear()
        else:
            break
    while True:
        initial_angle = input("Enter the initial angle of orientation in degree(+-30)")
        if int(initial_angle)%30!=0: # check if the angle entered is valid 
            print("Enter a valid angle (+-30 degrees)")
        else:
            if int(initial_angle)<0:
                initial_angle = 360 + int(initial_angle)
            initial_state.append(int(initial_angle))
            break
    
    return initial_state, goal_state

def cost(node, uL, uR, canvas):
    """ this function calculate the linear velocity and angular velocity of each node 

    Args:
        node: current node
        uL: left wheel rpm
        uR: right wheel rpm
        canvas: canvas

    Returns:
        clear_path_flag: check if a explored path collide obstacles or walls
        return_node: explored node 
        distance: distance between current node and explored node
        curve_x and curve_y: nodes between current node and explored node. For the purpose of plotting curve path
    """
    t = 0
    r = g_wheel_radius
    L = g_wheel_distance
    clear_path_flag = True

    new_node = node.copy()
    uL = uL * 3.14 / 30
    uR = uR * 3.14 / 30

    new_x = new_node[0]
    new_y = new_node[1]
    new_theta = np.deg2rad(new_node[2])
    curve_x = []
    curve_y = []
    curve_x.append(new_x)
    curve_y.append(new_y)
    distance = 0

    # The distance travelled and angle covered in each time step. 
    # Total time is 1 sec, and the time step is 0.2
    while t < 1:
        t = t + g_dt
        delta_x = 0.5 * r * (uL + uR) * math.cos(new_theta) * g_dt
        delta_y = 0.5 * r * (uL + uR) * math.sin(new_theta) * g_dt
        new_x += delta_x
        new_y += delta_y
        new_theta += (r / L) * (uR - uL) * g_dt
        distance = distance + math.sqrt(math.pow(delta_x, 2)+math.pow(delta_y, 2))
        curve_x.append(new_x)
        curve_y.append(new_y)
        if(round(new_y) < 0 or round(new_y) >= g_scaling_canvas_height) or \
          (round(new_x) < 0 or round(new_x) >= g_scaling_canvas_width):
            clear_path_flag = False
            break
        elif (is_within_obstacle(canvas, new_y, new_x) == False) :
            clear_path_flag = False
            break

    new_theta = np.rad2deg(new_theta)
    return_node = trans_node(new_x, new_y, new_theta)
    return clear_path_flag, return_node, round(distance), curve_x, curve_y

def action_set(node, canvas, rpm1, rpm2):
    """ this function calculate the linear velocity and angular velocity of each node 

    Args:
        node: current node
        canvas: canvas
        rpm1: the first wheel rpm set by user input
        rpm2: the second wheel rpm set by user input

    Returns:
        paths: all the explored nodes in the action set
        distance: distance of all the explored nodes in the action set
        curves_x, curves_y: all the nodes between each explored node and current node
    """
    paths = []
    path_distance = []
    curves_x = []
    curves_y = []
    
    actions=[[0, rpm1], 
             [rpm1, 0],
             [rpm1, rpm1],
             [0, rpm2],
             [rpm2, 0],
             [rpm2,rpm2],
             [rpm1, rpm2],
             [rpm2, rpm1]]
    
    for action in actions:
        clear_path_flag, new_node, distance, curve_x, curve_y = cost(node, action[0], action[1], canvas)
        if clear_path_flag == True:
            paths.append(new_node)
            path_distance.append(distance)
            # print(curve)
            # curves = np.append(curves, curve)
            curves_x.append(curve_x)
            curves_y.append(curve_y)
    
    # print('Curves len: ', len(curves_x))
    # print(curves_x)

    return paths, path_distance, curves_x, curves_y

def get_radius_and_clearance():

    while True:
        clearance = input("Enter the clearance:")
        if int(clearance)<0:
            print("Invalid clearance. Try again..")
        else:
            break

    while True:
        radius = input("Enter the radius:")
        if int(radius)<0:
            print("Invalid radius. Try again..")
        else:
            break    
    return int(clearance),int(radius)

def trans_node(x, y, angle):
    new_node = []
    if angle < 0:
        angle += 360
    angle = round(angle % 360)
    x = round(x)
    y = round(y)
    new_node = [x, y, angle]
    return new_node

def a_star(initial_state, goal_state, canvas, rpm1, rpm2):
    """ this function perfoms the A* algorithm for a mobile robot to navigate the map 

    
    Args:
        initial_state : the node where the robot spawns
        goal_state : the node where the robot should navigate to 
        canvas : the map in which the navigation is performed 
    """
    # store min cost to come of each node
    cost_to_come = {}

    # scaling initial node and goal node
    scaling_init_state = initial_state.copy()
    # scaling_init_state[0] = initial_state[0] * g_scaling
    # scaling_init_state[1] = initial_state[1] * g_scaling
    scaling_goal_state = goal_state.copy()
    # scaling_goal_state[0] = goal_state[0] * g_scaling
    # scaling_goal_state[1] = goal_state[1] * g_scaling
    print(initial_state, goal_state, scaling_init_state, scaling_goal_state)
    print('dis: ', (euclidean_distance(scaling_init_state, scaling_goal_state)))

    # store parent node of each node
    parent_track = {}
    start_key = (scaling_init_state[0], scaling_init_state[1], scaling_init_state[2])
    # parent_track format:[node, parent node, cost to come, curve_x, curve_y]
    parent_track[start_key] = [[scaling_init_state[0], scaling_init_state[1]], [g_initial_parent, g_initial_parent], 0, None, None]
    # store visited nodes
    visited = {}
    
    fileNodes = open("Nodes.txt", "w")
    fileParents = open("Parents.txt", "w")
    fileParents.writelines('Start parent: ' + str(parent_track[start_key]) + '\n')
    open_list = [] # empty list representing the open list
    explored_curves_x, explored_curves_y = [], []
    back_track_flag = False
    iteration = 0
    hq.heapify(open_list)
    hq.heappush(open_list,[0, scaling_init_state])
    print("Node exploration started")
    while(len(open_list) > 0):
        node = hq.heappop(open_list)
        current_cost, current_node = node[0], node[1]
        current_key = (current_node[0], current_node[1], current_node[2])  # Tuple key: (x, y, theta)
        
        if current_key in visited:
            if (visited[current_key] == 2):
                fileNodes.writelines('Closed' + str(node) + '\n')
                continue
        fileNodes.writelines('Curr ' + str(node) + '\n')

        # the node is within the threshold distance of the goal node
        # if (euclidean_distance(current_node, scaling_goal_state) <= g_goal_threshold):
        if (current_cost == -1):
            back_track_flag = True
            last_node = current_node
            print("Finding the path...") 
            break 
        
        # perfom the actions
        next_nodes, distance, curves_x, curves_y = action_set(current_node, canvas, rpm1, rpm2)
        node_ctc = parent_track[current_key][2]
        for i, next_node in enumerate(next_nodes):
            next_node_key = (next_node[0], next_node[1], next_node[2])
            next_node_ctc = node_ctc + distance[i]
            cost = next_node_ctc + weighted_cost_to_go(euclidean_distance(next_node, scaling_goal_state))
            if next_node_key in visited:
                if visited[next_node_key] == 1: # visited, but not closed
                    previous_cost = parent_track[next_node_key][2]
                    if (next_node_ctc < previous_cost):
                        parent_track[next_node_key][2] = next_node_ctc
                        hq.heappush(open_list, [cost, list(next_node)])
                        hq.heapify(open_list)
                        parent_track[next_node_key][1] = current_node
                        parent_track[next_node_key][3] = curves_x[i]
                        parent_track[next_node_key][4] = curves_y[i]
                        fileNodes.writelines('Switch ' + str(next_node) + '\n')
            else:
                parent_track[next_node_key] = [next_node, current_node, next_node_ctc, curves_x[i], curves_y[i]]
                visited[next_node_key] = 1
                if (euclidean_distance(next_node, scaling_goal_state) <= g_goal_threshold):
                    cost = -1
                hq.heappush(open_list, [cost, list(next_node)])
                hq.heapify(open_list)
                explored_curves_x.append(curves_x)
                explored_curves_y.append(curves_y)
                fileNodes.writelines('Explore ' + str(next_node) + '\n')
                fileParents.writelines('Explore parent: ' + str(parent_track[next_node_key]) + '\n')

        # mark the current node as in the closed list. 0: unexplored 1: visited; 2: closed
        visited[current_key] = 2
        hq.heapify(open_list)
        iteration += 1

    if(back_track_flag):
        print("Solved!!")
        print('Total parent: ', len(parent_track))
        #Call the backtrack function
        _, _, _, path_curve_x, path_curve_y = optimal_path(last_node, parent_track)
        generate_path(initial_state,last_node,canvas, explored_curves_x, explored_curves_y, path_curve_x, path_curve_y)
        fileNodes.writelines('Total explored: ' + str(len(explored_curves_x)) + '\n')
        fileParents.writelines('Total explored: ' + str(len(explored_curves_x)) + '\n')
    else:
        print("Solution Cannot Be Found")
    
    fileNodes.close()
    fileParents.close()

    return

def euclidean_distance(node, goal_node):
    """ this function calculates a heuristic function by using Euclidean distance
    
    Args:
        node: current node

    Returns:
        the distance between the goal node and current nodes
    """
    dis = round(math.sqrt((goal_node[0] - node[0])**2 + (goal_node[1] - node[1])**2), 2)
    # dis = math.sqrt((goal_node[0] - node[0])**2 + (goal_node[1] - node[1])**2)

    return dis

def weighted_cost_to_go(dis):
    """calculates weighted cost to go 

    Returns:
         weighted cost to go 
    """
    return dis*g_weighted_a_star


def optimal_path(last_node, parent_track):
    """finds the optimal path by backtracking

    Args:
        last_node : last node 
        parent_track_x : x_path of parent node 
        parent_track_y : y_path of parent nod 
        parent_track_theta : theta path of parent node 

    Returns:
        optimal path from the start to goal node  
    """
    path_x, path_y, path_theta, curve_x, curve_y = [], [], [], [], []
    track_node = (last_node[0], last_node[1], last_node[2])
    path_x.append(last_node[0])
    path_y.append(last_node[1])
    path_theta.append(last_node[2])
    # print(last_node, last_node[0], last_node[1], last_node[2], type(last_node))
    # print(last_node, parent_track[last_node[1]][last_node[0]][orientation(last_node[2])])
    while parent_track[track_node][1][0] != g_initial_parent:
        pre_x = parent_track[track_node][1][0]
        pre_y = parent_track[track_node][1][1]
        pre_theta = parent_track[track_node][1][2]
        pre_curve_x = parent_track[track_node][3]
        pre_curve_y = parent_track[track_node][4]

        track_node = (pre_x, pre_y, pre_theta)

        path_x.append(pre_x)
        path_y.append(pre_y)
        path_theta.append(pre_theta)
        curve_x.append(pre_curve_x)
        curve_y.append(pre_curve_y)

    return path_x, path_y, path_theta, curve_x, curve_y

def generate_path(initial_state, final_state, gen_canvas, explored_curves_x, explored_curves_y, path_curve_x, path_curve_y):
    """ 
    this function visualises the node exploration  
    """

    fourcc = cv2.VideoWriter_fourcc(*'XVID')    # Creating video writer to generate a video.
    output = cv2.VideoWriter('node_exploration.avi',fourcc,500,(g_canvas_width, g_canvas_height))
    
    print("Total Number of Nodes Explored = ",len(explored_curves_x))
    
    cv2.circle(gen_canvas,(initial_state[0], initial_state[1]),50,(0,0,255),-1)
    cv2.circle(gen_canvas,(final_state[0], final_state[1]),50,(0,0,255),-1)
    resize_canvas = imutils.resize(gen_canvas, width=g_canvas_width)
    output.write(resize_canvas)
    
    # Visualizing the explored path
    for i in range(len(explored_curves_x)):
        for j in range(len(explored_curves_x[i])):
            curve = np.column_stack((explored_curves_x[i][j][:], explored_curves_y[i][j][:]))
            cv2.polylines(gen_canvas, np.int32([curve]), False, (0,255,0), 10)
            resize_canvas = imutils.resize(gen_canvas, width=g_canvas_width) # down size
            cv2.imshow("Visualization of node exploration",resize_canvas)
            cv2.waitKey(1)
            output.write(resize_canvas)        

    # Visualizing the optimal path
    for i in reversed(range(len(path_curve_x))):
        path_curve = np.column_stack((path_curve_x[i][:], path_curve_y[i][:]))
        cv2.polylines(gen_canvas, np.int32([path_curve]), False, (255,0,196), 10)
        resize_canvas = imutils.resize(gen_canvas, width=g_canvas_width)
        cv2.waitKey(1)
        output.write(resize_canvas)

    cv2.circle(gen_canvas,(initial_state[0], initial_state[1]),50,(0,0,255),-1)
    cv2.circle(gen_canvas,(final_state[0], final_state[1]),50,(0,0,255),-1)
    resize_canvas = imutils.resize(gen_canvas, width=g_canvas_width)
    output.write(resize_canvas)

    output.release()    
# ---------- MAIN FUNCTION ------------

if __name__ == '__main__':
    
    # start the timer to keep track of total runtime
    start_time = time.time()
    # make an empty canvas 
    canvas = np.ones((g_scaling_canvas_height, g_scaling_canvas_width, 3), dtype="uint8") 
    # specify the amount of clearance by which the obstacles are to be bloated
    rpm1, rpm2, clearance = user_inputs()
    # clearance , radius = get_radius_and_clearance()
    # clearance , radius = 5, 5
    # add the obstacles in the free space of the map, and add the clearance area around them 
    canvas = draw_obstacles(canvas, g_robot_radius, clearance) 

    # validate the initial and final points before perfoming the algorithm
    initial_state, goal_state = validate_points(canvas)
    # initial_state, goal_state = [500, 500, 0], [5750, 1000]
    initial_state[1] = g_scaling_canvas_height-1 - initial_state[1]
    goal_state[1] = g_scaling_canvas_height-1 - goal_state[1]

    # perform A* algorithm
    a_star(initial_state, goal_state, canvas, rpm1, rpm2)
    # end the clock 
    end_time = time.time()
    # Resize image
    canvas = imutils.resize(canvas, width=g_canvas_width)
    cv2.imshow("Optimal path",canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # calculate the total runtime 
    run_time = end_time - start_time 
    print(f'Total run time : {round(run_time,3)} sec')

