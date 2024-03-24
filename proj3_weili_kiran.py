import cv2
import numpy as np
import heapq as hq
import time
from math import sqrt


# Global variable
g_angle = 30
g_action_set_number = 5
g_total_degree = 360
g_matrix_threshold = 0.5
g_scaling = int(1/g_matrix_threshold)
g_goal_threshold = 1.5 * g_scaling
g_canvas_height = 250
g_canvas_width = 600
g_sacling_canvas_height = 250 * g_scaling
g_sacling_canvas_width = 600 * g_scaling
g_initial_parent = -1

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
def draw_obstacles(canvas,robot_radius,clearance):   
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
    print('Dimensions of the canvas')
    print('------------------------')
    print(f"height * width: {height} * {width}")
    for i in range(width): # traverse through the width of the canvas 
        for j in range(height): # traverse through the height of the canvas
            # model the left-most rectangle
            # ----- offset -----------
            if(i-100+offset>=0 and i-175-offset<=0 and height-j-100+offset>=0 and height-j-500-offset<0):
                canvas[j][i] = [0,0,255] 
            # model the 2nd rectangle
            if(i-275+offset>=0 and i-350-offset<=0 and height-j+offset>=0 and height-j-425-offset<=0):
                canvas[j][i] = [0,0,255]
            
            # model the C-shaped figure 
            if(i-900+offset>=0 and i-1020-offset<=0 and height-j-375+offset>=0 and height-j-450-offset<=0) or (i-900+offset>=0 and i-1020-offset<=0 and height-j-50+offset>=0 and height-j-125-offset<0) or (i-1020+offset>=0 and i-1100-offset<=0 and height-j-50+offset>=0 and height-j-450-offset<=0):
                canvas[j][i] = [0,0,255]

            # model the hexagon 
            if(i+offset>=500 and i-offset<=800) and (j-offset<=(0.5*i)+75) and (j+offset>=(0.5*i)-225) and  (j-offset<=(-0.5*i)+725) and (j+offset>=(-0.5*i)+425): 
                canvas[j][i] = [0,0,255] 

            # --------- obstacle space --------
            if(i-100>=0 and i-175<=0 and height-j-100>=0 and height-j-500<0):
                canvas[j][i] = [255,255,255] 
            # model the 2nd rectangle
            if(i-275>=0 and i-350<=0 and height-j>=0 and height-j-425<=0):
                canvas[j][i] = [255,255,255]
            
            # model the C-shaped figure 
            if(i-900>=0 and i-1020<=0 and height-j-375>=0 and height-j-450<=0) or (i-900>=0 and i-1020<=0 and height-j-50>=0 and height-j-125<0) or (i-1020>=0 and i-1100<=0 and height-j-50>=0 and height-j-450<=0):
                canvas[j][i] = [255,255,255]

            # model the hexagon 
            if(i>=500 and i<=800) and (j<=(0.5*i)+75) and (j>=(0.5*i)-225) and  (j<=(-0.5*i)+725) and (j>=(-0.5*i)+425): 
                canvas[j][i] = [255,255,255]           
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

def action_rotate_zero_degrees(node, canvas, visited, step):    
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
        else: # mark the node as visited 
            visited[new_height][new_width][new_angle//g_angle] = 1
            return True, new_node, False
    else: # invalid action, new position is either in the obstacle space or outside the map 
        return False, new_node, False
    


def action_rotate_negative_thirty_degrees(node, canvas, visited, step):   
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
        else:
            visited[new_height][new_width][new_angle//g_angle] = 1
            return True, new_node, False
    else:
        return False, new_node, False
    
def action_rotate_negative_sixty_degrees(node, canvas, visited, step):    
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
        else:
            visited[new_height][new_width][new_angle//g_angle] = 1
            return True, new_node, False
    else:
        return False, new_node, False


def action_rotate_positive_thirty_degrees(node, canvas, visited, step):    
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
        else:
            visited[new_height][new_width][new_angle//g_angle] = 1
            return True, new_node, False
    else:
        return False, new_node, False


def action_rotate_positive_sixty_degrees(node, canvas, visited, step):    # Local angles
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
        else:
            visited[new_height][new_width][new_angle//g_angle] = 1
            return True, new_node,False
    else:
        return False, new_node,False
    
    
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
    while True:
        # check if each entered point is within the free space of the map 
        while True:
            state = input(" Start node X : ")
            if(int(state)<0 or int(state)>canvas.shape[1]-1):
                print("Retry with a different X :")
                continue
            else:
                initial_state.append(int(state))
                break
        while True:
            state = input(" Start node Y : ")
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
            state = input("Goal node X : ")
            if(int(state)<0 or int(state)>canvas.shape[1]-1):
                print("Retry with a different X :")
                continue
            else:
                goal_state.append(int(state))
                break
        while True:
            state = input("Goal node Y : ")
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
    while True:
        goal_angle = input("Enter the goal angle of orientation in degree(+-30)")
        if int(goal_angle)%30!=0: # check if the angle entered is valid 
            print("Enter a valid angle (+-30 degrees)")
        else:
            if int(goal_angle)<0:
                goal_angle = 360 + int(goal_angle)
            goal_state.append(int(goal_angle))
            break
    while True:
        step_size = input("Enter the step size (1-10): ")
        if int(step_size)<1 and int(step_size)>10: # check if the step size entered is valid 
            print("Invalid step size,try again..")
        else:
            break
    return initial_state,goal_state, int(step_size)


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

def orientation(angle):
    if angle < 0:
        angle += 360
    return angle // g_angle

def a_star(initial_state, goal_state, canvas, step_size):
    """ this function perfoms the A* algorithm for a mobile robot to navigate the map 

    
    Args:
        initial_state : the node where the robot spawns
        goal_state : the node where the robot should navigate to 
        canvas : the map in which the navigation is performed 
    """
    # store min cost of each node
    closed_list = {}

    cost_to_come_array = np.zeros((g_sacling_canvas_height, g_sacling_canvas_width, g_total_degree // g_angle))

    # scaling initial node and goal node
    scaling_init_state = initial_state.copy()
    scaling_init_state[0] = initial_state[0] * g_scaling
    scaling_init_state[1] = initial_state[1] * g_scaling
    scaling_goal_state = goal_state.copy()
    scaling_goal_state[0] = goal_state[0] * g_scaling
    scaling_goal_state[1] = goal_state[1] * g_scaling
    print(initial_state, goal_state, scaling_init_state, scaling_goal_state)
    print('dis: ', (euclidean_distance(scaling_init_state, scaling_goal_state)))

    # store parent node of each node
    # parent_track = np.zeros((g_sacling_canvas_height, g_sacling_canvas_width, g_total_degree // g_angle))
    # parent_track[scaling_init_state[1]][scaling_init_state[0]][orientation(scaling_init_state[2])] = g_initial_parent
    parent_track_x = np.zeros((g_sacling_canvas_height, g_sacling_canvas_width, g_total_degree // g_angle))
    parent_track_y = np.zeros((g_sacling_canvas_height, g_sacling_canvas_width, g_total_degree // g_angle))
    parent_track_theta = np.zeros((g_sacling_canvas_height, g_sacling_canvas_width, g_total_degree // g_angle))
    parent_track_x[scaling_init_state[1]][scaling_init_state[0]][orientation(scaling_init_state[2])] = g_initial_parent
    
    # store visited nodes
    visited = np.zeros((g_sacling_canvas_height, g_sacling_canvas_width, g_total_degree // g_angle))
    
    fileNodes = open("Nodes.txt", "w")
    fileParents = open("Parents.txt", "w")
    open_list = [] # empty list representing the open list 
    back_track_flag = False
    iteration = 0
    hq.heapify(open_list)
    hq.heappush(open_list,[0, scaling_init_state])
    print("Node exploration started")
    while(len(open_list) > 0):
        node = hq.heappop(open_list)
        current_cost, current_node = node[0], node[1]
        current_key = (current_node[0], current_node[1], current_node[2])  # Tuple key: (x, y, theta)

        canvas_vis = canvas.copy()
        # cv2.circle(canvas_vis, (int(current_node[0]), int(current_node[1])), 3, (0, 255, 0), -1)
        # cv2.imshow("Path Planning Visualization", canvas_vis)
        # cv2.waitKey(1) 
        if current_key not in closed_list:
            closed_list[current_key] = (parent_track_x[current_node[1]][current_node[0]][orientation(current_node[2])],
                                        parent_track_y[current_node[1]][current_node[0]][orientation(current_node[2])],
                                        parent_track_theta[current_node[1]][current_node[0]][orientation(current_node[2])])
        # node_key = (node[1][0], node[1][1], node[1][2])  # Example key using x, y, and theta
        # # Once a node is popped from the open list, it is considered in the closed list
        # closed_list[node_key] = parent_track_x[node[1][1]][node[1][0]][orientation(node[1][2])], \
        #                         parent_track_y[node[1][1]][node[1][0]][orientation(node[1][2])], \
        #                         parent_track_theta[node[1][1]][node[1][0]][orientation(node[1][2])]
        
    
        present_cost = node[0]
        fileNodes.writelines('Curr' + str(node) + '\n')
        # print(node, cost_to_come_array[node[1][1]][node[1][0]][orientation(node[1][2])])
        # print(parent_track[node[1][1]][node[1][0]][orientation(node[1][2])])
        # fileNodes.writelines('Curr' + str(parent_track) + '\n')

        # the node is within the threshold distance of the goal node
        if (euclidean_distance(list(node[1]), scaling_goal_state) <= g_goal_threshold) and (node[1][2] == scaling_goal_state[2]):
            back_track_flag = True
            last_node = list(node[1])
            print("Finding the path...") 
            break 
        
        # perfom the actions 
        flag_valid, next_node, flag_visited = action_rotate_zero_degrees(node[1], canvas, visited, step_size)
        
        # print(next_node)
        if(flag_valid):
            cost_to_come = cost_to_come_array[node[1][1]][node[1][0]][orientation(node[1][2])] + step_size
            cost = cost_to_come + euclidean_distance(next_node, scaling_goal_state)
            fileNodes.writelines('0: ' + str(cost) + ' ' + str(next_node) + '\n')
            # print('cost:', cost)
            if flag_visited == False:
                cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))] = cost_to_come
                hq.heappush(open_list, [cost, list(next_node)])
                hq.heapify(open_list)
                parent_track_x[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][0]
                parent_track_y[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][1]
                parent_track_theta[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][2]
                # parent_track[next_node[1]][next_node[0]][orientation(next_node[2])] = np.ravel_multi_index([node[1][1], 
                #                                                                                             node[1][0], 
                #                                                                                             orientation(node[1][2])], 
                #                                                                                             (g_sacling_canvas_height, g_sacling_canvas_width, g_total_degree // g_angle))
            else:
                previous_cost = cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))]
                if (cost_to_come < previous_cost):
                    cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))] = cost_to_come
                    hq.heappush(open_list, [cost, list(next_node)])
                    hq.heapify(open_list)
                    parent_track_x[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][0]
                    parent_track_y[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][1]
                    parent_track_theta[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][2]
                    # parent_track[next_node[1]][next_node[0]][orientation(next_node[2])] = np.ravel_multi_index([node[1][1], 
                    #                                                                                         node[1][0], 
                    #                                                                                         orientation(node[1][2])], 
                    #                                                                                         (g_sacling_canvas_height, g_sacling_canvas_width, g_total_degree // g_angle))
        
        flag_valid, next_node, flag_visited = action_rotate_negative_thirty_degrees(node[1], canvas, visited, step_size)
        # print(next_node)

        if(flag_valid):
            cost_to_come = cost_to_come_array[node[1][1]][node[1][0]][orientation(int(node[1][2]))] + step_size
            cost = cost_to_come + euclidean_distance(next_node, scaling_goal_state)
            fileNodes.writelines('-1: ' + str(cost) + ' ' + str(next_node) + '\n')
            # print('cost:', cost)
            if flag_visited == False:
                cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))] = cost_to_come
                hq.heappush(open_list, [cost, list(next_node)])
                hq.heapify(open_list)
                parent_track_x[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][0]
                parent_track_y[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][1]
                parent_track_theta[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][2]
                # parent_track[next_node[1]][next_node[0]][orientation(next_node[2])] = np.ravel_multi_index([node[1][1], 
                #                                                                                             node[1][0], 
                #                                                                                             orientation(node[1][2])], 
                #                                                                                             (g_sacling_canvas_height, g_sacling_canvas_width, g_total_degree // g_angle))
            else:
                previous_cost = cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))]
                if (cost_to_come < previous_cost):
                    cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))] = cost_to_come
                    hq.heappush(open_list, [cost, list(next_node)])
                    hq.heapify(open_list)
                    parent_track_x[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][0]
                    parent_track_y[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][1]
                    parent_track_theta[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][2]
                    # parent_track[next_node[1]][next_node[0]][orientation(next_node[2])] = np.ravel_multi_index([node[1][1], 
                    #                                                                                         node[1][0], 
                    #                                                                                         orientation(node[1][2])], 
                    #                                                                                         (g_sacling_canvas_height, g_sacling_canvas_width, g_total_degree // g_angle))

        flag_valid, next_node, flag_visited = action_rotate_negative_sixty_degrees(node[1], canvas, visited, step_size)
        # print(next_node)

        if(flag_valid):
            cost_to_come = cost_to_come_array[node[1][1]][node[1][0]][orientation(int(node[1][2]))] + step_size
            cost = cost_to_come + euclidean_distance(next_node, scaling_goal_state)
            fileNodes.writelines('-2: ' + str(cost) + ' ' + str(next_node) + '\n')
            # print('cost:', cost)
            if flag_visited == False:
                cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))] = cost_to_come
                hq.heappush(open_list, [cost, list(next_node)])
                hq.heapify(open_list)
                parent_track_x[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][0]
                parent_track_y[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][1]
                parent_track_theta[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][2]
                # parent_track[next_node[1]][next_node[0]][orientation(next_node[2])] = np.ravel_multi_index([node[1][1], 
                #                                                                                             node[1][0], 
                #                                                                                             orientation(node[1][2])], 
                #                                                                                             (g_sacling_canvas_height, g_sacling_canvas_width, g_total_degree // g_angle))
            else:
                previous_cost = cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))]
                if (cost_to_come < previous_cost):
                    cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))] = cost_to_come
                    hq.heappush(open_list, [cost, list(next_node)])
                    hq.heapify(open_list)
                    parent_track_x[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][0]
                    parent_track_y[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][1]
                    parent_track_theta[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][2]
                    # parent_track[next_node[1]][next_node[0]][orientation(next_node[2])] = np.ravel_multi_index([node[1][1], 
                    #                                                                                         node[1][0], 
                    #                                                                                         orientation(node[1][2])], 
                    #                                                                                         (g_sacling_canvas_height, g_sacling_canvas_width, g_total_degree // g_angle))

        flag_valid, next_node, flag_visited = action_rotate_positive_thirty_degrees(node[1], canvas, visited, step_size)
        # print(next_node)

        if(flag_valid):
            cost_to_come = cost_to_come_array[node[1][1]][node[1][0]][orientation(int(node[1][2]))] + step_size
            cost = cost_to_come + euclidean_distance(next_node, scaling_goal_state)
            fileNodes.writelines('1: ' + str(cost) + ' ' + str(next_node) + '\n')
            # print('cost:', cost)
            if flag_visited == False:
                cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))] = cost_to_come
                hq.heappush(open_list, [cost, list(next_node)])
                hq.heapify(open_list)
                parent_track_x[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][0]
                parent_track_y[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][1]
                parent_track_theta[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][2]
                # parent_track[next_node[1]][next_node[0]][orientation(next_node[2])] = np.ravel_multi_index([node[1][1], 
                #                                                                                             node[1][0], 
                #                                                                                             orientation(node[1][2])], 
                #                                                                                             (g_sacling_canvas_height, g_sacling_canvas_width, g_total_degree // g_angle))
            else:
                previous_cost = cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))]
                if (cost_to_come < previous_cost):
                    cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))] = cost_to_come
                    hq.heappush(open_list, [cost, list(next_node)])
                    hq.heapify(open_list)
                    parent_track_x[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][0]
                    parent_track_y[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][1]
                    parent_track_theta[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][2]
                    # parent_track[next_node[1]][next_node[0]][orientation(next_node[2])] = np.ravel_multi_index([node[1][1], 
                    #                                                                                         node[1][0], 
                    #                                                                                         orientation(node[1][2])], 
                    #                                                                                         (g_sacling_canvas_height, g_sacling_canvas_width, g_total_degree // g_angle))
        
        flag_valid, next_node, flag_visited = action_rotate_positive_sixty_degrees(node[1], canvas, visited, step_size)
        # print(next_node)

        if(flag_valid):
            cost_to_come = cost_to_come_array[node[1][1]][node[1][0]][orientation(int(node[1][2]))] + step_size
            cost = cost_to_come + euclidean_distance(next_node, scaling_goal_state)
            fileNodes.writelines('2: ' + str(cost) + ' ' + str(next_node) + '\n')
            # print('cost:', cost)
            if flag_visited == False:
                cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))] = cost_to_come
                hq.heappush(open_list, [cost, list(next_node)])
                hq.heapify(open_list)
                parent_track_x[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][0]
                parent_track_y[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][1]
                parent_track_theta[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][2]
                # parent_track[next_node[1]][next_node[0]][orientation(next_node[2])] = np.ravel_multi_index([node[1][1], 
                #                                                                                             node[1][0], 
                #                                                                                             orientation(node[1][2])], 
                #                                                                                             (g_sacling_canvas_height, g_sacling_canvas_width, g_total_degree // g_angle))
            else:
                previous_cost = cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))]
                if (cost_to_come < previous_cost):
                    cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))] = cost_to_come
                    hq.heappush(open_list, [cost, list(next_node)])
                    hq.heapify(open_list)
                    parent_track_x[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][0]
                    parent_track_y[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][1]
                    parent_track_theta[next_node[1]][next_node[0]][orientation(next_node[2])] = node[1][2]
                    # parent_track[next_node[1]][next_node[0]][orientation(next_node[2])] = np.ravel_multi_index([node[1][1], 
                    #                                                                                         node[1][0], 
                    #                                                                                         orientation(node[1][2])], 
                    #                                                                                         (g_sacling_canvas_height, g_sacling_canvas_width, g_total_degree // g_angle))
        
        hq.heapify(open_list)
        iteration += 1
    
    if(back_track_flag):
        print("Solved!!")
        #Call the backtrack function
        # path = generate_path(last_node, parent_track)
        # path_x, path_y, path_theta = generate_path(initial_state,last_node,closed_list,canvas)
        generate_path(initial_state,last_node,closed_list,canvas)
        # path = generate_path(last_node, parent_track_x, parent_track_y, parent_track_theta, canvas)
        # print("path: ", path )
        # print('Optimal path: ', path_x)
        
    else:
        print("Solution Cannot Be Found")
    
    fileNodes.close()

    return

def euclidean_distance(node, goal_node):
    """ this function calculates a heuristic function by using Euclidean distance
    
    Args:
        node: current node

    Returns:
        the distance between the goal node and current nodes
    """
    dis = round(sqrt((goal_node[0] - node[0])**2 + (goal_node[1] - node[1])**2))
    # dis = sqrt((goal_node[0] - node[0])**2 + (goal_node[1] - node[1])**2)

    return dis

# def generate_path(last_node, parent_track_x, parent_track_y, parent_track_theta):
#     path_x, path_y, path_theta = [], [], []
#     # print(last_node, last_node[0], last_node[1], last_node[2], type(last_node))
#     # print(last_node, parent_track[last_node[1]][last_node[0]][orientation(last_node[2])])
#     while parent_track_x[last_node[1]][last_node[0]][orientation(last_node[2])] != g_initial_parent:
#         pre_x = parent_track_x[last_node[1]][last_node[0]][orientation(last_node[2])]
#         pre_y = parent_track_y[last_node[1]][last_node[0]][orientation(last_node[2])]
#         pre_theta = parent_track_theta[last_node[1]][last_node[0]][orientation(last_node[2])]
#         # last_node = np.unravel_index(int(parent_track[last_node[1]][last_node[0]][orientation(last_node[2])]), 
#         #                              (g_sacling_canvas_height, g_sacling_canvas_width, g_total_degree // g_angle))
#         print(last_node, pre_x, pre_y, pre_theta)

#         last_node[0] = int(pre_x)
#         last_node[1] = int(pre_y)
#         last_node[2] = int(pre_theta)
#         path_x.append(last_node[0])
#         path_y.append(last_node[1])
#         path_theta.append(last_node[2])

#     return path_x, path_y, path_theta

def generate_path(initial_state, final_state, closed_list, canvas):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter('node_exploration.avi', fourcc, 50, (canvas.shape[1], canvas.shape[0]))

    print("Total Number of Nodes Explored =", len(closed_list))

    # Initial key to start backtracking from the goal state
    parent_node_key = (final_state[0], final_state[1], final_state[2])

    path = [final_state]  # Initialize path with the goal state

    # Visualize all explored nodes
    for key in closed_list.keys():
        node_pos = key[:2]  # Get x, y position
        cv2.circle(canvas, (int(key[0]), int(key[1])), 2, (0, 255, 0), -1)  # Draw explored node
        cv2.circle(canvas,(int(node_pos[0]),int(node_pos[1])),2,(0,255,0),-1)
        canvas = cv2.arrowedLine(canvas, (int(node_pos[0]),int(node_pos[1])), (int(key[0]),int(key[1])), (0,0,255), 1, tipLength = 0.2)
        cv2.imshow("Visualization of node exploration",canvas)
        cv2.waitKey(1)
        output.write(canvas)

    # Backtracking from goal to start
    while parent_node_key in closed_list and parent_node_key != tuple(initial_state):
        parent_node_info = closed_list[parent_node_key]  # Get parent node information (x, y, theta)
        path.append(parent_node_info)  # Append parent node to path

        # Drawing the path
        # cv2.arrowedLine(canvas, 
        #                 (int(parent_node_key[0]), int(parent_node_key[1])), 
        #                 (int(parent_node_info[0]), int(parent_node_info[1])), 
        #                 (255, 0, 0), 1, tipLength=0.5)
        # cv2.imshow("Path Finding", canvas)
        # output.write(canvas)  # Write frame to video
        # cv2.waitKey(50)

        # Update parent_node_key to parent for next iteration
        parent_node_key = (parent_node_info[0], parent_node_info[1], parent_node_info[2])

    # Ensure the start state is in the path
    if path[-1] != initial_state:
        path.append(initial_state)

    # # Optional: Visualize the final path in a different color
    for i in range(len(path) - 1):
        cv2.line(canvas, 
                 (int(path[i][0]), int(path[i][1])), 
                 (int(path[i + 1][0]), int(path[i + 1][1])), 
                 (0, 255, 0), 2)
        output.write(canvas)  # Write frame to video

    # Release the video writer and close windows
    output.release()
    cv2.destroyAllWindows()

    print("Path has been generated and visualized.")
# !!!!!!!!!!!!!!!!!!!!! ^^^^^^^^^^

#     output.release()
# ---------- MAIN FUNCTION ------------

if __name__ == '__main__':
    
    # start the timer to keep track of total runtime
    start_time = time.time()
    # make an empty canvas 
    canvas = np.ones((500,1200,3),dtype="uint8") 
    # specify the amount of clearance by which the obstacles are to be bloated
    # clearance , radius = get_radius_and_clearance()
    clearance , radius = 5, 5
    # add the obstacles in the free space of the map, and add the clearance area around them 
    canvas = draw_obstacles(canvas,radius,clearance) 
    # cv2.imshow("Canvas",canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # validate the initial and final points before perfoming the algorithm
    initial_state,goal_state ,step_size = validate_points(canvas)
    # initial_state, goal_state ,step_size = [5, 5, 0], [50, 50, 30], 5 
    initial_state[1] = g_canvas_height-1 - initial_state[1]
    goal_state[1] = g_canvas_height-1 - goal_state[1]
    # to downscale the image to speed up the video 
    # scale_factor = 0.5  
    # canvas = downsample_image(canvas, scale_factor=scale_factor)
    # # scaling the initial and goal points
    # initial_state = [int(x * scale_factor) for x in initial_state]
    # goal_state = [int(x * scale_factor) for x in goal_state]
    # # perform dijikstra's algorithm 
    # dijkstra(initial_state,goal_state,canvas)

    # perform A* algorithm
    a_star(initial_state, goal_state, canvas, step_size)
    # end the clock 
    end_time = time.time()
    cv2.imshow("Canvas",canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # calculate the total runtime 
    run_time = end_time - start_time 
    print(f'Total run time : {round(run_time,3)} sec')

