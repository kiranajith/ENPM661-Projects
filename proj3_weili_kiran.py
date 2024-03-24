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

def is_within_obstacle(canvas,new_height,new_width):
    if canvas[int(round(new_height))][int(round(new_width))][0]==255:
        return False
    else:
        return True

def draw_obstacles(canvas):   
    """ this function is used to pbstacles in the map
    the obstacles are marked in white pixels

    Args:
        canvas : the empty/map on which obstacles are to be drwan 

    Returns:
        : canvas with obstacles
    """
    height,width,_ = canvas.shape 
    print('Dimensions of the canvas')
    print('------------------------')
    print(f"height * width: {height} * {width}")
    for i in range(width): # traverse through the width of the canvas 
        for j in range(height): # traverse through the height of the canvas
            # model the left-most rectangle
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
    return angle // g_angle

def a_star(initial_state, goal_state, canvas, step_size):
    """ this function perfoms the A* algorithm for a mobile robot to navigate the map 

    
    Args:
        initial_state : the node where the robot spawns
        goal_state : the node where the robot should navigate to 
        canvas : the map in which the navigation is performed 
    """
    # store min cost of each node
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
    # parent_array = np.zeros((canvas.shape[0], canvas.shape[1], g_total_degree // g_angle))
    # parent_array_x = np.zeros((canvas.shape[0], canvas.shape[1], g_total_degree // g_angle))
    # parent_array_y = np.zeros((canvas.shape[0], canvas.shape[1], g_total_degree // g_angle))
    
    # store visited nodes
    visited = np.zeros((g_sacling_canvas_height, g_sacling_canvas_width, g_total_degree // g_angle))
    
    fileNodes = open("Nodes.txt", "w")
    open_list = [] # empty list representing the open list 
    back_track_flag = False
    iteration = 0
    hq.heapify(open_list)
    hq.heappush(open_list,[0, scaling_init_state])
    print("Node exploration started")
    while(len(open_list) > 0):
        node = hq.heappop(open_list)
        present_cost = node[0]
        fileNodes.writelines('Curr' + str(node) + '\n')
        # print(node, cost_to_come_array[node[1][1]][node[1][0]][orientation(node[1][2])])

        # the node is within the threshold distance of the goal node
        if euclidean_distance(list(node[1]), scaling_goal_state) <= g_goal_threshold:
            back_track_flag = True
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
            else:
                previous_cost = cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))]
                if (cost_to_come < previous_cost):
                    cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))] = cost_to_come
                    hq.heappush(open_list, [cost, list(next_node)])
                    hq.heapify(open_list)
        
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
            else:
                previous_cost = cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))]
                if (cost_to_come < previous_cost):
                    cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))] = cost_to_come
                    hq.heappush(open_list, [cost, list(next_node)])
                    hq.heapify(open_list)

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
            else:
                previous_cost = cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))]
                if (cost_to_come < previous_cost):
                    cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))] = cost_to_come
                    hq.heappush(open_list, [cost, list(next_node)])
                    hq.heapify(open_list)

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
            else:
                previous_cost = cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))]
                if (cost_to_come < previous_cost):
                    cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))] = cost_to_come
                    hq.heappush(open_list, [cost, list(next_node)])
                    hq.heapify(open_list)
        
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
            else:
                previous_cost = cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))]
                if (cost_to_come < previous_cost):
                    cost_to_come_array[next_node[1]][next_node[0]][orientation(int(next_node[2]))] = cost_to_come
                    hq.heappush(open_list, [cost, list(next_node)])
                    hq.heapify(open_list)
        
        hq.heapify(open_list)
        iteration += 1
    
    if(back_track_flag):
        #Call the backtrack function
        # generate_path(initial_state,goal_state,closed_list,canvas)
        print("Solved!!")

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
    canvas = draw_obstacles_with_clearance(canvas,clearance) 
    cv2.imshow("Canvas",canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # validate the initial and final points before perfoming the algorithm
    # initial_state,goal_state ,step_size = validate_points(canvas)
    initial_state, goal_state ,step_size = [5, 5, 0], [60, 60, 30], 10 
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
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # calculate the total runtime 
    run_time = end_time - start_time 
    print(f'Total run time : {round(run_time,3)} sec')

