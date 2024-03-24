import cv2
import numpy as np
import heapq as hq
import time
from math import sqrt


# Global variable
g_angle = 30
g_action_set_number = 5
g_total_degree = 360
g_goal_threshold = 1.5

# def is_within_obstacle(canvas,new_height,new_width):
#     if canvas[int(round(new_height))][int(round(new_width))][0]==255:

#         return False
#     else:
#         return True
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
            # ------ clearance area -----------
            # model the left-most rectangle
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

            # -------obstacle area ------------
            
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
    res = round(n*2)/2
    return res

def action_rotate_zero_degrees(node, canvas, visited, step):    
    new_node = node.copy() # copy the current node avoid modifying the original
    new_angle = new_node[2] + 0    # add angle of rotation, 0 here 

    if new_angle < 0: # check for valid rotation angles 
        new_angle += 360 
    new_angle %= 360

    new_width = threshold(new_node[0] + step*np.cos(np.deg2rad(new_angle)))
    new_height = threshold(new_node[1] + step*np.sin(np.deg2rad(new_angle)))

    # if (round(new_height)>0 and round(new_height)<canvas.shape[0]) and (round(new_width)>0 and round(new_width)<canvas.shape[1]) and  (canvas[int(round(new_height))][int(round(new_width))][0]!=255)   :
    if (0 < round(new_height) < canvas.shape[0]) and \
        (0 < round(new_width) < canvas.shape[1]) and \
        is_within_obstacle(canvas,new_height,new_width):
        new_node[0] = new_width
        new_node[1] = new_height
        new_node[2] = new_angle

        if(visited[int(new_height*2)][int(new_width*2)][int(new_angle/30)] == 1):
            return True, new_node, True
        else: # mark the node as visited 
            visited[int(new_height*2)][int(new_width*2)][int(new_angle/30)] = 1
            return True, new_node, False
    else: # invalid action, new position is either in the obstacle space or outside the map 
        return False, new_node, False
    


def action_rotate_negative_thirty_degrees(node, canvas, visited, step):   
    new_node= node.copy()
    new_angle = new_node[2] + 30    
    
    if new_angle < 0:
        new_angle += 360 
    new_angle %= 360
    new_width = threshold(new_node[0] + step * np.cos(np.deg2rad(new_angle)))
    new_height = threshold(new_node[1] + step * np.sin(np.deg2rad(new_angle)))

    if (round(new_height)>0 and round(new_height)<canvas.shape[0]) and \
        (round(new_width)>0 and round(new_width)<canvas.shape[1]) and \
        is_within_obstacle(canvas,new_height,new_width) :
        new_node[0] = new_width
        new_node[1] = new_height
        new_node[2] = new_angle
        
        if(visited[int(new_height*2)][int(new_width*2)][int(new_angle/30)] == 1):
            return True, new_node, True
        else:
            visited[int(new_height*2)][int(new_width*2)][int(new_angle/30)] = 1
            return True, new_node, False
    else:
        return False, new_node, False
    
def action_rotate_negative_sixty_degrees(node, canvas, visited, step):    
    new_node = node.copy()
    new_angle = new_node[2] + 60    
    
    if new_angle < 0:
        new_angle += 360
    
    new_angle %= 360 
    new_width = threshold(new_node[0] + step*np.cos(np.deg2rad(new_angle)))
    new_height = threshold(new_node[1] + step*np.sin(np.deg2rad(new_angle)))

    if (round(new_height)>0 and round(new_height)<canvas.shape[0]) and \
       (round(new_width)>0 and round(new_width)<canvas.shape[1]) and \
       (is_within_obstacle(canvas,new_height,new_width)) :
        new_node[0] = new_width
        new_node[1] = new_height
        new_node[2] = new_angle

        if(visited[int(new_height*2)][int(new_width*2)][int(new_angle/30)] == 1):
            return True, new_node,True
        else:
            visited[int(new_height*2)][int(new_width*2)][int(new_angle/30)] = 1
            return True, new_node, False
    else:
        return False, new_node, False


def action_rotate_positive_thirty_degrees(node, canvas, visited, step):    
    new_node = node.copy()
    new_angle = new_node[2] - 30    

    if new_angle < 0:
        new_angle += 360 
    new_angle %= 360
    new_width = threshold(new_node[0] + step*np.cos(np.deg2rad(new_angle)))
    new_height = threshold(new_node[1] + step*np.sin(np.deg2rad(new_angle)))

    if (round(new_height)>0 and round(new_height)<canvas.shape[0]) and \
        (round(new_width)>0 and round(new_width)<canvas.shape[1]) and \
        (is_within_obstacle(canvas,new_height,new_width)) :
        new_node[0] = new_width
        new_node[1] = new_height
        new_node[2] = new_angle

        if(visited[int(new_height*2)][int(new_width*2)][int(new_angle/30)] == 1):
            return True, new_node, True
        else:
            visited[int(new_height*2)][int(new_width*2)][int(new_angle/30)] = 1
            return True, new_node, False
    else:
        return False, new_node, False


def action_rotate_positive_sixty_degrees(node, canvas, visited, step):    # Local angles
    new_node = node.copy()
    new_angle = new_node[2] - 60    

    if new_angle < 0:
        new_angle += 360
    new_angle %= 360
    new_width = threshold(new_node[0] + step*np.cos(np.deg2rad(new_angle)))
    new_height = threshold(new_node[1] + step*np.sin(np.deg2rad(new_angle)))

    if (round(new_height)>0 and round(new_height)<canvas.shape[0]) and \
        (round(new_width)>0 and round(new_width)<canvas.shape[1]) and \
        (is_within_obstacle(canvas,new_height,new_width)) :
        new_node[0] = new_width
        new_node[1] = new_height
        new_node[2] = new_angle
            
        if(visited[int(new_height*2)][int(new_width*2)][int(new_angle/30)] == 1):
            return True, new_node,True
        else:
            visited[int(new_height*2)][int(new_width*2)][int(new_angle/30)] = 1
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
    angle %= 360

    return angle // g_angle
import heapq as hq
import numpy as np

def a_star(initial_state, goal_state, canvas, step_size):
    """This function performs the A* algorithm for a mobile robot to navigate the map.

    Args:
        initial_state: the node where the robot spawns
        goal_state: the node where the robot should navigate to
        canvas: the map in which the navigation is performed
        step_size: the increment by which the robot moves in the canvas
    """
    # Assuming global variables g_total_degree, g_angle, g_goal_threshold are defined elsewhere
    cost_to_come_array = np.zeros((canvas.shape[0], canvas.shape[1], g_total_degree // g_angle))

    parent_array_x = np.zeros((canvas.shape[0], canvas.shape[1], g_total_degree // g_angle))
    parent_array_y = np.zeros((canvas.shape[0], canvas.shape[1], g_total_degree // g_angle))
    
    visited = np.zeros((canvas.shape[0]*2, canvas.shape[1]*2, g_total_degree // g_angle))
    
    open_list = []
    closed_list = {} # Using a dictionary for the closed list
    back_track_flag = False
    iteration = 0
    hq.heappush(open_list, [0, initial_state, initial_state])
    print("Node exploration started")
    
    while len(open_list) > 0:
        node = hq.heappop(open_list)
        current_state = tuple(node[2])  # Convert the current node to tuple for dict key
        
        # If node is already in closed_list, continue to next iteration
        if current_state in closed_list:
            continue
        
        closed_list[current_state] = node[1]  # Add to closed list

        present_cost = node[0]
        if euclidean_distance(list(node[2]), goal_state) <= g_goal_threshold:
            back_track_flag = True
            print("Finding the path...")
            closed_list[tuple(goal_state)] = node[2]  # Ensure goal state is added with parent

            break
    
        # These functions should return flag_valid, next_node, flag_visited
        for action in [action_rotate_zero_degrees, action_rotate_negative_thirty_degrees, action_rotate_negative_sixty_degrees, action_rotate_positive_thirty_degrees, action_rotate_positive_sixty_degrees]:
            flag_valid, next_node, flag_visited = action(node[2], canvas, visited, step_size)
            if(flag_valid):
                cost_to_come = present_cost + step_size
                cost = cost_to_come + euclidean_distance(next_node, goal_state)
                next_node_tuple = tuple(next_node)
                if next_node_tuple not in closed_list:
                    cost_to_come_array[int(next_node[1])][int(next_node[0])][orientation(int(next_node[2]))] = cost
                    parent_array_x[int(next_node[1])][int(next_node[0])][orientation(int(next_node[2]))] = node[2][0]
                    parent_array_y[int(next_node[1])][int(next_node[0])][orientation(int(next_node[2]))] = node[2][1]
                    hq.heappush(open_list, [cost, node[2], list(next_node)])
                else:
                    previous_cost = cost_to_come_array[int(next_node[1])][int(next_node[0])][orientation(int(next_node[2]))]
                    if cost < previous_cost:
                        cost_to_come_array[int(next_node[1])][int(next_node[0])][orientation(int(next_node[2]))] = cost
                        parent_array_x[int(next_node[1])][int(next_node[0])][orientation(int(next_node[2]))] = node[2][0]
                        parent_array_y[int(next_node[1])][int(next_node[0])][orientation(int(next_node[2]))] = node[2][1]
                        hq.heappush(open_list, [cost, node[2], list(next_node)])
        
        iteration += 1
    
    if back_track_flag:
        # call backtrack function
        generate_path(initial_state, goal_state, closed_list, canvas)
        print("Solved!!")
    else:
        print("Solution Cannot Be Found")


def euclidean_distance(node, goal_node):
    """ this function calculates a heuristic function by using Euclidean distance
    
    Args:
        node: current node

    Returns:
        the distance between the goal node and current nodes
    """
    ctg = sqrt((goal_node[0] - node[0])**2 + (goal_node[1] - node[1])**2)

    return ctg

def generate_path(initial_state, final_state, closed_list, canvas):

    fourcc = cv2.VideoWriter_fourcc(*'XVID')    # Creating video writer to generate a video.
    output = cv2.VideoWriter('node_exploration.avi',fourcc,500,(canvas.shape[1],canvas.shape[0]))
    
    print("Total Number of Nodes Explored = ",len(closed_list)) 
    
    keys = closed_list.keys()    # Returns all the nodes that are explored
    path = []    # Stack to store the path from start to goal
    
    # Visualizing the explored nodes
    keys = list(keys)
    for key in keys:
        current_node = closed_list[tuple(key)]
        cv2.circle(canvas,(int(key[0]),int(key[1])),2,(0,0,255),-1)
        cv2.circle(canvas,(int(current_node[0]),int(current_node[1])),2,(0,0,255),-1)
        canvas = cv2.arrowedLine(canvas, (int(current_node[0]),int(current_node[1])), (int(key[0]),int(key[1])), (0,255,0), 1, tipLength = 0.2)
        cv2.imshow("Visualization of node exploration",canvas)
        cv2.waitKey(1)
        output.write(canvas)

    parent_node = closed_list[tuple(final_state)]
    path.append(final_state)    # Appending the final state because of the loop starting condition
    
    while(parent_node != initial_state):
        path.append(parent_node)
        parent_node = closed_list[tuple(parent_node)]
    
    path.append(initial_state)    # Appending the initial state because of the loop breaking condition
    print("\nOptimal Path: ")
    start_node = path.pop()
    print(start_node)

    # Visualizing the optimal path
    while(len(path) > 0):
        path_node = path.pop()
        cv2.line(canvas,(int(start_node[0]),int(start_node[1])),(int(path_node[0]),int(path_node[1])),(255,0,196),5)
        print(path_node)
        start_node = path_node.copy()
        output.write(canvas)
    
    output.release()

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
    canvas = draw_obstacles(canvas,radius, clearance) 
    # cv2.imshow("Canvas",canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # validate the initial and final points before perfoming the algorithm
    initial_state,goal_state ,step_size = validate_points(canvas)
    # initial_state, goal_state ,step_size = [5, 5, 0], [200, 200, 30], 1 
    initial_state[1] = canvas.shape[0]-1 - initial_state[1]
    goal_state[1] = canvas.shape[0]-1 - goal_state[1]
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

