# #!/usr/bin/env python3

import cv2
import numpy as np
import heapq as hq
import time
import math
import imutils
import rclpy
from geometry_msgs.msg import Twist


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
g_weighted_a_star = 1
g_dt = 0.1

# Turtlebot3 waffle spec with unit mm
g_robot_radius = 220
g_wheel_radius = 33
g_wheel_distance = 287


rclpy.init(args=None)
node = rclpy.create_node('a_star_pub')
vel_pub = node.create_publisher(Twist, '/cmd_vel', 10)
rate = node.create_rate(100)
twist = Twist()


# def calculate_velocity(Thetai, UL, UR, r, L):
#     '''
#     Calculates the linear and angular velocity of the robot 
#     '''
#     UL = UL*2*np.pi/60
#     UR = UR*2*np.pi/60
#     thetan = 3.14 * Thetai / 180
#     theta_dot = (r / L) * (UR - UL) 
#     change_theta = theta_dot + thetan
#     x_dot = (r / 2) * (UL + UR) * np.cos(change_theta) 
#     y_dot = (r / 2) * (UL + UR) * np.sin(change_theta) 
#     vel_mag = np.sqrt(x_dot** 2 + y_dot** 2) 
#     return vel_mag, theta_dot

def calculate_linear_velocity(Thetai, UL, UR, r, L):
    '''
    Calculates the linear velocity of the robot.
    '''
    # Convert wheel velocities from RPM to rad/s
    UL = UL * 2 * np.pi / 60
    UR = UR * 2 * np.pi / 60
    # Calculate the linear velocity
    x_dot = (r / 2) * (UL + UR) * np.cos(np.pi * Thetai / 180)
    y_dot = (r / 2) * (UL + UR) * np.sin(np.pi * Thetai / 180)
    # Magnitude of the linear velocity
    vel_mag = np.sqrt(x_dot**2 + y_dot**2)
    return vel_mag

def calculate_angular_velocity(Thetai, UL, UR, r, L):
    '''
    Calculates the angular velocity of the robot.
    '''
    # Convert wheel velocities from RPM to rad/s
    UL = UL * 2 * np.pi / 60
    UR = UR * 2 * np.pi / 60
    # Calculate the angular velocity
    theta_dot = (r / L) * (UR - UL)
    return theta_dot

def publishVelocity(v_list,node):
    '''
    Publishes the velocity to /cmd_vel topic of the turtlebot
    
    '''
    # print("V_list",v_list)
    # endTime = rclpy.get_clock().now() + rclpy.Duration(1)
    endTime = node.get_clock().now() + rclpy.time.Duration(seconds=1)


    while node.get_clock().now() < endTime:
        twist.linear.x = v_list[0][0]/10
        twist.linear.y = 0.0 
        twist.linear.z = 0.0
        twist.angular.x = 0.0 
        twist.angular.y = 0.0 
        twist.angular.z = v_list[0][1]*10
        vel_pub.publish(twist)
        rate.sleep()

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
            if (i - 4200 ) ** 2 + (j - 1200) ** 2 - (600 + offset)**2 <= 0:
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
            if (i - 4200) ** 2 + (j - 1200) ** 2 - 600**2 <= 0:
                canvas[j][i] = [255,255,255]
            # model the hexagon 
            # if(i>=500 and i<=800) and (j<=(0.5*i)+75) and (j>=(0.5*i)-225) and  (j<=(-0.5*i)+725) and (j>=(-0.5*i)+425): 
            #     canvas[j][i] = [255,255,255]           
    return canvas

#------------- ACTION FUNCTIONS ---------------

def threshold(n):
    res = round(n*g_scaling)
    return res

    
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
            state = int(state) * g_scaling
            if(int(state)<0 or int(state)>canvas.shape[1]-1):
                print("Retry with a different X :")
                continue
            else:
                initial_state.append(int(state))
                break
        while True:
            state = input(" Start node Y : ")
            state = int(state) * g_scaling
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
            state = int(state) * g_scaling
            if(int(state)<0 or int(state)>canvas.shape[1]-1):
                print("Retry with a different X :")
                continue
            else:
                goal_state.append(int(state))
                break
        while True:
            state = input("Goal node Y : ")
            state = int(state) * g_scaling
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

def cost(node, uL, uR):
    t = 0
    r = g_wheel_radius
    L = g_wheel_distance

    new_node = node.copy()
    uL = uL * 3.14 / 30
    uR = uR * 3.14 / 30

    new_x = new_node[0]
    new_y = new_node[1]
    new_theta = np.deg2rad(new_node[2])
# Xi, Yi,Thetai: Input point's coordinates
# Xs, Ys: Start point coordinates for plot function
# Xn, Yn, Thetan: End point coordintes
    distance = 0
    while t < 1:
        t = t + g_dt
        delta_x = 0.5 * r * (uL + uR) * math.cos(new_theta) * g_dt
        delta_y = 0.5 * r * (uL + uR) * math.sin(new_theta) * g_dt
        new_x += delta_x
        new_y += delta_y
        new_theta += (r / L) * (uR - uL) * g_dt
        distance = distance + math.sqrt(math.pow(delta_x, 2)+math.pow(delta_y, 2))

    new_theta = np.rad2deg(new_theta)
    return_node = trans_node(new_x, new_y, new_theta)
    return return_node, distance

def action_set(node, canvas, rpm1, rpm2):
    paths = []
    path_distance = []
    velocity_list  = []

    
    actions=[[0, rpm1], 
             [rpm1, 0],
             [rpm1, rpm1],
             [0, rpm2],
             [rpm2, 0],
             [rpm2,rpm2],
             [rpm1, rpm2],
             [rpm2, rpm1]]
    velocity_list.append([rpm1,rpm2])
    
    for action in actions:
        new_node, distance =cost(node, action[0], action[1])
        new_width = new_node[0]
        new_height = new_node[1]
        if(round(new_height)>0 and round(new_height)<canvas.shape[0]) and \
        (round(new_width)>0 and round(new_width)<canvas.shape[1]) and \
        (is_within_obstacle(canvas, new_height, new_width)) :
            paths.append(new_node)
            path_distance.append(distance)

    return paths, path_distance,velocity_list

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
    # parent_track format:[node, parent node, cost to come]
    parent_track[start_key] = [[scaling_init_state[0], scaling_init_state[1]], [g_initial_parent, g_initial_parent], 0]
    # cost_to_come[start_key] = 0
    # store visited nodes
    # visited = np.zeros((g_sacling_canvas_height, g_sacling_canvas_width, g_total_degree // g_angle))
    visited = {}
    velocity_cal = {}

    
    fileNodes = open("Nodes.txt", "w")
    fileParents = open("Parents.txt", "w")
    open_list = [] # empty list representing the open list
    explored_nodes =[]
    # explored_nodes.append(scaling_init_state)
    back_track_flag = False
    iteration = 0
    hq.heapify(open_list)
    hq.heappush(open_list,[0, scaling_init_state])
    print("Node exploration started")
    while(len(open_list) > 0):
        node = hq.heappop(open_list)
        _, current_node = node[0], node[1]
        current_key = (current_node[0], current_node[1], current_node[2])  # Tuple key: (x, y, theta)
        
        if current_key in visited:
            if (visited[current_key] == 2):
                fileNodes.writelines('Closed' + str(node) + '\n')
                continue
        fileNodes.writelines('Curr' + str(node) + '\n')

        # the node is within the threshold distance of the goal node
        if (euclidean_distance(current_node, scaling_goal_state) <= g_goal_threshold):
            back_track_flag = True
            last_node = current_node
            print("Finding the path...") 
            break 
        
        # perfom the actions
        next_nodes, distance ,vel_list = action_set(current_node, canvas, rpm1, rpm2)
        node_ctc = parent_track[current_key][2]
        # print(next_nodes)
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
                        velocity_cal[tuple(next_node_key)] = vel_list
                        hq.heapify(open_list)
                        # parent_track[next_node_key][0] = next_node
                        parent_track[next_node_key][1] = current_node
            else:
                parent_track[next_node_key] = [next_node, current_node, next_node_ctc]
                hq.heappush(open_list, [cost, list(next_node)])
                hq.heapify(open_list)
                velocity_cal[tuple(next_node_key)] =vel_list
                explored_nodes.append([(current_node[0], current_node[1]), (next_node[0], next_node[1])])

        # mark the current node as in the closed list. 0: unexplored 1: visited; 2: closed
        visited[current_key] = 2
        hq.heapify(open_list)
        iteration += 1

    if(back_track_flag):
        print("Solved!!")
        #Call the backtrack function
        path_x, path_y, path_theta = optimal_path(last_node, parent_track)
        generate_path(initial_state,last_node,explored_nodes,canvas, path_x, path_y, path_theta,velocity_cal)
        
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
    dis = round(math.sqrt((goal_node[0] - node[0])**2 + (goal_node[1] - node[1])**2))
    # dis = sqrt((goal_node[0] - node[0])**2 + (goal_node[1] - node[1])**2)

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
    path_x, path_y, path_theta = [], [], []
    track_node = (last_node[0], last_node[1], last_node[2])
    # print(last_node, last_node[0], last_node[1], last_node[2], type(last_node))
    # print(last_node, parent_track[last_node[1]][last_node[0]][orientation(last_node[2])])
    while parent_track[track_node][1][0] != g_initial_parent:
        pre_x = parent_track[track_node][1][0]
        pre_y = parent_track[track_node][1][1]
        pre_theta = parent_track[track_node][1][2]

        track_node = (pre_x, pre_y, pre_theta)

        path_x.append(pre_x)
        path_y.append(pre_y)
        path_theta.append(pre_theta)

    return path_x, path_y, path_theta

def generate_path(initial_state, final_state, explored_nodes, gen_canvas, path_x, path_y, path_theta,velocity_calc):
    """ 
    this function visualises the node exploration  
    """

    # gen_canvas = canvas.copy()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')    # Creating video writer to generate a video.
    output = cv2.VideoWriter('node_exploration.avi',fourcc,500,(g_canvas_width, g_canvas_height))
    
    print("Total Number of Nodes Explored = ",len(explored_nodes)) 
    
    cv2.circle(gen_canvas,(initial_state[0], initial_state[1]),50,(0,0,255),-1)
    cv2.circle(gen_canvas,(final_state[0], final_state[1]),50,(0,0,255),-1)
    resize_canvas = imutils.resize(gen_canvas, width=g_canvas_width)
    output.write(resize_canvas)
    
    for i in range(len(explored_nodes)):
        cv2.arrowedLine(gen_canvas, explored_nodes[i][0], explored_nodes[i][1], (0,255,0), 10, tipLength = 0.2)
        resize_canvas = imutils.resize(gen_canvas, width=g_canvas_width)
        cv2.imshow("Visualization of node exploration",resize_canvas)
        cv2.waitKey(1)
        output.write(resize_canvas)

    # Visualizing the optimal path
    for i in reversed(range(len(path_x)-1)):
        cv2.line(gen_canvas, (path_x[i+1], path_y[i+1]), (path_x[i], path_y[i]), (255,0,196), 50)
        resize_canvas = imutils.resize(gen_canvas, width=g_canvas_width)
        cv2.waitKey(1)
        output.write(resize_canvas)
        X_dot = velocity_calc[tuple(path_x)] 
        linear_velocity = calculate_linear_velocity(path_x[i], path_y[i], path_x[i+1], path_y[i+1])
        angular_velocity = calculate_angular_velocity(path_theta[i], path_theta[i+1])
        publishVelocity([(linear_velocity,angular_velocity)],node)

    cv2.circle(gen_canvas,(initial_state[0], initial_state[1]),50,(0,0,255),-1)
    cv2.circle(gen_canvas,(final_state[0], final_state[1]),50,(0,0,255),-1)
    resize_canvas = imutils.resize(gen_canvas, width=g_canvas_width)
    output.write(resize_canvas)

    output.release()    


def take_robot_inputs():
    '''
    Gets the robot radius and clearance inputs from the user.
    '''
    
    clearance = 0
    robot_radius = 0
    left_wheel_vel = 0
    right_wheel_vel = 0

    while True:
        clearance = input("Enter the robot's clearance: ")
        if(float(clearance)<0):
            print("Enter a valid Robot Clearance!")
        else:
            break
    while True:
        left_wheel_vel = input("Enter Left Wheel Velocity in RPM: ")
        if(int(left_wheel_vel)<0):
            print("Enter a valid input velocity")
        else:
            break
    while True:
        right_wheel_vel = input("Enter Right Wheel Velocity in RPM: ")
        if(int(right_wheel_vel)<0):
            print("Enter a valid input velocity")
        else:
            break
    
    return float(clearance), int(robot_radius), int(left_wheel_vel), int(right_wheel_vel)


# ---------- MAIN FUNCTION ------------

def main():
    # start the timer to keep track of total runtime
    start_time = time.time()
    # make an empty canvas 
    canvas = np.ones((g_scaling_canvas_height, g_scaling_canvas_width, 3), dtype="uint8") 
    # specify the amount of clearance by which the obstacles are to be bloated
    # clearance , radius = get_radius_and_clearance()
    clearance , robot_radius = 5, 5
    # clearance, robot_radius, vel_L, vel_R = take_robot_inputs()

    # add the obstacles in the free space of the map, and add the clearance area around them 
    canvas = draw_obstacles(canvas,robot_radius,clearance) 
    # cv2.imshow("Canvas",canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # validate the initial and final points before perfoming the algorithm
    # initial_state,goal_state ,step_size = validate_points(canvas)
    initial_state, goal_state = [500, 500, 0], [2000, 1500]
    initial_state[1] = g_scaling_canvas_height-1 - initial_state[1]
    goal_state[1] = g_scaling_canvas_height-1 - goal_state[1]

    rpm1, rpm2 = 20, 40


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

if __name__ == '__main__':
    main()
    


