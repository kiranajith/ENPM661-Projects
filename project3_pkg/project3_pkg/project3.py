
import cv2
import numpy as np
import heapq as hq
import time
import math
import imutils
import rclpy
from rclpy.node import Node
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
g_weighted_a_star = 2
g_dt = 0.2
rpm1, rpm2 = 50, 100


# Turtlebot3 waffle spec with unit mm
g_robot_radius = 220
g_wheel_radius = 33
g_wheel_distance = 287


rclpy.init()  # Initialize ROS2 Python client library
node = rclpy.create_node('path_planner_node')  # Create a new ROS2 node
vel_pub = node.create_publisher(Twist, '/cmd_vel', 10)
rate = node.create_rate(1)  # 1 Hz

def calculate_velocity(Thetai, UL, UR, r, L):
    '''
    Calculates the linear and angular velocity of the robot 
    '''
    UL = UL*2*np.pi/60
    UR = UR*2*np.pi/60
    thetan = 3.14 * Thetai / 180
    theta_dot = (r / L) * (UR - UL) 
    change_theta = theta_dot + thetan
    x_dot = (r / 2) * (UL + UR) * np.cos(change_theta) 
    y_dot = (r / 2) * (UL + UR) * np.sin(change_theta) 
    vel_mag = np.sqrt(x_dot** 2 + y_dot** 2) / 1000
    print(vel_mag, theta_dot) 
    return vel_mag, theta_dot

def publishVelocity(linear,angular):
    '''
    Publishes the velocity to /cmd_vel topic of the turtlebot
    
    '''
    # print("V_list",v_list)
    # print('Pub: ', linear,angular)
    node.get_logger().info('Publishing velocity...')
    node.get_logger().info(f'linear:{linear}\nangular:{angular}')
    twist = Twist()

    twist.linear.x = linear
    twist.linear.y = 0.0
    twist.linear.z = 0.0
    twist.angular.x = 0.0 
    twist.angular.y = 0.0 
    twist.angular.z = angular
    vel_pub.publish(twist)
    # rate.sleep()
    

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

def cost(node, uL, uR, canvas):
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
    paths = []
    path_distance = []
    curves_x = []
    curves_y = []
    # velocities = []  # List to store velocities for each action
    action_list = []
    
    actions=[[0, rpm1], 
             [rpm1, 0],
             [rpm1, rpm1],
             [0, rpm2],
             [rpm2, 0],
             [rpm2,rpm2],
             [rpm1, rpm2],
             [rpm2, rpm1]]
    
    for i, action in enumerate(actions):
        clear_path_flag, new_node, distance, curve_x, curve_y = cost(node, action[0], action[1], canvas)
        if clear_path_flag == True:
            paths.append(new_node)
            path_distance.append(distance)
            action_list.append(i)
            # print(curve)
            # curves = np.append(curves, curve)
            curves_x.append(curve_x)
            curves_y.append(curve_y)
            # linear_vel, angular_vel = calculate_velocity(new_node[2], action[0], action[1], g_wheel_radius, g_wheel_distance)
            # velocities.append((linear_vel, angular_vel))
            # velocity_dict[tuple(new_node)] = (linear_vel, angular_vel)
    
    # print('Curves len: ', len(curves_x))
    # print(curves_x)
    # print(action_list)

    return paths, path_distance, curves_x, curves_y , action_list

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
    # parent_track format:[node, parent node, cost to come, curve_x, curve_y, action_list]
    parent_track[start_key] = [[scaling_init_state[0], scaling_init_state[1]], [g_initial_parent, g_initial_parent], 0, None, None, None]
    # store visited nodes
    visited = {}
    global velocity_dict

    
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
        next_nodes, distance, curves_x, curves_y, action_list = action_set(current_node, canvas, rpm1, rpm2)
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
                        parent_track[next_node_key][5] = action_list[i]
                        fileNodes.writelines('Switch ' + str(next_node) + '\n')
            else:
                parent_track[next_node_key] = [next_node, current_node, next_node_ctc, curves_x[i], curves_y[i], action_list[i]]
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
        path_x, path_y, path_theta, path_curve_x, path_curve_y, action_list = optimal_path(last_node, parent_track)
        generate_path(initial_state,last_node,canvas, explored_curves_x, explored_curves_y, path_curve_x, path_curve_y, path_x, path_y, path_theta, action_list, rpm1, rpm2)
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
    path_x, path_y, path_theta, curve_x, curve_y, action_list = [], [], [], [], [], []
    track_node = (last_node[0], last_node[1], last_node[2])
    path_x.append(last_node[0])
    path_y.append(last_node[1])
    path_theta.append(last_node[2])
    # action_list.append(-1)
    # print(last_node, last_node[0], last_node[1], last_node[2], type(last_node))
    # print(last_node, parent_track[last_node[1]][last_node[0]][orientation(last_node[2])])
    while parent_track[track_node][1][0] != g_initial_parent:
        pre_x = parent_track[track_node][1][0]
        pre_y = parent_track[track_node][1][1]
        pre_theta = parent_track[track_node][1][2]
        pre_curve_x = parent_track[track_node][3]
        pre_curve_y = parent_track[track_node][4]
        pre_action_list = parent_track[track_node][5]

        track_node = (pre_x, pre_y, pre_theta)

        path_x.append(pre_x)
        path_y.append(pre_y)
        path_theta.append(pre_theta)
        curve_x.append(pre_curve_x)
        curve_y.append(pre_curve_y)
        action_list.append(pre_action_list)

    return path_x, path_y, path_theta, curve_x, curve_y, action_list

def generate_velocities_for_optimal_path(path_x, path_y, path_theta, action_list, rpm1, rpm2):
    actions=[[0, rpm1], 
             [rpm1, 0],
             [rpm1, rpm1],
             [0, rpm2],
             [rpm2, 0],
             [rpm2,rpm2],
             [rpm1, rpm2],
             [rpm2, rpm1]]
    optimal_path_velocities = {}
    count = 0 
    for i in reversed(range(len(path_x)-1)):
        # Calculate velocities for each segment of the optimal path
        linear_vel, angular_vel = calculate_velocity(path_theta[i], actions[action_list[i]][0], actions[action_list[i]][1], g_wheel_radius, g_wheel_distance)
        # publishVelocity(linear_vel,angular_vel)
        # time.sleep(1)
        # count += 1
        # print('count:',count)
        optimal_path_velocities[(path_x[i], path_y[i], path_theta[i])] = (linear_vel, angular_vel)
    return optimal_path_velocities


def generate_path(initial_state, final_state, gen_canvas, explored_curves_x, explored_curves_y, path_curve_x, path_curve_y, path_x, path_y, path_theta, action_list, rpm1, rpm2):
    """
    This function visualizes the node exploration and publishes the velocities to move the robot along the optimal path.
    """
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Creating video writer to generate a video.
    # output = cv2.VideoWriter('node_exploration.avi', fourcc, 500, (g_canvas_width, g_canvas_height))

    print("Total Number of Nodes Explored = ", len(explored_curves_x))

    cv2.circle(gen_canvas, (initial_state[0], initial_state[1]), 50, (0, 0, 255), -1)
    cv2.circle(gen_canvas, (final_state[0], final_state[1]), 50, (0, 0, 255), -1)
    resize_canvas = imutils.resize(gen_canvas, width=g_canvas_width)
    # output.write(resize_canvas)

    # Visualizing the explored path
    for i in range(len(explored_curves_x)):
        for j in range(len(explored_curves_x[i])):
            curve = np.column_stack((explored_curves_x[i][j][:], explored_curves_y[i][j][:]))
            cv2.polylines(gen_canvas, np.int32([curve]), False, (0, 255, 0), 10)
            resize_canvas = imutils.resize(gen_canvas, width=g_canvas_width)
            cv2.imshow("Visualization of node exploration", resize_canvas)
            cv2.waitKey(1)
            # output.write(resize_canvas)

    # Generate velocities for the optimal path
    optimal_path_velocities = generate_velocities_for_optimal_path(path_x, path_y, path_theta, action_list, rpm1, rpm2)

    # Visualizing the optimal path
    for i in range(len(path_x) - 1):
        cv2.line(gen_canvas, (path_x[i], path_y[i]), (path_x[i + 1], path_y[i + 1]), (255, 0, 196), 10)
        resize_canvas = imutils.resize(gen_canvas, width=g_canvas_width)
        cv2.waitKey(1)
        # output.write(resize_canvas)

    # Publishing velocities for the optimal path
    for i in reversed(range(len(path_x)-1)):
        node_key = (path_x[i], path_y[i], path_theta[i])
        if node_key in optimal_path_velocities:
            linear_vel, angular_vel = optimal_path_velocities[node_key]
            publishVelocity(linear_vel, angular_vel)
            # Assuming a delay to simulate real robot movement - may need to be adjusted
            time.sleep(1)
            # rate.sleep()

    # Stop the robot after completing the path
    publishVelocity(0.0, 0.0)

    # output.release()
    cv2.destroyAllWindows()

# ---------- MAIN FUNCTION ------------

def main():
        # start the timer to keep track of total runtime
    start_time = time.time()
    # make an empty canvas 
    canvas = np.ones((g_scaling_canvas_height, g_scaling_canvas_width, 3), dtype="uint8") 
    # specify the amount of clearance by which the obstacles are to be bloated
    # clearance , radius = get_radius_and_clearance()
    clearance , radius = 5, 5
    # add the obstacles in the free space of the map, and add the clearance area around them 
    canvas = draw_obstacles(canvas,radius,clearance) 
    # cv2.imshow("Canvas",canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # validate the initial and final points before perfoming the algorithm
    # initial_state,goal_state ,step_size = validate_points(canvas)
    initial_state, goal_state = [500, 1000, 0], [2200, 1000]
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

if __name__ == '__main__':
    main()