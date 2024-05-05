
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import time
import constant
import cv2
import imutils
        
def is_within_obstacle(canvas, start_node, end_node, dist, theta):
    """
    Checks if the given position is within an obstacle or the clearance zone.
    """
    # Check if nodes are within obstacle or clearance zone (red pixels indicating clearance)
    start_pixel = canvas[int(round(start_node.y))][int(round(start_node.x))]
    end_pixel = canvas[int(round(end_node.y))][int(round(end_node.x))]
    if start_pixel[0] == 255 or start_pixel[2] == 255 or end_pixel[0] == 255 or end_pixel[2] == 255:  # Obstacle (white) or clearance (red)
        return True
    
    # Check if the edge between two nodes is within obstacle or clearance zone (red pixels indicating clearance)
    if dist != None:
        for i in range(dist):
            x_line = int(round(start_node.x + i * math.cos(theta)))
            y_line = int(round(start_node.y + i * math.sin(theta)))
            line_pixel = canvas[y_line][x_line]
            if line_pixel[0] == 255 or line_pixel[2] == 255:  # Obstacle (white) or clearance (red)
                return True

    return False
    


def draw_obstacles(canvas, robot_radius, clearance):   
    """ this function is used to pbstacles in the map
    the obstacles are marked in white pixels

    Args:
        canvas : the empty/map on which obstacles are to be drwan 

    Returns:
        : canvas with obstacles
    """
    offset = int(robot_radius + clearance)  # Total enlargement of obstacles
    height, width, _ = canvas.shape 
    height,width,_ = canvas.shape
    scale = constant.scaling 

    for i in range(width): # traverse through the width of the canvas 
        for j in range(height): # traverse through the height of the canvas
            # model the left-most rectangle
            # ----- offset -----------
            if(i-150*scale+offset>=0 and i-160*scale-offset<=0 and height-j-100*scale+offset>=0 and height-j-200*scale-offset<0):
                canvas[j][i] = [0,0,255] 
            # model the 2nd rectangle
            if(i-270*scale+offset>=0 and i-280*scale-offset<=0 and height-j+offset>=0 and height-j-100*scale-offset<=0):
                canvas[j][i] = [0,0,255]
            # model the circle
            if (i - 420*scale ) ** 2 + (j - 80*scale) ** 2 - (30*scale + offset)**2 <= 0:
                canvas[j][i] = [0,0,255]

            # --------- obstacle space --------
            if(i-150*scale>=0 and i-160*scale<=0 and height-j-100*scale>=0 and height-j-200*scale<0):
                canvas[j][i] = [255,255,255] 
            # model the 2nd rectangle
            if(i-270*scale>=0 and i-280*scale<=0 and height-j>=0 and height-j-100*scale<=0):
                canvas[j][i] = [255,255,255]
            # model the circle
            if (i - 420*scale) ** 2 + (j - 80*scale) ** 2 - (30*scale)**2 <= 0:
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

class Node:
    def __init__(self, node):
        self.x = node[0]
        self.y = node[1]
        self.parent = None


class RRT:
    def __init__(self, start_node, goal_node, step_len, iter, canvas):
        self.start_node = Node(start_node)
        self.goal_node = Node(goal_node)
        self.step_len = step_len
        self.iter = iter
        self.vertex = [self.start_node]
        self.canvas = canvas

    def planning(self):
        # fig = plt.figure()
        x_new, y_new, explored_path = [], [], []
        solved_flag = False
        node_cnt = 0
        for i in range(self.iter):
            random_node = self.generate_random_node()
            near_node = self.nearest_neighbor(self.vertex, random_node)
            new_node, dist, theta = self.generate_new_node(near_node, random_node)

            if (is_within_obstacle(self.canvas, near_node, new_node, dist, theta) == False):
                node_cnt += 1
                self.vertex.append(new_node)
                x_new.append(new_node.x)
                y_new.append(new_node.y)
                explored_path.append([(near_node.x, near_node.y), (new_node.x, new_node.y)])
                dist = self.euclidean_distance(new_node, self.goal_node)
                if dist <= self.step_len and (is_within_obstacle(self.canvas, new_node, self.goal_node, dist, theta) == False):
                    self.generate_new_node(new_node, self.goal_node)
                    solved_flag = True
                    break

        # plt.plot(self.start_node.x, self.start_node.y, color = 'r', marker = 'o')
        # plt.plot(self.goal_node.x, self.goal_node.y, color = 'r', marker = 'x')
        # plt.scatter(x_new, y_new, color = 'b', s = 1)
        # plt.show()
        if solved_flag == True:
            path = self.optimal_path(new_node)
            self.plot_path(path, explored_path)
            print('Node Number: ', node_cnt)
            return path
        else:
            return None

    def generate_random_node(self):       
        if np.random.random() > constant.goal_sample_rate:  # 0.05
            return Node((int(np.random.uniform(constant.minX, constant.maxX)),
                         int(np.random.uniform(constant.minY, constant.maxY))))
        else:
            return self.goal_node


    @staticmethod
    def nearest_neighbor(vertex, random_node):
        min = math.hypot(vertex[0].x - random_node.x, vertex[0].y - random_node.y)
        min_idx = 0
        for idx in range(len(vertex)):
            temp = math.hypot(vertex[idx].x - random_node.x, vertex[idx].y - random_node.y)
            if(temp < min):
                min = temp
                min_idx = idx
        
        # print(vertex[min_idx].x, vertex[min_idx].y)
        return vertex[min_idx]

    def generate_new_node(self, near_node, random_node):

        dist = min(self.euclidean_distance(random_node, near_node), self.step_len)
        theta = math.atan2((random_node.y - near_node.y), (random_node.x - near_node.x))

        new_node = Node((int(near_node.x + dist * math.cos(theta)), int(near_node.y + dist * math.sin(theta))))
        new_node.parent = near_node

        return new_node, dist, theta

    def optimal_path(self, new_node):
        path = [(self.goal_node.x, self.goal_node.y)]
        pre_node = new_node
        path.append((pre_node.x, pre_node.y))

        while pre_node.parent is not None:
            pre_node = pre_node.parent
            path.append((pre_node.x, pre_node.y))

        return path

    @staticmethod
    def euclidean_distance(node, goal_node):
        return int(math.hypot((goal_node.x - node.x), (goal_node.y - node.y)))
    
    def plot_path(self, path, explored_nodes):
        cv2.circle(self.canvas,(self.start_node.x, self.start_node.y),5*constant.scaling,(0,0,255),-1)
        cv2.circle(self.canvas,(self.goal_node.x, self.goal_node.y),5*constant.scaling,(0,0,255),-1)

        for i in range(len(explored_nodes)):
            cv2.line(self.canvas, explored_nodes[i][0], explored_nodes[i][1], (0,255,0), 1*constant.scaling)
            resize_canvas = imutils.resize(self.canvas, width=constant.canvas_width)
            cv2.imshow("Visualization of node exploration", resize_canvas)
            cv2.waitKey(1)

        # Visualizing the optimal path
        for i in reversed(range(len(path)-1)):
            cv2.line(self.canvas, (path[i+1][0], path[i+1][1]), (path[i][0], path[i][1]), (255,0,196), 5*constant.scaling)
            cv2.waitKey(1)

# ---------- MAIN FUNCTION ------------

if __name__ == '__main__':
    
    # start the timer to keep track of total runtime
    start_time = time.time()
    # make an empty canvas 
    canvas = np.ones((constant.scaling_canvas_height, constant.scaling_canvas_width, 3), dtype="uint8") 
    # add the obstacles in the free space of the map, and add the clearance area around them 
    canvas = draw_obstacles(canvas, constant.robot_radius, constant.clearance) 

    # Set start node and goal node
    initial_state, goal_state = [50*constant.scaling, 100*constant.scaling, 0], [550*constant.scaling, 100*constant.scaling]
    # initial_state, goal_state = [50*constant.scaling, 150*constant.scaling, 0], [200*constant.scaling, 150*constant.scaling]
    initial_state[1] = constant.scaling_canvas_height-1 - initial_state[1]
    goal_state[1] = constant.scaling_canvas_height-1 - goal_state[1]
    start_node = (initial_state[0], initial_state[1])
    goal_node = (goal_state[0], goal_state[1])

    # perform RRT
    rrt = RRT(start_node, goal_node, 5*constant.scaling, 2000, canvas)  # (s_start, s_goal, step_len, iter_max)
    path = rrt.planning()
    if path is None:
        print("No path!!")
        print('start node, goal node: ', start_node, goal_node)
    else:
        print("Solved!!")
    # end the clock 
    end_time = time.time()
    # Resize image
    canvas = imutils.resize(canvas, width=constant.canvas_width)
    cv2.imshow("Optimal path",canvas)
    cv2.imwrite("RRT_ptimal_path.jpg",canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # calculate the total runtime 
    run_time = end_time - start_time 
    print(f'Total run time : {round(run_time,3)} sec')