import cv2
import numpy as np
import heapq as hq
import time


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


def validate_point(canvas):
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
    return initial_state,goal_state


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



if __name__ == '__main__':
    
    # start the timer to keep track of total runtime
    start_time = time.time()
    # make an empty canvas 
    canvas = np.ones((500,1200,3),dtype="uint8") 
    # specify the amount of clearance by which the obstacles are to be bloated
    clearance , radius = get_radius_and_clearance()
    # add the obstacles in the free space of the map, and add the clearance area around them 
    canvas = draw_obstacles_with_clearance(canvas,clearance) 
    cv2.imshow("Canvas",canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # validate the initial and final points before perfoming the algorithm
    initial_state,goal_state = validate_point(canvas) 
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
    # end the clock 
    end_time = time.time()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # calculate the total runtime 
    run_time = end_time - start_time 
    print(f'Total run time : {round(run_time,3)} sec')

