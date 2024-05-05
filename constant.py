# Global variable
angle = 30
action_set_number = 5
total_degree = 360
matrix_threshold = 0.1
scaling = int(1/matrix_threshold)
goal_threshold = 10 * scaling
canvas_height = 200
canvas_width = 600
scaling_canvas_height = canvas_height * scaling
scaling_canvas_width = canvas_width * scaling
initial_parent = -1
weighted_a_star = 4
dt = 0.2
clearance = 0.5 * scaling

# Turtlebot3 waffle spec with unit mm
robot_radius = 22 * scaling
wheel_radius = 3.3 * scaling
wheel_distance = 28.7 * scaling

# Min and Max x, y position. For the purpose of generating valid random points
minX = clearance + robot_radius
minY = clearance + robot_radius
maxX = scaling_canvas_width - clearance - robot_radius
maxY = scaling_canvas_height - clearance - robot_radius

goal_sample_rate = 0.01
path_resolution = 0.5 * scaling
