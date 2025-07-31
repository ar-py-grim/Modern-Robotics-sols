import csv
import os
import numpy
import numpy as np

# each node gets a unique id starting from '1'
NEXT_ID = 1

# Max number of nodes in the tree
MAX_TREE_SIZE = 80
# Try to reach the goal regularly after given number of steps
TRY_GOAL_AFTER_STEPS = 15

# Robot size
ROBOT_DIAMETER=0.005

class Node:
    """A node for the random tree"""
    def __init__(self, point):
        """Create a node with a given point in the plane"""
        global NEXT_ID
        self.id = NEXT_ID
        NEXT_ID += 1
        self.point = point

    def __repr__(self):
        """Used for printing a node"""
        return f"{self.id}: {self.point[0]}, {self.point[1]}"


def distance(a, b):
    """Calculate the distance between two nodes"""
    return numpy.linalg.norm(a.point - b.point)


def find_nearest_neighbor(tree, x_sample):
    """Find the nearest neighbor of a given sample in the tree"""
    result = None
    for node in tree:
        if not result:
            result = node
        else:
            d_result = distance(result, x_sample)
            d_node = distance(node, x_sample)
            if d_node < d_result:
                result = node
    return result


def distance_from_line_segment_to_circle_center(a, b, center):
    """Calculate the shortest distance from the center of an obstacle to a given line"""
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]
    px = x2-x1
    py = y2-y1
    norm = px*px + py*py
    u =  ((center[0] - x1) * px + (center[1] - y1) * py) / float(norm)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    x = x1 + u * px
    y = y1 + u * py
    dx = x - center[0]
    dy = y - center[1]
    dist = np.sqrt(dx*dx + dy*dy)

    return dist


def collision_free(x_nearest, x_new, obstacles):
    """Check if the path from the new sample to the nearest neighbor is without collision"""
    for obstacle in obstacles:
        x, y, diameter = obstacle
        # Until now, we assumed that the robot is a point without size.
        # Therefor we add the robots size to the obstacle size here
        # to ensure that the robot does not collide due to its real size.
        diameter += ROBOT_DIAMETER
        centre = np.array([x, y])
        dist = distance_from_line_segment_to_circle_center(x_nearest.point, x_new.point, centre)
        if dist <= diameter / 2.0:
            return False
    return True


def get_result_path(tree, start, goal):
    """Get the result path from the calculated tree"""
    result = [goal]

    current = tree[goal]
    while current != start:
        result = [current] + result
        current = tree[current]

    assert current == start
    assert not tree[current]
    result = [start] + result
    return result


def rrt(start, goal, obstacles):
    """Implementation of the Rapidly exploring Random Trees algorithm"""
    nodes = [start, goal]
    tree = {start: None}
    for step in range(MAX_TREE_SIZE):
        if step % TRY_GOAL_AFTER_STEPS == 0:
            x_sample = goal
        else:
            x_sample = Node(numpy.random.rand(2, 1) - 0.5)
            nodes.append(x_sample)
        x_nearest = find_nearest_neighbor(tree.keys(), x_sample)

        if collision_free(x_nearest, x_sample, obstacles):
            tree[x_sample] = x_nearest
            if x_sample == goal:
                return tree, nodes
    raise Exception("no path found")


PATH_TO_RESULTS = os.path.join(os.path.split(__file__)[0], "..", "results")

def read_obstacles_file():
    """Read the obstacles file. With center of obstacle and its diameter."""
    with open(os.path.join(PATH_TO_RESULTS, 'obstacles.csv')) as obstacles_f:
        obstacles = csv.reader(obstacles_f)
        # x,y,diameter
        return [[float(obst[0]), float(obst[1]), float(obst[2])] for obst in obstacles]

def write_edges(tree):
    with open(os.path.join(PATH_TO_RESULTS, 'edges.csv'), 'w') as edges_f:
        edges = csv.writer(edges_f)
        for node, parent in tree.items():
            if parent:
                # root has no parent
                cost = distance(node, parent)
                # ID1,ID2,cost
                edges.writerow([node.id, parent.id, cost])


def write_nodes(nodes, goal):
    with open(os.path.join(PATH_TO_RESULTS, 'nodes.csv'), 'w') as nodes_f:
        nodes_to_write = csv.writer(nodes_f)
        for node in nodes:
            heuristic_cost = distance(node, goal)
            # ID,x,y,heuristic-cost-to-go
            nodes_to_write.writerow([node.id,
                            *node.point[0],
                            *node.point[1],
                            heuristic_cost])


def write_path_file(result_path):
    """Write the file with the result path."""
    with open(os.path.join(PATH_TO_RESULTS, 'path.csv'), 'w') as path_f:
        path_csv = csv.writer(path_f)
        p = [n.id for n in result_path]
        path_csv.writerow(p)

def main():
    # input: obstacles.csv
    obstacles = read_obstacles_file()

    start = Node(numpy.array([[-0.5], [-0.5]]))
    goal = Node(numpy.array([[0.5], [0.5]]))

    tree, nodes = rrt(start, goal, obstacles)
    result = get_result_path(tree, start, goal)

    write_edges(tree)
    write_nodes(nodes, goal)
    write_path_file(result)


if __name__ == "__main__":
    main()
