import heapq
import math
import pandas as pd

def load_data(nodes_file, edges_file, obstacles_file):
    nodes_df = pd.read_csv(nodes_file, comment='#', header=None, names=["ID", "x", "y", "h"])
    edges_df = pd.read_csv(edges_file, comment='#', header=None, names=["ID1", "ID2", "cost"])
    obstacles_df = pd.read_csv(obstacles_file, comment='#', header=None, names=["x", "y", "diameter"])
    return nodes_df, edges_df, obstacles_df

def heuristic(n1, n2, nodes):
    x1, y1 = nodes[n1]["x"], nodes[n1]["y"]
    x2, y2 = nodes[n2]["x"], nodes[n2]["y"]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def is_inside_obstacle(node, obstacles):
    x, y = node["x"], node["y"]
    for _, obs in obstacles.iterrows():
        ox, oy, r = obs["x"], obs["y"], obs["diameter"] / 2
        if math.sqrt((x - ox) ** 2 + (y - oy) ** 2) < r:
            return True
    return False

def build_graph(edges_df, nodes_df, obstacles_df):
    graph = {}
    valid_nodes = {row["ID"]: row for _, row in nodes_df.iterrows() if not is_inside_obstacle(row, obstacles_df)}
    
    for _, row in edges_df.iterrows():
        ID1, ID2, cost = int(row["ID1"]), int(row["ID2"]), row["cost"]
        if ID1 in valid_nodes and ID2 in valid_nodes:
            if ID1 not in graph:
                graph[ID1] = []
            if ID2 not in graph:
                graph[ID2] = []
            graph[ID1].append((ID2, cost))
            graph[ID2].append((ID1, cost))
    return graph

def a_star_search(start, goal, graph, nodes):
    open_set = []
    heapq.heappush(open_set, (0, start))  # (f-score, node)
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic(start, goal, nodes)
    
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return list(reversed(path))
        
        for neighbor, cost in graph.get(current, []):
            tentative_g_score = g_score[current] + cost
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal, nodes)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return [start]  # No path found

def main():
    nodes_file = r"MR_part4/week1/results/nodes.csv"
    edges_file = r"MR_part4/week1/results/edges.csv"
    obstacles_file = r"MR_part4/week1/results/obstacles.csv"
    
    nodes_df, edges_df, obstacles_df = load_data(nodes_file, edges_file, obstacles_file)
    nodes = nodes_df.set_index("ID").to_dict("index")
    graph = build_graph(edges_df, nodes_df, obstacles_df)
    start, goal = 1, max(nodes.keys())
    
    path = a_star_search(start, goal, graph, nodes)
    
    with open("MR_part4/week1/results/path.csv", "w") as f:
        f.write(",".join(map(str, path)))
    print("Path saved to path.csv")

if __name__ == "__main__":
    main()
