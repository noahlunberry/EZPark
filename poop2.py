import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
import pickle

def load_graph(pickle_file):
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)

def bfs_violations(G, start_node, max_distance=500):
    visited = set()
    queue = [(start_node, 0)]
    violations = 0

    while queue:
        node, distance = queue.pop(0)
        if node not in visited and distance <= max_distance:
            visited.add(node)
            violations += 1
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    new_distance = distance + G[node][neighbor]['weight']
                    queue.append((neighbor, new_distance))

    return violations

def dijkstra_violations(G, start_node, max_distance=500):
    distances = {node: float('infinity') for node in G.nodes()}
    distances[start_node] = 0
    violations = 0

    pq = [(0, start_node)]
    while pq:
        current_distance, current_node = min(pq)
        pq.remove((current_distance, current_node))

        if current_distance > max_distance:
            break

        violations += 1

        for neighbor in G.neighbors(current_node):
            distance = current_distance + G[current_node][neighbor]['weight']
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                pq.append((distance, neighbor))

    return violations

def find_nearest_node(G, lat, lon):
    coords = np.array([data['pos'] for _, data in G.nodes(data=True)])
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(coords)
    _, indices = nbrs.kneighbors([[lat, lon]])
    nearest_node = list(G.nodes())[indices[0][0]]
    return nearest_node

# Usage example
if __name__ == "__main__":
    pickle_file = 'parking_violations_graph.pickle'

    # Load the graph from pickle file
    G = load_graph(pickle_file)

    # Example coordinates
    lat, lon = 38.890, -77.005

    # Find the nearest node
    start_node = find_nearest_node(G, lat, lon)
    print(f"Nearest node to ({lat}, {lon}): {start_node}")

    # Use the nearest node for BFS and Dijkstra's
    print(f"Violations within 500m (BFS): {bfs_violations(G, start_node)}")
    print(f"Violations within 500m (Dijkstra's): {dijkstra_violations(G, start_node)}")
