import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
import pickle

def create_graph(csv_file, output_pickle, k=5):
    # Load the dataset
    df = pd.read_csv(csv_file)
    print(df.head())

    # Create a k-NN graph
    coords = df[['LATITUDE', 'LONGITUDE']].values
    nn = NearestNeighbors(n_neighbors=k+1, metric='haversine', n_jobs=-1)
    nn.fit(np.radians(coords))
    distances, indices = nn.kneighbors()

    # Convert distances from radians to meters
    distances = distances * 6371000  # Earth's radius in meters

    # Create the graph
    G = nx.Graph()

    for i, (distances_row, indices_row) in enumerate(tqdm(zip(distances, indices), total=len(df), desc="Creating graph")):
        ticket_number = df.iloc[i]['TICKET_NUMBER']
        lat, lon = coords[i]
        G.add_node(ticket_number, pos=(lat, lon), fine_amount=df.iloc[i]['FINE_AMOUNT'])
        
        for j, distance in zip(indices_row[1:], distances_row[1:]):  # Skip the first one as it's the node itself
            if distance <= 500:  # Only add edges for distances up to 500 meters
                neighbor_ticket = df.iloc[j]['TICKET_NUMBER']
                G.add_edge(ticket_number, neighbor_ticket, weight=distance)

    # Save the graph as a pickle file
    with open(output_pickle, 'wb') as f:
        pickle.dump(G, f)

    print((G))
    return G

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

# Usage example
if __name__ == "__main__":
    csv_file = 'cleaned_parking_violations.csv'
    pickle_file = 'parking_violations_graph.pickle'

    # Create and save the graph (only need to run this once)
    G = create_graph(csv_file, pickle_file, k=10)

    # Load the graph from pickle file
    # G = load_graph(pickle_file)

    # Example usage of BFS and Dijkstra's
    start_node = list(G.nodes())[0]  # Just using the first node as an example
    print(f"Violations within 500m (BFS): {bfs_violations(G, start_node)}")
    print(f"Violations within 500m (Dijkstra's): {dijkstra_violations(G, start_node)}")