import pandas as pd
import numpy as np
import heapq
import pickle
from geopy.distance import geodesic
from collections import deque
from tqdm import tqdm

# Function to calculate geographical distance
def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).meters

# BFS Algorithm
def bfs(graph, start):
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            queue.extend(graph[node] - visited)
    return visited

# Dijkstra's Algorithm
def dijkstra(graph, start):
    queue, distances = [(0, start)], {start: 0}
    while queue:
        cost, node = heapq.heappop(queue)
        for neighbor, weight in graph[node]:
            distance = cost + weight
            if neighbor not in distances or distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
    return distances

# Load the pre-built graphs
print("Loading unweighted graph from pickle...")
with open('graph_unweighted.pkl', 'rb') as f:
    graph_unweighted = pickle.load(f)
print("Unweighted graph loaded.")

print("Loading weighted graph from pickle...")
with open('graph_weighted.pkl', 'rb') as f:
    graph_weighted = pickle.load(f)
print("Weighted graph loaded.")

# User's target location
target_lat = 38.910  # 38°55'19.2"N
target_lon = -77.044  # 77°02'27.6"W

# Run BFS with progress bar
print("Running BFS...")
import time
start_time = time.time()
bfs_result = bfs(graph_unweighted, (target_lat, target_lon))
bfs_time = time.time() - start_time

# Run Dijkstra's Algorithm with progress bar
print("Running Dijkstra's Algorithm...")
start_time = time.time()
dijkstra_result = dijkstra(graph_weighted, (target_lat, target_lon))
dijkstra_time = time.time() - start_time

print(f"BFS Time: {bfs_time:.4f} seconds")
print(f"Dijkstra's Time: {dijkstra_time:.4f} seconds")

# Load the cleaned CSV file
file_path = 'cleaned_parking_violations.csv'
data = pd.read_csv(file_path)

# Function to calculate probability
def calculate_probability(data, target_lat, target_lon, radius):
    data['distance'] = data.apply(lambda row: haversine(target_lat, target_lon, row['LATITUDE'], row['LONGITUDE']), axis=1)
    tickets_within_radius = data[data['distance'] <= radius].shape[0]
    total_tickets = data.shape[0]
    probability = tickets_within_radius / total_tickets
    return probability, tickets_within_radius, total_tickets

# Calculate the probability of getting a ticket
radius = 500  # Define radius in meters
probability, tickets_within_radius, total_tickets = calculate_probability(data, target_lat, target_lon, radius)

print(f"The probability of getting a ticket at the specified location is {probability:.4f}")
print(f"Tickets within {radius} meters: {tickets_within_radius}")
print(f"Total tickets: {total_tickets}")
