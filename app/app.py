from flask import Flask, render_template, request
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
import folium
from geopy.distance import geodesic
import time

app = Flask(__name__)

# load the graph
def load_graph(pickle_file):
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)

G = load_graph('parking_violations_graph.pickle')

# load the original data
data = pd.read_csv('cleaned_parking_violations.csv')

def find_nearest_node(lat, lon):
    coords = np.array([data['pos'] for _, data in G.nodes(data=True)])
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(coords)
    _, indices = nbrs.kneighbors([[lat, lon]])
    nearest_node = list(G.nodes())[indices[0][0]]
    return nearest_node

def bfs_violations(start_node, max_distance=500):
    visited = set()
    queue = [(start_node, 0)]
    violations = []

    while queue:
        node, distance = queue.pop(0)
        if node not in visited and distance <= max_distance:
            visited.add(node)
            violations.append(node)
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    new_distance = distance + G[node][neighbor]['weight']
                    queue.append((neighbor, new_distance))

    return violations

def dijkstra_violations(start_node, max_distance=500):
    distances = {node: float('infinity') for node in G.nodes()}
    distances[start_node] = 0
    violations = []

    pq = [(0, start_node)]
    while pq:
        current_distance, current_node = min(pq)
        pq.remove((current_distance, current_node))

        if current_distance > max_distance:
            break

        violations.append(current_node)

        for neighbor in G.neighbors(current_node):
            distance = current_distance + G[current_node][neighbor]['weight']
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                pq.append((distance, neighbor))

    return violations

def create_map(lat, lon, violations):
    map_center = [lat, lon]
    m = folium.Map(location=map_center, zoom_start=15)

    folium.Circle(
        radius=500,
        location=map_center,
        popup='500m radius',
        color='crimson',
        fill=True,
    ).add_to(m)

    for violation in violations:
        node_data = G.nodes[violation]
        folium.Marker(
            location=[node_data['pos'][0], node_data['pos'][1]],
            popup=(
                f"<b>Fine Amount:</b> ${node_data['fine_amount']}"
            ),
        ).add_to(m)

    return m

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            lat = float(request.form['latitude'])
            lon = float(request.form['longitude'])
            algorithm = request.form['algorithm']

            start_node = find_nearest_node(lat, lon)

            start_time = time.time()
            if algorithm == 'bfs':
                violations = bfs_violations(start_node)
            else:
                violations = dijkstra_violations(start_node)
            end_time = time.time()

            execution_time = end_time - start_time
            
            violations_data = [data[data['TICKET_NUMBER'] == v] for v in violations]
            total_fines = sum([v['FINE_AMOUNT'].values[0] for v in violations_data])
            probability = min(len(violations) / 100, 1)  # Assuming 100 violations is 100% probability

            m = create_map(lat, lon, violations)
            
            return render_template('result.html', 
                                   map=m._repr_html_(), 
                                   execution_time=execution_time,
                                   total_violations=len(violations),
                                   total_fines=total_fines,
                                   probability=probability,
                                   algorithm_violations=len(violations))

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)