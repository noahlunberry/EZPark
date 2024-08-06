import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
import pickle

def create_graph(csv_file, output_pickle, k=5):
    # load the dataset
    df = pd.read_csv(csv_file)
    print(df.head())

    # create a k-NN graph
    coords = df[['LATITUDE', 'LONGITUDE']].values
    nn = NearestNeighbors(n_neighbors=k+1, metric='haversine', n_jobs=-1)
    nn.fit(np.radians(coords))
    distances, indices = nn.kneighbors()

    # convert distances from radians to meters
    distances = distances * 6371000  # this is just earths radius

    # Create the graph
    G = nx.Graph()

    for i, (distances_row, indices_row) in enumerate(tqdm(zip(distances, indices), total=len(df), desc="Creating graph")):
        ticket_number = df.iloc[i]['TICKET_NUMBER']
        lat, lon = coords[i]
        G.add_node(ticket_number, pos=(lat, lon), fine_amount=df.iloc[i]['FINE_AMOUNT'])
        
        for j, distance in zip(indices_row[1:], distances_row[1:]):  
            if distance <= 500:  # only for up to 500 meters
                neighbor_ticket = df.iloc[j]['TICKET_NUMBER']
                G.add_edge(ticket_number, neighbor_ticket, weight=distance)

    # save the grpah as a pikle file for faster 
    with open(output_pickle, 'wb') as f:
        pickle.dump(G, f)

    print((G))
    return G


