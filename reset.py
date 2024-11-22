import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors




def reset_network(n , k , p , seed):
    # Generate the graph
    n = n # Number of nodes
    k = k # Number of initial links
    p = p
    G = nx.watts_strogatz_graph(n, k, p, seed)
    # Customize the graph
    # Calculate the degree of each node
    degrees = dict(G.degree())
    # Sort nodes by degree and select the top 3
    hubs = sorted(degrees, key=degrees.get, reverse=True)[:3]
    for node in G.nodes():
        if node in hubs:
            G.nodes[node]['color'] = 'black'
        else:
            G.nodes[node]['color'] = 'gray'
        G.nodes[node]['size'] = G.degree(node)
    W = np.zeros((n, n)) # Initialize the weight matrix with zeros
    for u, v in G.edges(): # Loop over the edges in the graph
        w = np.random.uniform(0, 1)
        q = np.random.uniform(0, 1)
        W[u][v] = w # Assign the weight to the edge
        W[v][u] = q # Make the matrix symmetric
    np.fill_diagonal(W, np.random.uniform(low=0, high=1, size=n)) # Assign self-loop values
    # Set incoming edges of hub nodes to zero and self-loops to 1
    for i in hubs:
        for j in np.arange(n):
            if i == j:
                W[i][i] = 1
        # Normalize each column
    W = W / np.sum(W, axis=0)
