import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
def net30():
   
  # Generate the graph
  n = 30 # Number of nodes
  k = 5 # Number of initial links
  p = 0.5
  seed = 5000
  G = nx.watts_strogatz_graph(n, k, p, seed)
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
  W = W / np.sum(W, axis=0)
  pos = nx.spring_layout(G)
  edge_colors = [W[u][v] for u, v in G.edges()]
  node_colors = [G.nodes[node]['color'] for node in G.nodes()]
  node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
  node_dotsizes = [G.nodes[node]['size'] * 10 for node in G.nodes()]
  node_dotcolors = ['blue' if i in hubs else plt.cm.Blues(1 - W[i][i]) for i in range(n)]
  fig, ax = plt.subplots()
  nx.draw_networkx_nodes(G, pos=pos, node_color=node_colors,
                         node_size=node_sizes, ax=ax)
  nx.draw_networkx_edges(G, pos=pos, edge_color=edge_colors,
                         edge_cmap=plt.cm.Blues, ax=ax)
  nx.draw_networkx_nodes(G, pos=pos, node_color=node_dotcolors,
                         node_size=node_dotsizes, ax=ax)
  ax.set_title('small-world network with 3 hubs')
  ax.set_axis_off()
  plt.show()
  
  cmap = plt.cm.cool
  cmap.set_over('black')
  
  norm = colors.Normalize(vmax=1)
  
  plt.imshow(W, cmap=cmap, norm=norm, interpolation='nearest')
  plt.colorbar(label='Edge Weight')
  plt.title('Heatmap of the Weight Matrix')
  plt.show()