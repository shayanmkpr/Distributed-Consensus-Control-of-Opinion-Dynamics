import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def update_opinions(y_hubs, hubs , W , y):
    n = 30
    # Copy the input vector
    y[hubs] = y_hubs
    # Update the opinions of the non-hub nodes
    for i in range(30):
        if i not in hubs:
            y[i] = max(W[i, i] * y[i] + (1 - W[i, i]) * np.sum([W[j, i] * y[j] for j in range(n) if j != i]) ,1)
    # Return the updated opinion vector
    return y