import keras
from keras import layers
from keras import regularizers
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from reset import reset_network
from update30 import update_opinions
from reward import reward_function
from net30 import net30



# Generate the graph
n = 30 # Number of nodes
k = 5 # Number of initial links
p = 0.5
seed = 5000
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

num_agents = 3  # Example number of agents

input_shape = (300,)  # Example shape

input_shapes = [(300,) for _ in range(num_agents)] 

inputs = [layers.Input(shape=shape) for shape in input_shapes]

actor_networks = []

for i in range(num_agents):
    input_actor = layers.Input(shape=input_shape)
    inputs[i] = input_actor 
    actor_hidden = layers.Dense(units=300, activation="mish", kernel_regularizer=regularizers.l2(0.001))(input_actor)
    action_hidden = layers.LayerNormalization()(actor_hidden)
    action_hidden = layers.Dense(units=75, activation="mish")(action_hidden)
    action_hidden = layers.Dropout(0.1)(action_hidden)
    action_hidden = layers.LayerNormalization()(action_hidden)
    action_hidden = layers.Dense(units=64, activation="mish")(action_hidden)
    action_hidden = layers.Dropout(0.1)(action_hidden)
    action_hidden = layers.LayerNormalization()(action_hidden)
    action_hidden = layers.Dense(units=32, activation="mish")(action_hidden)
    action_hidden = layers.Dropout(0.1)(action_hidden)
    action_hidden = layers.Dense(units=32, activation="mish")(action_hidden)
    actor_output = layers.Dense(units=1 , activation='softsign' , name = "action_output"[i] , kernel_regularizer=regularizers.l2(0.001))(action_hidden)
    
    # Create actor model
    actor_model = models.Model(inputs=input_actor, outputs=actor_output)
    
    # Add actor model to the list
    actor_networks.append(actor_model)

yinput_critic = layers.Input(shape=(num_agents,))
concatenated_actions = layers.Concatenate(axis=1)([actor.output for actor in actor_networks])
critic_input = layers.Concatenate()([input_critic, concatenated_actions])
critic_hidden = layers.Dense(64, activation='mish', kernel_regularizer=regularizers.l2(0.001))(critic_input)
critic_hidden = layers.LayerNormalization()(critic_hidden)
critic_hidden = layers.Dense(32, activation="mish")(critic_hidden)
critic_output = layers.Dense(1, activation="linear", name = "critic_output", kernel_regularizer=regularizers.l2(0.001))(critic_hidden)

# Create critic model

# critic_model = models.Model(inputs=[input_critic] + [actor.input for actor in actor_networks], outputs=critic_output)


# Define the combined actor-critic model
combined_outputs = [actor(inputs) for actor in actor_networks] + [critic_output]
actor_critic_model = models.Model(inputs=inputs + [input_critic], outputs=combined_outputs)
# Compile actor-critic model
actor_critic_model.compile(optimizer="adam", loss=['mse'] * num_agents + ['mse'])
print("Summary of actor-critic model")
actor_critic_model.summary()