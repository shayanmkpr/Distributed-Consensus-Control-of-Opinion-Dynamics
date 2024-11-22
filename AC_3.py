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


n = 30 # Number of nodes
k = 5 # Number of initial links
p = 0.5
seed = 5000
G = nx.watts_strogatz_graph(n, k, p, seed)
degrees = dict(G.degree())
hubs = sorted(degrees, key=degrees.get, reverse=True)[:3]
for node in G.nodes():
  if node in hubs:
    G.nodes[node]['color'] = 'black'
  else:
    G.nodes[node]['color'] = 'gray'
  G.nodes[node]['size'] = G.degree(node)

# Assign weights to edges
W = np.zeros((n, n)) 
for u, v in G.edges(): 
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


iteration = 20
ref_opinion = 0.9
loss_plot = []
reward_plot = []
mean_plot = []
y_hist = []
progress = []
episode_count  = 50

from keras import layers, models, regularizers

num_agents = 3  # Example number of agents
input_shape = (300,)  # Example shape

# Define input for each agent's action
input_actors = [layers.Input(shape=input_shape) for _ in range(num_agents)]

# Initialize actor networks for each agent
actor_networks = []
critic_networks = []
for i in range(num_agents):
    # Actor architecture
    input_actor = input_actors[i]
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
    actor_output = layers.Dense(units=1 , activation='softsign' , name = f"action_output_{i}" , kernel_regularizer=regularizers.l2(0.001))(action_hidden)

    # Critic architecture
    critic_input = layers.Concatenate()([input_actor, actor_output])
    critic_hidden = layers.Dense(64, activation='mish', kernel_regularizer=regularizers.l2(0.001))(critic_input)
    critic_hidden = layers.LayerNormalization()(critic_hidden)
    critic_hidden = layers.Dense(32, activation="mish")(critic_hidden)
    critic_output = layers.Dense(1, activation="linear", name = f"critic_output_{i}", kernel_regularizer=regularizers.l2(0.001))(critic_hidden)

    # Create actor and critic models
    actor_model = models.Model(inputs=input_actor, outputs=actor_output)
    critic_model = models.Model(inputs=input_actor, outputs=critic_output)
    
    # Add actor and critic models to the lists
    actor_networks.append(actor_model)
    critic_networks.append(critic_model)

# Define the combined actor-critic model
actor_outputs = [actor(inputs) for inputs, actor in zip(input_actors, actor_networks)]
combined_outputs = actor_outputs + [critic(inputs) for inputs, critic in zip(input_actors, critic_networks)]
actor_critic_model = models.Model(inputs=input_actors, outputs=combined_outputs)

# Compile actor-critic model
actor_critic_model.compile(optimizer="adam", loss=['mse'] * num_agents + ['mse'] * num_agents)
print("Summary of actor-critic model")
actor_critic_model.summary()



y = (np.random.uniform(low=0, high=1, size=30))

episode = 0
for episode in range(episode_count):
    # y = (np.random.uniform(low=0, high=1, size=n))
    print(np.mean(y))
    reward = 0
    reset_network(30 ,5 , 0.5,  5000)
    print("epsiode = " , episode)
    for i in range(1,iteration):
        y_hubs =0
        y_reshaped = y.reshape(1, 30) 
        y_hubs = actor_critic_model.predict(x=[y_reshaped])
        y = update_opinions(y_hubs[0] , hubs , W , y)
        y_hist.append(y)
        # setting the systems's reward the objective function of the system
        reward = reward_function(y , W , ref_opinion , 30 ,i , 0.0005 , 0.0005 , 1)
        print("progress value(must decrease) " , np.mean(y))
        print("Episode:", episode, "Iteration:", i, "Reward:", reward)
        progress.append(np.mean(y))
        reward_plot.append(-1*reward)
        actor_critic_model.fit(x = y.reshape(1,30) , y = [y_hubs[0].reshape(1,-1)*(reward) , -1*reward.reshape(1,1)])
        loss_plot = actor_critic_model.history.history["loss"]