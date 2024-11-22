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



from keras import layers, models, regularizers

num_agents = 3
input_shape = (300,)
input_shapes = [(300,) for _ in range(num_agents)]

inputs = [layers.Input(shape=shape) for shape in input_shapes]

actor1_input = layers.Input(shape=input_shape)
actor2_input = layers.Input(shape=input_shape)
actor3_input = layers.Input(shape=input_shape)

def build_actor_model(input_actor):
    actor_hidden = layers.Dense(units=300, activation="mish", kernel_regularizer=regularizers.l2(0.001))(input_actor)
    action_hidden = layers.LayerNormalization()(actor_hidden)
    
    for _ in range(3):  # Using a loop for repetitive layers
        action_hidden = layers.Dense(units=64, activation="mish")(action_hidden)
        action_hidden = layers.Dropout(0.1)(action_hidden)
        action_hidden = layers.LayerNormalization()(action_hidden)

    actor_output = layers.Dense(units=1, activation='softsign', kernel_regularizer=regularizers.l2(0.001))(action_hidden)

    return actor_output

actor1_output = build_actor_model(actor1_input)
actor2_output = build_actor_model(actor2_input)
actor3_output = build_actor_model(actor3_input)

input_critic = layers.Input(shape=(num_agents,))
concatenated_actions = layers.Concatenate(axis=1)([actor1_output, actor2_output, actor3_output])
critic_input = layers.Concatenate()([input_critic, concatenated_actions])
critic_hidden = layers.Dense(64, activation='mish', kernel_regularizer=regularizers.l2(0.001))(critic_input)
critic_hidden = layers.LayerNormalization()(critic_hidden)
critic_hidden = layers.Dense(32, activation="mish")(critic_hidden)
critic_output = layers.Dense(1, activation="linear", name="critic_output", kernel_regularizer=regularizers.l2(0.001))(critic_hidden)

actor_critic_model = models.Model(inputs=[actor1_input, actor2_input, actor3_input, input_critic], outputs=[actor1_output, actor2_output, actor3_output, critic_output])

actor_critic_model.compile(optimizer="adam", loss=['mse'] * num_agents + ['mse'])

# Print the summary of the actor-critic model
print("Summary of actor-critic model")
actor_critic_model.summary()




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
W = np.zeros((n, n)) 
for u, v in G.edges(): 
  w = np.random.uniform(0, 1) 
  q = np.random.uniform(0, 1) 
  W[u][v] = w 
  W[v][u] = q 
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

num_episodes = 100
max_steps_per_episode = 500
batch_size = 32
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995

replay_buffer = []

for episode in range(num_episodes):
    state = reset_network(n, k, p, seed)
    state = np.reshape(state, (1, -1))  # Reshape state to fit the input shape
    total_reward = 0
    
    for step in range(max_steps_per_episode):
        actions = []
        for actor_model in actor_networks:
            if np.random.rand() <= epsilon:
                action = np.random.uniform(-1, 1)
            else:
                action = actor_model.predict(state)
            actions.append(action)
        
        new_state = update_opinions(actions, hubs, W, state)
        reward = reward_function(new_state, y_hubs, ref_opinion, network_size, step, alpha, beta)
        
        replay_buffer.append((state, actions, reward, new_state))
        
        state = new_state
        
        total_reward += reward
        
        batch_indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
        batch = [replay_buffer[i] for i in batch_indices]
        
        inputs_critic = []
        targets_critic = []
        for state_batch, actions_batch, reward_batch, new_state_batch in batch:
            inputs_critic.append([np.concatenate((state_batch, action_batch)) for action_batch in actions_batch])
            q_values = []
            for i, actor_model in enumerate(actor_networks):
                q_values.append(actor_model.predict(new_state_batch))
            target_critic = reward_batch + gamma * np.max(q_values)
            targets_critic.append(target_critic)
        
        critic_model.train_on_batch(inputs_critic, targets_critic)
        
        for actor_model in actor_networks:
            actor_model.train_on_batch(state, actions)
        
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        if np.allclose(state, new_state):
            break
    
    # Print episode results
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
