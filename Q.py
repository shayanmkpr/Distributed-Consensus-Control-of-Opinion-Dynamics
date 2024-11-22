import random


class Agent:
    def __init__(self, bias, neighbors):
        self.bias = bias
        self.neighbors = neighbors
        self.opinion = random.uniform(0, 1)  # Random initial opinion

    def update_opinion(self):
        weighted_sum = 0
        total_weight = 0
        for neighbor, weight in self.neighbors.items():
            weighted_sum += weight * neighbor.opinion
            total_weight += weight
        normalized_sum = weighted_sum / total_weight
        self.opinion = (self.bias + normalized_sum) / 2


class Network:
    def __init__(self, size, input_nodes):
        self.size = size
        self.agents = {i: Agent(random.uniform(0, 1), {}) for i in range(size)}
        self.input_nodes = set(input_nodes)

        # Create all agents first
        for i in range(size):
            self.agents[i] = Agent(random.uniform(0, 1), {})

        # Then create connections between agents with 70% probability
        for i in range(size):
            for j in range(i + 1, size):
                if random.random() < 0.7:
                    weight = random.uniform(0, 1)
                    self.agents[i].neighbors[self.agents[j]] = weight
                    self.agents[j].neighbors[self.agents[i]] = weight

    def get_opinion(self, agent_id):
        return self.agents[agent_id].opinion

    def set_input(self, agent_id, opinion):
        if agent_id in self.input_nodes:
            self.agents[agent_id].opinion = opinion

    def update(self):
        for agent in self.agents.values():
            agent.update_opinion()


# Example usage
network = Network(10, [0, 1, 2])  # Network of 10 agents with 3 inputs

# Set input opinions
network.set_input(0, 0.1)
network.set_input(1, 0.8)
network.set_input(2, 0.5)

# Update the network 10 times
for _ in range(10):
    network.update()

# Read all agent opinions
for agent_id, agent in network.agents.items():
    print(f"Agent {agent_id}: {agent.opinion}")


Table_size = 100

# input = actions, between 0 , 1, 0.05 steps = 1-0 / (0.05) = 200
# state = mean_network_opinion 200
# reward = difference from ref-opinion
def Q_learning(Q , alpha , gamma , network_mean , agent_input , reward):
    #finding which state and which action
    state = int((network_mean*Table_size)%Table_size)
    action = int((agent_input*Table_size)%Table_size)
    #Q-learning off-policy updating Q-table
    max_Q = np.max(Q[state , :])
    Q[state , action] = Q[state , action] + alpha*(reward + gamma * (max_Q - Q[state , action]))
    #finding best action to take in this step
    index = np.argmax(Q[state , :])
    #epsilon_greeedy choosing the action, a normal implimentation
    state_action = index/Table_size + alpha*np.random.normal(loc = 0.025 , scale=0.025/3)
    epsilon = 0.1
    if(np.random.normal(loc = 0.5 , scale = 0.25)<epsilon or np.random.normal(loc = 0.5 , scale = 0.25)>1-epsilon):
        state_action = np.ceil(np.random.randint(low = 0 , high=Table_size))/Table_size + alpha*np.random.normal(loc = 0.025 , scale=0.025/3)
    return Q , 0.000001 * state_action

def net_mean(network , N):
    array = []
    for i in range(N):
        array.append(network.get_opinion(i))
    mean = np.mean(array)
    return mean

def reward(network_mean , ref_opinion , agent_input , Betha , gamma_prime):
    # betha should be bigger than one, i repeat, bigger than one
    Distance = np.abs(ref_opinion - network_mean) #should be minimized
    agent_effort = agent_input * Betha # should be minimized? betah should be bigger than one
    reward = 0.01/Distance + gamma_prime/(agent_effort+0.001)
    return reward





Q_table_agent_0 = np.random.uniform(low = 0 , high = 1 , size=(Table_size , Table_size))
Q_table_agent_1 = np.random.uniform(low = 0 , high = 1 , size=(Table_size , Table_size))
Q_table_agent_2 = np.random.uniform(low = 0 , high = 1 , size=(Table_size , Table_size))
Q_table_agent_3 = np.random.uniform(low = 0 , high = 1 , size=(Table_size , Table_size))

G = 0.95

G = 0.5

G = 0.1

episode = 0
mean_episode = []
start_episode_mean = []
N = 15
R = []
ref_opinion = 1
while episode < 10:

    a = random.randint(a = 0 , b = N-7)
    b = random.randint(a = 0 , b = N-6)
    c = random.randint(a = 0 , b = N-5)
    d = random.randint(a = 0 , b = N-4)
    network = Network(N , input_nodes=[a, b+1 , c+2 , d+3])
    mean = net_mean(network , N)
    start_episode_mean.append(mean)
    agent_0_input = random.uniform(0 , 1)
    agent_1_input = random.uniform(0 , 1)
    agent_2_input = random.uniform(0 , 1)
    agent_3_input = random.uniform(0 , 1)

    reward_0 = reward(mean , ref_opinion , agent_0_input, 1 , -0.001)
    reward_1 = reward(mean , ref_opinion , agent_1_input, 1 , -0.001)
    reward_2 = reward(mean , ref_opinion , agent_2_input, 1 , -0.001)
    reward_3 = reward(mean , ref_opinion , agent_3_input, 1 , -0.001)

    iteration = 1000
    for i in range(iteration):
        mean = net_mean(network , N)
        Q_table_agent_0 ,agent_0_input =Q_learning(Q_table_agent_0 , 0.05 , G , mean , agent_0_input, reward_0)#fast learn, past looker
        Q_table_agent_1 ,agent_1_input =Q_learning(Q_table_agent_1 , 0.005 , G , mean , agent_1_input, reward_1)#slow learner, past looker
        Q_table_agent_2 ,agent_2_input =Q_learning(Q_table_agent_2 , 0.05 , G , mean , agent_2_input, reward_2)#fast learner, present looker
        Q_table_agent_3 ,agent_3_input =Q_learning(Q_table_agent_3 , 0.005 , G , mean , agent_3_input, reward_3)# slow learner, present looker
        network.set_input(a, agent_0_input)
        network.set_input(b, agent_1_input)
        network.set_input(c, agent_2_input)
        network.set_input(d, agent_3_input)

        R.append(reward_0)
        
        network.update()
    
    mean_episode.append(net_mean(network , N))
    episode+= 1