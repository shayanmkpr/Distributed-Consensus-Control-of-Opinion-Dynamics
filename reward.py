import numpy as np

def reward_function(y , W, y_hubs , ref_opinion , network_size , iteration, alpha , betha):
    reward= 0
    local_cost = np.zeros(shape=(3, 1))
    for i in range(1 , 3):
        for j in range(1,network_size):
            if(j!=i):
                local_cost[i] += W[i][j]*(np.linalg.norm(y[j] - y_hubs[i])**2)
    
    global_cost = np.mean(y - ref_opinion*np.ones(shape=(network_size,1))) * iteration
    global_alt = abs(np.mean(y) - ref_opinion) * np.sqrt(np.sqrt(iteration))
    reward =  np.sum(local_cost) +10* global_alt

    return reward