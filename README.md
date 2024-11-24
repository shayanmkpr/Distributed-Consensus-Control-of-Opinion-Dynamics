# Distributed Consensus Control of Opinion Dynamics

This repository contains the implementation for the optimization of **distributed consensus control** in a multi-agent system to influence opinion dynamics. Below is a comprehensive explanation of the code, mathematical framework, and methodology used.

## Problem Description

The objective is to optimize the opinions of agents in a network while balancing local and global costs. This involves using distributed algorithms where agents adjust their opinions iteratively based on shared information.

### Objective Function

The total cost to be minimized is the combination of **local cost** and **global cost**:

$
\text{Objective Function} = \sum_{i=1}^N \text{cost}_{\text{local}}[i] + \text{cost}_{\text{global}}
$

1. **Local Cost**:  
   For an agent \( j \), the local cost is defined as:
   $$
   \text{cost}_{\text{local}}[i] = \sum_{j=1}^N w[i][j] \cdot |x[j][i] - M[i]|^2
   $$
   Where:
   - \( x[j][i] \): Opinion of node \( j \) with respect to agent \( i \)
   - \( M[i] \): Shared opinion value for agent \( i \)
   - \( w[i][j] \): Weight of the connection between agent \( i \) and its neighbor \( j \)

2. **Global Cost**:  
   $$
   \text{cost}_{\text{global}} = \alpha \cdot |X̄ - \text{fav}| \cdot \beta \sqrt{k}
   $$
   Where:
   - \( \alpha, \beta \): Constants controlling trade-off between local and global optimization
   - \( X̄ \): Mean opinion
   - \( \text{fav} \): Favorable opinion target
   - \( k \): Iteration index

## Algorithm Dynamics

### Initialization
1. Initialize opinions \( X \), shared opinion values \( M[i][k] \), and constants \( \alpha \) and \( \beta \).
2. Define constant connection weights \( w[i][j] \).

### Iterative Optimization
The optimization process follows these steps:
1. Adjust opinions \( X \) using optimization results.
2. Update shared opinion values \( M[i][k] \) for the next iteration.
3. Calculate the mean opinion \( X̄ \) and other metrics based on updated values.

## Centralized Training and Decentralized Execution

The system leverages a centralized training phase where global information is utilized for optimizing parameters, but execution happens in a fully decentralized manner.

### MAAC Framework
The **Multi-Agent Actor-Critic (MAAC)** approach is used:
1. **Actor Network**:
   - Inputs: Local observations
   - Outputs: Probability distribution of actions
   - Architecture:
     - Dense layers with Mish activation functions
     - Dropout and Layer Normalization for stability
   - Output formula:
     $$
     a_i = \text{softsign}(\text{Actor}(s_i))
     $$

2. **Critic Network**:
   - Inputs: Joint states \( [s_1, s_2, \ldots, s_N] \) and actions \( [a_1, a_2, \ldots, a_N] \)
   - Outputs: Action-value \( Q(s, a_1, \ldots, a_N) \)
   - Loss function:
     $$
     \text{Loss}_{\text{critic}} = \frac{1}{N} \sum_{i=1}^N \left( y_i - Q(s, a_1, \ldots, a_N) \right)^2
     $$
   - Target value \( y_i \):
     $$
     y_i = r_i + \gamma Q(s', \pi_1(s'), \ldots, \pi_N(s'))
     $$
   - \( r_i \): Reward received, \( \gamma \): Discount factor.

## Policy Gradient Updates

The policy update for each agent \( i \) is derived as:
$$
\nabla_{\theta_i} J(\pi_i) = \mathbb{E}_{\pi_i} \left[ \sum_{t=0}^T \nabla_{\theta_i} \log \pi_i(a_{i,t} | s_t) Q^\pi(s_t, a) \right]
$$
Where:
- \( \theta_i \): Parameters of agent \( i \)'s policy \( \pi_i \)
- \( a \): Joint action vector
- \( Q^\pi \): Action-value function.

## Counterfactual Multi-Agent (COMA) Extension

To enhance learning efficiency:
1. A **counterfactual baseline** \( b_i(s_t) \) is subtracted during gradient computation:
   $$
   \nabla_{\theta_i} J(\pi_i) = \mathbb{E}_{\pi_i} \left[ \sum_{t=0}^T \nabla_{\theta_i} \log \pi_i(a_{i,t} | s_t) \left( Q^\pi(s_t, a) - b_i(s_t) \right) \right]
   $$
2. The joint action-value function \( Q^\pi(s, a) \) is estimated as:
   $$
   Q^\pi(s_t, a) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^T r_t \right]
   $$
