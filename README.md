# Spatial_Preception
Robot Learning: Supervised vs Reinforcement Learning Comparison

This repository explores and compares supervised learning and reinforcement learning (DQN) for autonomous robot obstacle avoidance using a custom simulation environment.

1. Supervised Learning Pipeline

Initially, a teacher-driven system was implemented in sim.py, where a pre-programmed rule-based controller generated optimal motor actions for obstacle avoidance. This teacher was used to collect structured datasets in CSV format containing:

Sensor inputs: left, right, front distances
Outputs: Motor A and Motor B control values
Files used:
sim.py – simulation + teacher policy
dataset.csv – collected training data
neuralnet.py – supervised neural model
train.ipynb – training pipeline
model.pth – trained weights
databaseprocess.ipynb – preprocessing

Observation: Supervised learning struggled to generalize beyond the teacher’s behavior and required large datasets, making it inefficient for dynamic adaptation.

2. Reinforcement Learning (DQN) Pipeline

A Deep Q-Network agent was implemented to learn directly through interaction with the environment.

Files used:
gym.py – custom RL environment and reward system
deepqnetwork.py – Q-value approximator neural network
reinforced_train.py – training loop with epsilon-greedy policy
Key techniques:
Experience replay
Target Q-value updates
Epsilon-greedy exploration
3. Observed Issues in RL Training

Despite convergence, the agent exhibits reward stagnation (~constant ~51 reward) due to:

1. Reward plateau (flat optimization landscape)

The agent converges to a stable policy where:

E[R]≈constant

No meaningful gradient exists between states, causing training to stall.

2. Vanishing TD error

The temporal difference update:

TD error = reward + gamma * max Q(next_state, next_action) - Q(current_state, action)

approaches zero, reducing learning updates and freezing Q-values.

3. Over-exploitation due to epsilon decay

Low epsilon values reduce exploration, locking the agent into suboptimal but stable policies.

4. Local optimum convergence

The agent stabilizes in a “good enough” policy rather than discovering globally optimal behavior.

4. Future Improvements

This project is open for improvements and experimentation with advanced RL techniques such as:

Proximal Policy Optimization (PPO)
Soft Actor-Critic (SAC)
Intrinsic curiosity-based exploration
Reward shaping for dense feedback
Noisy networks instead of epsilon-greedy
5. Goal

The purpose of this project is to experimentally compare:

Imitation learning (supervised teacher-based control)
Reinforcement learning (self-discovered policies)

and analyze their limitations in real-world robotic navigation tasks.
