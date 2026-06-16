#Learning Spatial Intelligence for Autonomous Navigation via Deep Reinforcement Learning with Structured Perception Modeling

1. Problem Definition

This project investigates autonomous obstacle avoidance using two learning paradigms: supervised imitation learning and deep reinforcement learning (Deep Q-Network). The objective is to evaluate differences in generalization, stability, and learning efficiency under identical simulation constraints.

2. Repository Structure
2.1 Supervised Learning Pipeline
sim.py : Rule-based expert controller and environment simulator used to generate training trajectories
dataset.csv : Collected dataset containing sensor-action mappings from expert policy
databaseprocess.ipynb : Data preprocessing and normalization pipeline
neuralnet.py : Feedforward neural network for behavior cloning
train.ipynb : Training pipeline for supervised regression model
model.pth : Saved weights of trained supervised model
2.2 Reinforcement Learning Pipeline
gym.py : Custom reinforcement learning environment with reward function and physics simulation
deepqnetwork.py : Q-value approximation neural network
reinforced_train.py : DQN training loop using epsilon-greedy exploration
3. Supervised Learning Framework (Imitation Learning)
3.1 Methodology

A deterministic rule-based expert policy implemented in sim.py generates optimal control signals. These are used as labeled data for supervised learning.

3.2 Dataset

Input features:

left sensor distance
right sensor distance
front sensor distance
additional state variables depending on version

Output:

motor_A velocity
motor_B velocity

Dataset stored in dataset.csv.

3.3 Model

A feedforward neural network is trained to minimize regression loss:

loss = mean squared error between predicted_action and expert_action

3.4 Limitation

The model exhibits strong dependency on expert trajectories and fails under distributional shift due to absence of exploration mechanism.

4. Reinforcement Learning Framework (DQN)
4.1 Learning Objective

The agent learns a policy maximizing expected cumulative reward using Q-learning:

Q(s,a) = r + gamma * max(Q(s’,a’))

where:

s is state
a is action
r is reward
gamma is discount factor
5. Reinforcement Learning Evolution

This project contains two versions of DQN implementation.

5.1 Version 1 (Baseline DQN)
State Representation
5-dimensional state vector
limited environmental awareness
Neural Network
smaller architecture
no advanced initialization
Training Configuration
epochs: ~2000
loss function: mean squared error loss
basic reward function

reward = 5 * progress - 0.05

Issues Observed
unstable convergence
poor generalization
reward stagnation
high variance in Q-value updates
5.2 Version 2 (Improved DQN System)
State Representation
6-dimensional state vector
includes ray-based perception + goal direction + orientation
Neural Network Improvements
architecture: 6 → 128 → 128 → 128 → 64 → 3
activation: Leaky ReLU (alpha = 0.01)
weight initialization: Xavier initialization
Environment Fixes
corrected coordinate system from (x,y) indexing to (y,x) indexing
improved spatial consistency in grid representation
Reward Function Redesign

reward = 2 * progress - 0.01
goal reward = +10
collision reward = -10

where:
progress = old_distance - new_distance

Training Improvements
epochs increased to 15000
replay buffer size increased from 10000 to 50000
loss function changed to Smooth L1 loss (Huber loss)
warm-up threshold increased from 32 to 1000 samples
Exploration Strategy

epsilon updated as:

epsilon = 0.999 * epsilon

with minimum exploration bound applied.
