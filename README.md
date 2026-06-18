# Autonomous Edge Robotics: Behavioral Cloning vs Deep Q-Networks

A comparative study of two machine learning paradigms for autonomous obstacle avoidance
in constrained 2D simulation environments.

---

## Repository Structure

```text
dev-prashanna/
├── supervised_model_v2/
│   ├── neural_net.py                  # Feedforward MLP for behavioral cloning
│   ├── sim_v1.py                      # Expert controller + trajectory generation
│   └── train.ipynb                    # Training pipeline
│
├── reinforced_model_v2/               # Baseline DQN
├── reinforced_model_v3/               # Improved DQN
├── reinforced_model_v5/               # Final DQN version
│   ├── deep_q_network_v4.py           # Q-network architecture
│   ├── gym_v4.py                      # Custom RL environment
│   └── reinforced_train_v4.py        # Training loop
│
├── dataset.csv                        # 100,000 expert state-action pairs
├── model_v1.pth                       # Supervised model weights
├── weights.json                       # DQN exported weights
│
├── dqn_plateau_training_4.png         # Reward curve
├── reward_vs_episode_training_4.png   # Reward vs episode plot
├── steps_vs_episode_training_4.png    # Steps vs episode plot
│
├── training_1_log.txt                 # Training logs
├── training_2_log.txt
├── training_3_log.txt
│
├── Blueprint.pdf                      # Project blueprint
├── paper_draft1(3).pdf                # Research paper draft
└── README.md
```

---

## Overview

| | Behavioral Cloning | Deep Q-Network |
|---|---|---|
| **Paradigm** | Supervised Imitation Learning | Model-Free Reinforcement Learning |
| **Environment** | Continuous 800×600 obstacle space | Discrete 10×10 grid with narrow corridor |
| **Input** | 3 raycast sensor readings | 7-dimensional state vector |
| **Output** | Left/right motor velocities | Discrete action (forward / turn left / turn right) |
| **Training data** | 100,000 expert trajectories | 15,000 episodes |
| **Loss** | Mean Squared Error | Smooth L1 (Huber) |

---

## Environments

**Environment A — Behavioral Cloning**
Continuous 800×600 coordinate space with randomized rectangular obstacles and boundary
walls. A rule-based expert controller generates labeled trajectories.

**Environment B — Deep Q-Network**
Discrete 10×10 grid world with a horizontal wall barrier at row y=5 and a narrow
two-cell corridor at x∈{4,5}. The agent must discover the corridor through exploration.

---

## Models

### Behavioral Cloning
- Architecture: `3 → 32 → 32 → 16 → 2`
- Activation: ReLU (hidden), Sigmoid (output)
- Optimizer: Adam

### Deep Q-Network (Final Version)
- Architecture: `7 → 256 → 256 → 128 → 3`
- Activation: LeakyReLU (slope 0.01) + Layer Normalization
- Method: Double-DQN with soft target updates (τ = 0.005)
- Replay buffer: 100,000 transitions

---

## Key Results

| Metric | Value |
|---|---|
| Training Episodes | 15,000 |
| Mean Episodic Reward | 21.24 |
| Goal Success Rate | 47.47% |
| Wall Collision Rate | 52.51% |
| Episode Timeout Rate | 0.03% |

---

## Paper

A full write-up covering the simulation design, network architectures, reward shaping,
failure analysis, and proposed improvements is available as `paper_draft1(3).pdf`.

---

## References

- Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature.*
- Van Hasselt et al. (2016). Deep RL with double Q-learning. *AAAI.*
- Pomerleau (1989). ALVINN. *NeurIPS.*
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction.* MIT Press.
