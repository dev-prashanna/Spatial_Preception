# Autonomous Edge Robotics: Behavioral Cloning vs Deep Q-Networks

[![Python](https://img.shields.io/badge/Python-3.10+-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comparative study of two machine learning paradigms for autonomous obstacle avoidance in constrained 2D simulation environments.

## Overview

This research compares **Behavioral Cloning** (supervised imitation learning) and **Deep Q-Networks** (model-free reinforcement learning) for autonomous robot navigation. The study evaluates both approaches on obstacle avoidance tasks with different environment complexities and sensor configurations.

### Key Findings

- Behavioral Cloning achieves faster training but requires expert demonstration data
- DQN discovers novel strategies through exploration but requires more training episodes
- Both methods show distinct failure modes that inform hybrid approach design

## Comparison

| Aspect | Behavioral Cloning | Deep Q-Network |
|--------|-------------------|----------------|
| **Paradigm** | Supervised Imitation Learning | Model-Free Reinforcement Learning |
| **Environment** | Continuous 800x600 obstacle space | Discrete 10x10 grid with narrow corridor |
| **Input** | 3 raycast sensor readings | 7-dimensional state vector |
| **Output** | Left/right motor velocities | Discrete action (forward / turn left / turn right) |
| **Training Data** | 100,000 expert trajectories | 15,000 episodes |
| **Loss Function** | Mean Squared Error | Smooth L1 (Huber) |

## Architecture

### Behavioral Cloning

```
Input (3) -> Dense(32, ReLU) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Output(2, Sigmoid)
```

### Deep Q-Network

```
Input (7) -> Dense(256, LeakyReLU) -> LayerNorm -> Dense(256, LeakyReLU) -> LayerNorm -> Dense(128, LeakyReLU) -> Output(3)
```

**DQN Details:**
- Double-DQN with soft target updates (tau = 0.005)
- Replay buffer: 100,000 transitions
- Epsilon-greedy exploration with decay

## Environments

### Environment A -- Behavioral Cloning

Continuous 800x600 coordinate space with randomized rectangular obstacles and boundary walls. A rule-based expert controller generates labeled trajectories for supervised training.

### Environment B -- Deep Q-Network

Discrete 10x10 grid world with a horizontal wall barrier at row y=5 and a narrow two-cell corridor at x in {4,5}. The agent must discover the corridor through exploration.

## Results

| Metric | Behavioral Cloning | DQN |
|--------|-------------------|-----|
| Training Episodes | N/A (supervised) | 15,000 |
| Mean Episodic Reward | N/A | 21.24 |
| Goal Success Rate | TBD | 47.47% |
| Wall Collision Rate | TBD | 52.51% |
| Episode Timeout Rate | TBD | 0.03% |

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- NumPy
- Matplotlib

### Setup

```bash
# Clone the repository
git clone https://github.com/dev-prashanna/Spatial_Preception.git
cd Spatial_Preception

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install torch numpy matplotlib
```

## Usage

### Behavioral Cloning

```python
from supervised_model_v2.neural_net import BehavioralCloningNet
from supervised_model_v2.sim_v1 import ExpertController

# Initialize expert and generate trajectories
expert = ExpertController()
trajectories = expert.generate(num_trajectories=100000)

# Train model
model = BehavioralCloningNet()
model.train(trajectories, epochs=100)
```

### Deep Q-Network

```python
from reinforced_model_v5.deep_q_network_v4 import DQNAgent
from reinforced_model_v5.gym_v4 import GridWorldEnv

# Initialize environment and agent
env = GridWorldEnv()
agent = DQNAgent(state_dim=7, action_dim=3)

# Train agent
for episode in range(15000):
    state = env.reset()
    while True:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.update()
        state = next_state
        if done:
            break
```

## Project Structure

```
Spatial_Preception/
├── supervised_model_v2/
│   ├── neural_net.py              # Behavioral cloning MLP
│   ├── sim_v1.py                  # Expert controller
│   └── train.ipynb                # Training pipeline
│
├── reinforced_model_v5/
│   ├── deep_q_network_v4.py       # DQN architecture
│   ├── gym_v4.py                  # Custom RL environment
│   └── reinforced_train_v4.py     # Training loop
│
├── docs/
│   ├── Blueprint.pdf              # Project blueprint
│   └── paper_draft.pdf            # Research paper draft
│
├── figures/
│   ├── dqn_plateau_training.png   # Reward curve
│   ├── reward_vs_episode.png      # Reward vs episode
│   └── steps_vs_episode.png       # Steps vs episode
│
├── data/
│   ├── dataset.csv                # Expert trajectories
│   ├── weights.json               # DQN weights
│   └── model_v1.pth               # Supervised model weights
│
├── logs/
│   ├── training_1_log.txt
│   ├── training_2_log.txt
│   └── training_3_log.txt
│
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## Future Work

- [ ] Implement hybrid BC+DQN approach
- [ ] Test on continuous action spaces
- [ ] Add real-world robot deployment
- [ ] Explore curiosity-driven exploration
- [ ] Implement PPO and SAC baselines
- [ ] Add multi-agent scenarios

## References

- Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature.*
- Van Hasselt et al. (2016). Deep RL with double Q-learning. *AAAI.*
- Pomerleau (1989). ALVINN: An Autonomous Neural Network Vehicle. *NeurIPS.*
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction.* MIT Press.
- Ross et al. (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. *AISTATS.*

## Citation

If you use this work in your research, please cite:

```bibtex
@article{tiwari2026spatial,
  title={Autonomous Edge Robotics: Behavioral Cloning vs Deep Q-Networks},
  author={Tiwari, Prashanna},
  year={2026},
  url={https://github.com/dev-prashanna/Spatial_Preception}
}
```

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details.

## Author

**Prashanna Tiwari**
- GitHub: [@dev-prashanna](https://github.com/dev-prashanna)
- LinkedIn: [Prashanna Tiwari](https://www.linkedin.com/in/prashanna-tiwari-1b9a01398/)
