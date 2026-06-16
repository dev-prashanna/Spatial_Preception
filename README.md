## Learning Spatial Intelligence for Autonomous Navigation via Deep Reinforcement Learning with Structured Perception Modeling

### Goal
This project studies **autonomous obstacle avoidance** using two learning paradigms under the **same simulation constraints**:

1. **Supervised Imitation Learning** (Behavior Cloning)
2. **Deep Reinforcement Learning** using a **Deep Q-Network (DQN)**

We compare:
- **Generalization**
- **Stability**
- **Learning efficiency**

---

## Repository Structure (clean + easy to understand)

```text
project-root/
├── data/
│   ├── dataset.csv                      # Labeled expert sensor→action dataset
│   └── (optional) dataset_notes.md
│
├── models/
│   └── supervised/
│       └── model.pth                  # Saved imitation model weights
│
├── src/
│   ├── supervised/
│   │   ├── sim.py                      # Expert controller + trajectory generation
│   │   ├── datasetprocess.ipynb       # Preprocessing / normalization
│   │   ├── neuralnet.py                # Feedforward model for behavior cloning
│   │   └── train.ipynb               # Train imitation model
│   │
│   └── reinforcement/
│       ├── gym.py                      # Custom RL environment (physics + reward)
│       ├── deepqnetwork.py            # Q-network architecture
│       ├── reinforced_train.py       # DQN training loop (epsilon-greedy)
│       └── (optional) versioning.md
│
├── docs/
│   └── project_overview.md            # Extra explanations / diagrams
│
└── README.md                          # This file
```

> If you want to keep your current filenames exactly, that’s fine—this structure is mainly about making what each file *does* obvious.

---

## 1. Problem Definition

The task is to learn a policy that controls two motors to **avoid obstacles** using sensor observations.  
We evaluate:

- **Imitation learning**: learns directly from expert trajectories (no exploration)
- **DQN**: learns by interacting with the environment (with exploration and reward-driven learning)

---

## 2. Supervised Learning Framework (Imitation Learning)

### 2.1 Methodology
A deterministic **rule-based expert controller** in `sim.py` generates training trajectories.

These expert trajectories are converted into a dataset:
- **Inputs**: sensor readings (+ optional state variables depending on version)
- **Outputs**: motor commands produced by the expert

Then a feedforward neural network is trained to regress expert actions.

### 2.2 Dataset

**Inputs (typical)**
- left sensor distance
- right sensor distance
- front sensor distance
- additional state variables depending on version

**Outputs**
- `motor_A` velocity
- `motor_B` velocity

Dataset file:
- `data/dataset.csv`

### 2.3 Model
A feedforward neural network trained with **Mean Squared Error (MSE)**:


### 2.4 Limitation
Imitation learning depends heavily on the distribution of expert trajectories.
When the robot encounters states not covered by expert data (**distribution shift**), the model may fail because it has **no exploration mechanism**.

---

## 3. Reinforcement Learning Framework (DQN)

### 3.1 Learning Objective
The agent learns a policy that maximizes expected cumulative reward using Q-learning:


Where:
- \(s\): state
- \(a\): action
- \(r\): reward
- \(\gamma\): discount factor

---

## 4. DQN Evolution (Two Versions)

You currently have two implementations. The differences matter for comparison, so they’re documented clearly below.

---

### 4.1 Version 1 — Baseline DQN

**State Representation**
- 5-dimensional state vector
- limited environmental awareness

**Neural Network**
- smaller architecture
- no advanced initialization

**Training Configuration**
- epochs: ~2000
- loss: MSE
- basic reward function

**Reward (baseline)**
```text
reward = 5 * progress - 0.05
```

**Observed Issues**
- unstable convergence
- poor generalization
- reward stagnation
- high variance in Q-value updates

---

### 4.2 Version 2 — Improved DQN System

#### State Representation
- 6-dimensional state vector
- includes:
  - ray-based perception
  - goal direction
  - orientation

#### Neural Network Improvements
- architecture: `6 → 128 → 128 → 128 → 64 → 3`
- activation: Leaky ReLU (`alpha = 0.01`)
- weight initialization: Xavier initialization

#### Environment Fixes
- corrected coordinate system issue `(x,y)` indexing to `(y,x)`
- improved spatial consistency in grid representation

#### Reward Function Redesign
```text
reward = 2 * progress - 0.01
goal reward = +10
collision reward = -10
progress = old_distance - new_distance
```

#### Training Improvements
- epochs increased to 15000
- replay buffer size increased: 10000 → 50000
- loss: Smooth L1 (Huber loss)
- warm-up threshold increased: 32 → 1000 samples

#### Exploration Strategy
Epsilon decays as:
```text
epsilon = 0.999 * epsilon
```
with a minimum exploration bound.

---

## Suggested How-To Sections (optional, but helpful for GitHub)

### Supervised Training
Run:
- `src/supervised/train.ipynb`

Pipeline:
- generate expert data in `src/supervised/sim.py`
- preprocess dataset using `src/supervised/datasetprocess.ipynb`
- train network with `src/supervised/train.ipynb`

### Reinforcement Training
Run:
- `src/reinforcement/reinforced_train.py`

Environment:
- `src/reinforcement/gym.py` (reward + physics)

Q-network:
- `src/reinforcement/deepqnetwork.py`

---

## Quick Mapping: File → Purpose

| File | Component | Purpose |
|------|-----------|---------|
| `sim.py` | Supervised | expert policy + simulator for trajectories |
| `dataset.csv` | Data | sensor→expert action pairs |
| `datasetprocess.ipynb` | Supervised | preprocessing + normalization |
| `neuralnet.py` | Supervised | behavior cloning model |
| `train.ipynb` | Supervised | train imitation model |
| `gym.py` | RL | RL environment + reward |
| `deepqnetwork.py` | RL | Q-network definition |
| `reinforced_train.py` | RL | DQN training loop |

---
