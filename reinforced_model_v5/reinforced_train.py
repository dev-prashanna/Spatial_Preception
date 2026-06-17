import os
import json
import math
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from gym import RobotEnv
from deep_q_network import QNet

RANDOM_SEED = 42
EPISODES = 15000
MEMORY_SIZE = 100000
BATCH_SIZE = 64
MIN_REPLAY_SIZE = 10000

GAMMA = 0.99
LEARNING_RATE = 0.0003
TAU = 0.005  

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = RobotEnv()

qnet = QNet().to(device)
target_net = QNet().to(device)
target_net.load_state_dict(qnet.state_dict())

memory = deque(maxlen=MEMORY_SIZE)
optimizer = optim.Adam(qnet.parameters(), lr=LEARNING_RATE)
loss_fn = nn.SmoothL1Loss()

def soft_update(target_model, source_model, tau=0.005):
    for target_param, local_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def select_action(state, current_epsilon):
    if random.random() < current_epsilon:
        return random.randint(0, 2)

    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = qnet(state_tensor)
    return torch.argmax(q_values, dim=1).item()

def train():
    if len(memory) < MIN_REPLAY_SIZE:
        return False

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(np.array(actions)).to(device)
    rewards = torch.FloatTensor(np.array(rewards)).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(np.array(dones)).to(device)

    q_values = qnet(states)
    chosen_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

    with torch.no_grad():
        next_actions = qnet(next_states).argmax(dim=1)
        next_q_target = target_net(next_states)
        max_next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze()

    target_q = rewards + GAMMA * max_next_q * (1.0 - dones)

    loss = loss_fn(chosen_q_values, target_q)
    optimizer.zero_grad()
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(qnet.parameters(), max_norm=1.0)
    optimizer.step()
    return True

def export_weights_json(model, path="reinforced_model/qnet_weights.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    layers = []

    for i, layer in enumerate(model.network):
        if isinstance(layer, nn.Linear):
            layer_data = {
                "layer_index": i,
                "layer_type": "Linear",
                "input_features": layer.in_features,
                "output_features": layer.out_features,
                "W": layer.weight.detach().cpu().numpy().tolist(),
                "b": layer.bias.detach().cpu().numpy().tolist()
            }
            layers.append(layer_data)

    with open(path, "w") as f:
        json.dump(layers, f, indent=4)
    print(f"Weights saved → {os.path.abspath(path)}")

os.makedirs("reinforced_model", exist_ok=True)

episode_reward = []
episode_step = []
termination_history = []
successful_steps = []

goal_count = 0
collision_count = 0
timeout_count = 0
global_step_counter = 0
epsilon = 1.0

with open("reinforced_model/training_3_log.txt", "w") as log_file:
    for episode in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0.0
        episode_step_count = 0

        while not done:
            action = select_action(state, epsilon)
            next_state, reward, done = env.step(action)
            
            memory.append((state, action, reward, next_state, done))
            
            global_step_counter += 1
            episode_step_count += 1
            
            if train():
                soft_update(target_net, qnet, tau=TAU)

            state = next_state
            total_reward += reward

        reason = env.termination_reason()
        termination_history.append(reason)

        if reason == "goal":
            goal_count += 1
            successful_steps.append(episode_step_count)
        elif reason == "collision":
            collision_count += 1
        elif reason == "timeout":
            timeout_count += 1

        episode_reward.append(total_reward)
        episode_step.append(episode_step_count)
        
        epsilon = max(0.05, epsilon * 0.9999)

        log_message = f"Episode: {episode} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.4f} | Reason: {reason}"
        print(log_message)
        log_file.write(log_message + "\n")

mean_reward_last100 = np.mean(episode_reward[-100:])
recent_termination = termination_history[-100:]
recent_steps = episode_step[-100:]

goal_steps_last100 = [steps for r, steps in zip(recent_termination, recent_steps) if r == "goal"]
mean_steps_goal = np.mean(goal_steps_last100) if goal_steps_last100 else None

print("\n--- Training Run Metrics ---")
print(f"Mean Reward (Last 100): {mean_reward_last100:.2f}")
print(f"Mean Steps to Goal (Last 100): {mean_steps_goal}")
print(f"Goal Rate: {(goal_count / EPISODES) * 100:.2f}%")
print(f"Collision Rate: {(collision_count / EPISODES) * 100:.2f}%")
print(f"Timeout Rate: {(timeout_count / EPISODES) * 100:.2f}%")

plt.figure()
plt.plot(episode_reward)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward vs Episode")
plt.savefig("reinforced_model/reward_vs_episode.png")
plt.close()

plt.figure()
plt.plot(episode_step)
plt.xlabel("Episode")
plt.ylabel("Steps Survived")
plt.title("Steps vs Episode")
plt.savefig("reinforced_model/steps_vs_episode.png")
plt.close()

window = 50
if len(episode_reward) >= window:
    moving_avg = np.convolve(episode_reward, np.ones(window)/window, mode='valid')
    plt.figure()
    plt.plot(moving_avg)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("DQN Reward vs Episode (Moving Average)")
    plt.savefig("reinforced_model/dqn_plateau.png")
    plt.close()

export_weights_json(qnet)
torch.save(qnet.state_dict(), "reinforced_model/qnet_weights.pth")
print("PyTorch model state dict (.pth) saved successfully!")
print("Training execution process finished successfully.")

