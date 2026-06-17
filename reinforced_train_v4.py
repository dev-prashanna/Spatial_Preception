import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from gym import RobotEnv
from deep_q_network import QNet
import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("reinforced_model", exist_ok=True)
log_file = open("reinforced_model/training_3_log.txt", "w")
recent_mean=1
episodes=20000
epsilon_fixed = False
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
      torch.cuda.manual_seed(42)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

env = RobotEnv()

qnet = QNet().to(device)
target_net = QNet().to(device)

target_net.load_state_dict(qnet.state_dict())

memory = deque(maxlen=100000)

optimizer = optim.Adam(qnet.parameters(), lr=0.001)

loss_fn = nn.SmoothL1Loss()

gamma = 0.99
epsilon = 1.0
steps = 0
episode_reward=[]
episode_step=[]

def soft_update():
     
     tau=0.005

     for target_param,local_param in zip(target_net.parameters(),qnet.parameters()):
          
          target_param.data.copy_(tau*local_param.data+(1-tau)*target_param.data)

def select_action(state):

    global epsilon

    if random.random() < epsilon:
        return random.randint(0,2)

    state = torch.FloatTensor(state).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = qnet(state)

    return torch.argmax(q_values, dim=1).item()

def train():

    if len(memory) < 5000:
        return

    batch = random.sample(memory, 64)

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    for state, action, reward, next_state, done in batch:

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)

    states      = torch.FloatTensor(np.array(states)).to(device)
    actions     = torch.LongTensor(np.array(actions)).to(device)
    rewards     = torch.FloatTensor(np.array(rewards)).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones       = torch.FloatTensor(np.array(dones)).to(device)

    q_values = qnet(states)

    chosen_q_values = q_values.gather(
        1,
        actions.unsqueeze(1)
    ).squeeze()

    with torch.no_grad():
            # online network selects best action
        next_actions = qnet(next_states).argmax(dim=1)

        # target network evaluates selected action
        next_q_target = target_net(next_states)

        max_next_q = next_q_target.gather(
            1,
            next_actions.unsqueeze(1)
        ).squeeze()


    target_q = rewards + gamma * max_next_q * (1 - dones)


    loss = loss_fn(
        chosen_q_values,
        target_q
    )

    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(  # added gradient clipping to prevent exploding
    qnet.parameters(),
    1.0
)
    optimizer.step()

    return True 

episode_reward=[]
episode_step=[]
goal_count = 0
collision_count = 0
timeout_count = 0

successful_steps = []      # steps when goal reached
termination_history = []

for episode in range(episodes):

    state = env.reset()
    done = False
    total_reward = 0
    episode_step_count=0
    while not done:

        action = select_action(state)

        next_state, reward, done = env.step(action)

        memory.append(
            (state, action, reward, next_state, done)
        )

        steps += 1
        episode_step_count+=1
        updated=train()
        if updated:
         soft_update()

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
    
    epsilon = max(0.05, epsilon*0.9995)
        

    log_message = f"Episode: {episode} Reward: {total_reward} Epsilon: {epsilon}"

    print(log_message)

    log_file.write(log_message + "\n")
mean_reward_last100 = np.mean(episode_reward[-100:])

# successful episodes in last 100
recent_termination = termination_history[-100:]
recent_steps = episode_step[-100:]

goal_steps_last100 = []

for reason, steps in zip(recent_termination, recent_steps):

        if reason == "goal":
            goal_steps_last100.append(steps)

if len(goal_steps_last100) > 0:
            
            mean_steps_goal = np.mean(goal_steps_last100)
else:
            mean_steps_goal = None


goal_rate = (goal_count / episodes) * 100
collision_rate = (collision_count / episodes) * 100
timeout_rate = (timeout_count / episodes) * 100

print("Mean Reward (last 100):", mean_reward_last100)
print("Mean Steps to Goal:", mean_steps_goal)
print("Goal Rate:", goal_rate)
print("Collision Rate:", collision_rate)
print("Timeout Rate:", timeout_rate)

plt.plot(episode_reward)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward vs Episode")

plt.savefig("reinforced_model/reward_vs_episode.png")

plt.show()

plt.close()

plt.plot(episode_step)
plt.xlabel("Episode")
plt.ylabel("Steps Survived")
plt.title("Steps vs Episode")

plt.savefig("reinforced_model/steps_vs_episode.png")

plt.show()
plt.close()

window = 50

moving_avg = np.convolve(
    episode_reward,
    np.ones(window)/window,
    mode='valid'
)

plt.plot(moving_avg)

plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("DQN Reward vs Episode")

plt.savefig("dqn_plateau.png")
plt.show()

import json

def export_weights_json(model, path="reinforced_model/qnet_weights.json"):

    # make sure folder exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    layers = []

    # go through every module inside Sequential
    for i, layer in enumerate(model.network):

        # save only Linear layers
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

    # save json file
    with open(path, "w") as f:

        json.dump(
            layers,
            f,
            indent=4
        )

    print("Weights saved →", os.path.abspath(path))

export_weights_json(qnet)

print("Training finished")
log_file.close()
