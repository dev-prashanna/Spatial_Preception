import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from reinforced_model.reinforced_model.gym import RobotEnv
from reinforced_model.reinforced_model.deep_q_network import QNet
import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("reinforced_model", exist_ok=True)
log_file = open("reinforced_model/training_3_log.txt", "w")

episodes=15000

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

env = RobotEnv()

qnet = QNet().to(device)
target_net = QNet().to(device)

target_net.load_state_dict(qnet.state_dict())

memory = deque(maxlen=50000)

optimizer = optim.Adam(qnet.parameters(), lr=0.001)

loss_fn = nn.SmoothL1Loss()

gamma = 0.99
epsilon = 1.0
steps = 0
episode_reward=[]
episode_step=[]

def select_action(state):

    global epsilon

    if random.random() < epsilon:
        return random.randint(0,2)

    state = torch.FloatTensor(state).unsqueeze(0).to(device)

    qnet.eval()

    with torch.no_grad():
        q_values = qnet(state)

    return torch.argmax(q_values, dim=1).item()

def train():

    if len(memory) < 1000:
        return

    batch = random.sample(memory, 32)

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

    states = torch.FloatTensor(states).to(device)

    actions = torch.LongTensor(actions).to(device)

    rewards = torch.FloatTensor(rewards).to(device)

    next_states = torch.FloatTensor(next_states).to(device)

    dones = torch.FloatTensor(dones).to(device)


    q_values = qnet(states)

    chosen_q_values = q_values.gather(
        1,
        actions.unsqueeze(1)
    ).squeeze()


    with torch.no_grad():

        next_q_values = target_net(next_states)

        max_next_q = torch.max(
            next_q_values,
            dim=1
        )[0]


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


def update_target():
    target_net.load_state_dict(qnet.state_dict())
    
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

        train()

        steps += 1
        episode_step_count+=1
        if steps % 1000== 0:
            update_target()

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


    epsilon = max(0.1, epsilon * 0.999)

    episode_reward.append(total_reward)
    episode_step.append(episode_step_count)

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

   
print("Training finished")
log_file.close()

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

