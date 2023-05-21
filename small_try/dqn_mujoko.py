import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque
from itertools import count
import random
import numpy as np
import matplotlib.pyplot as plt

#debug
import sys

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent:
    # def __init__(self, n_observations, n_actions, lr=1e-3, gamma=0.99, epsilon=0.1):
    def __init__(self, n_observations, n_actions, lr=2.5e-4, gamma=0.99, epsilon=0.05):
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        self.policy_net = DQN(n_observations, n_actions)
        self.target_net = DQN(n_observations, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        state_tensor = state_tensor.unsqueeze(0)

        if random.random() < self.epsilon:
            action = np.random.uniform(-3, 3)  # Generate a random scalar value within the action range
        else:
            # Convert state to tensor
                state_tensor = torch.tensor(state, dtype=torch.float32)
                state_tensor = state_tensor.unsqueeze(0)

                # Get Q-values from the policy network
                q_values = self.policy_net(state_tensor)

                # Convert Q-values to numpy array
                q_values = q_values.squeeze().detach().numpy()

                # Map Q-values to valid indices
                action = np.argmax(q_values)
                print(action)

        return np.array([action])  # Convert the scalar value to a 1-dimensional array

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, env, total_episodes=10000, batch_size=32, target_update_freq=100):
        episode_rewards = []

        for episode in range(total_episodes):
            state, info= env.reset()

            total_reward = 0
            done = False

            for t in count():
                action = self.select_action(state)
               
                next_state, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    done = True

                total_reward += reward

                self.store_transition(state, action, reward, next_state, done)
                state = next_state

                if len(self.memory) >= batch_size:
                        batch = random.sample(self.memory, batch_size)
                        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

                        state_batch = torch.tensor(state_batch, dtype=torch.float32)
                        action_batch = torch.tensor(action_batch, dtype=torch.long).flatten()  # Flatten the action_batch tensor
                        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1)
                        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)
                        done_batch = torch.tensor(done_batch, dtype=torch.float32).unsqueeze(1)

                        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

                        next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
                        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

                        loss = F.smooth_l1_loss(q_values, expected_q_values)

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                if episode % target_update_freq == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                if done:
                        if info:
                             print(info)
                             sys.exit()
                        if episode % 100 == 0:
                            print("Episode:", episode, "Total Reward:", total_reward)
                        episode_rewards.append(total_reward) 
                        break

        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Progress")
        plt.savefig("mujoko.png")


if __name__ == "__main__":
    env = gym.make("InvertedPendulum-v4")
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    agent = DQNAgent(n_observations, n_actions)
    agent.train(env=env)
   