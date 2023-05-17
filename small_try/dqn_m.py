import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import count

import random
import math

env = gym.make("InvertedPendulum-v4")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

steps_done = 0

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

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to(device)).squeeze().cpu().numpy()
    else:
        return env.action_space.sample()



total_episodes = 5000

state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

n_actions = env.action_space.shape[0]
n_observations = env.observation_space.shape[0]

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

total = 0
for t in count():
    action = select_action(state)
    print(action)
    print(env.step(action))
    # obs, reward, terminated, truncated, _ = env.step(action.item())
    # print(obs)
    # print(reward)
    total += 1
    if total >= 10:
        break
        
print("finish")




   
    
