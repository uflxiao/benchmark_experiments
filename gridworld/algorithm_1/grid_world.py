from environments.GridWorld import GridWorld
from utilities.Policy import *
import numpy as np

#debug
import sys

env = GridWorld(width = 3,  \
        t_max = 3, \
        offline_data_number = int(1)
        )

class PolicyGradient:
    def __init__(self, env, policy, alpha) -> None:
        super().__init__()
        self.alpha = alpha
        self.env = env
        self.policy = policy

    def execute(self, episodes=100):
        for _ in range(episodes):
            actions = []
            states = []
            rewards = []
            
            state, _ = env.reset()
            state = env.state_to_phi_device(state, self.device)
            episode_reward = 0
            done = False

            while not done:
                action = self.policy.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = env.state_to_phi_device(next_state, self.device)

                

agent = PolicyGradient(env, None, None)
agent.execute()