#plot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from utils.deep_nn_policy import DeepNeuralNetworkPolicy

#debug
import sys


class PolicyGradient:
    def __init__(self, mdp, policy, alpha, policy_e=None, data=[]) -> None:
        super().__init__()
        self.alpha = alpha  # Learning rate (gradient update step-size)
        self.mdp = mdp
        self.policy = policy
        #plot
        self.rewards_over_episodes = []
        #generated policy data
        self.data = data
        self.policy_e = policy_e

    """ Generate and store an entire episode trajectory to use to update the policy """

    def execute(self, episodes=100):
        for i in range(episodes):
            actions = []
            states = []
            rewards = []

            state = self.mdp.get_initial_state()
            episode_reward = 0
            while not self.mdp.is_terminal(state):
                action = self.policy.select_action(state)
                next_state, reward = self.mdp.execute(state, action)

                # Store the information from this step of the trajectory
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            deltas = self.calculate_deltas(rewards)
            self.policy.update(states=states, actions=actions, deltas=deltas)
            #plot
            self.rewards_over_episodes.append(sum(rewards))
            #generate policy data
            if i in [100, 850]:
                policy_path = f"results/policy/pi_{i}"
                torch.save(self.policy.policy_network.state_dict(), policy_path)

    def generate(self, episodes=100):
        k = sum([len(x["rewards"]) for x in self.data])
        grad_loss = 0

        for i in range(len(self.data)):
            s = self.data[i]["states"]
            a = self.data[i]["actions"]

            grad_loss += 1/k * self.policy_e.grad_log(s, a)

        trajectory = self.data
        # trajectory = []
        for i in range(episodes):
            d = {"states":[], "actions": [], "rewards": []}

            state = self.mdp.get_initial_state()

            while not self.mdp.is_terminal(state):
                new = DeepNeuralNetworkPolicy(
                    self.mdp, state_space=len(self.mdp.get_initial_state()), action_space=4
                )

                new.policy_network.load_state_dict(self.policy_e.policy_network.state_dict().copy())
                new.compute_loss(grad_loss * 10)
                self.policy.policy_network.load_state_dict(new.policy_network.state_dict().copy())

                action = self.policy.select_action(state)
                next_state, reward = self.mdp.execute(state, action)

                d["states"].append(state)
                d["actions"].append(action)
                d["rewards"].append(reward)
                
                grad_loss = k/(k+1) * grad_loss + 1/(k+1) * self.policy_e.grad_log(state, action)
                k += 1
                state = next_state

            trajectory.append(d)
            
        return trajectory

    def calculate_deltas(self, rewards):
        """
        Generate a list of the discounted future rewards at each step of an episode
        Note that discounted_reward[T-2] = rewards[T-1] + discounted_reward[T-1] * gamma.
        We can use that pattern to populate the discounted_rewards array.
        """
        T = len(rewards)
        discounted_future_rewards = [0 for _ in range(T)]
        # The final discounted reward is the reward you get at that step
        discounted_future_rewards[T - 1] = rewards[T - 1]
        for t in reversed(range(0, T - 1)):
            discounted_future_rewards[t] = (
                rewards[t]
                + discounted_future_rewards[t + 1] * self.mdp.get_discount_factor()
            )
        deltas = []
        for t in range(len(discounted_future_rewards)):
            deltas += [
                self.alpha
                * (self.mdp.get_discount_factor() ** t)
                * discounted_future_rewards[t]
            ]
        return deltas
    #plot
    def plot(self):
        print("Done")
        print(self.rewards_over_episodes)
        # rewards_to_plot = [[reward[0] for reward in rewards] for rewards in self.rewards_over_episodes]
        # df1 = pd.DataFrame(rewards_to_plot).melt()
        df1 = pd.DataFrame(self.rewards_over_episodes).melt()
        df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
        sns.set(style="darkgrid", context="talk", palette="rainbow")
        sns.lineplot(x="episodes", y="reward", data=df1).set(
            title="REINFORCE for InvertedPendulum-v4"
        )
        plt.savefig("new_plot.png")
