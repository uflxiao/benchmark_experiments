from environments.gridworld import GridWorld
from utils.deep_nn_policy import DeepNeuralNetworkPolicy
from policy_gradient import PolicyGradient
import torch
import copy

model_path = f"results/policy/pi_100"
theta_e  = torch.load(model_path)
theta = copy.deepcopy(theta_e)

env = GridWorld(width=3, height=3)

policy_e = DeepNeuralNetworkPolicy(
    env, state_space=len(env.get_initial_state()), action_space=4
)

policy = DeepNeuralNetworkPolicy(
    env, state_space=len(env.get_initial_state()), action_space=4
)

policy_e.policy_network.load_state_dict(theta_e)

trajectory = []

for i in range(100):
    d = {"states":[], "actions": [], "rewards": []}
    state = env.get_initial_state()

    while not env.is_terminal(state):
        action = policy_e.select_action(state)
        next_state, reward = env.execute(state, action)

        d["states"].append(state)
        d["actions"].append(action)
        d["rewards"].append(reward)

        state = next_state
    
    trajectory.append(d)

data = PolicyGradient(env, policy, 0.1, policy_e, trajectory).generate()
print(len(data))

