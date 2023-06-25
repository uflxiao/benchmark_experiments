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

# model.load_state_dict(torch.load(model_path, map_location=device))

policy_e.policy_network.load_state_dict(theta_e)
policy.policy_network.load_state_dict(theta)

policy_gradient = PolicyGradient(env, policy, 0.1, policy_e).search()





