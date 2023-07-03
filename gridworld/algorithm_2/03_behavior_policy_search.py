from environments.gridworld import GridWorld
from utils.deep_nn_policy import DeepNeuralNetworkPolicy
from policy_gradient import PolicyGradient
import torch
import copy
import wandb
import yaml

wandb_switch = True

if wandb_switch:
    with open('./03_config_sweep.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    run = wandb.init(config=config)

    env_size = wandb.config.env_size
    policy_id = wandb.config.policy_id
    num_training_episode = wandb.config.num_train_episode

else:
    env_size = 3
    policy_id = 850
    num_training_episode = 1000

model_path = f"results/policy/pi_{policy_id}"
theta_e  = torch.load(model_path)
theta = copy.deepcopy(theta_e)

env = GridWorld(width=env_size, height=env_size)

policy_e = DeepNeuralNetworkPolicy(
    env, state_space=len(env.get_initial_state()), action_space=4
)

policy = DeepNeuralNetworkPolicy(
    env, state_space=len(env.get_initial_state()), action_space=4
)


policy_e.policy_network.load_state_dict(theta_e)
policy.policy_network.load_state_dict(theta)

policy_gradient = PolicyGradient(env, policy, 0.1, policy_e).search(num_training_episode)





