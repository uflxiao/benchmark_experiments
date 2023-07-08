from environments.gridworld import GridWorld
from utils.deep_nn_policy import DeepNeuralNetworkPolicy
from policy_gradient import PolicyGradient
import torch
import copy
import wandb
import yaml

wandb_switch = True

if wandb_switch:
    with open('./02_config_sweep.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    run = wandb.init(config=config)

    env_size = wandb.config.env_size
    policy_id = wandb.config.policy_id
    num_training_episode = wandb.config.num_train_episode

else:
    env_size = 3
    policy_id = 850
    num_training_episode = 1000

model_path = f"results/policy/pi_850"
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

data = PolicyGradient(env, policy, 0.1, policy_e, trajectory).generate(num_training_episode)

lst_est = []
for i in range(len(data)):
    if lst_est:
        reward = sum(data[i]["rewards"])
        lst_est.append((lst_est[-1] * len(lst_est) + reward) / (len(lst_est) + 1))
    else:
        lst_est.append(reward)
    if wandb_switch:
        wandb.log({"estimate": lst_est[-1], "episodes": 6})
