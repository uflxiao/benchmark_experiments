from environments.gridworld import GridWorld
from utils.deep_nn_policy import DeepNeuralNetworkPolicy
import torch
import yaml
import wandb

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

env = GridWorld(width=env_size, height=env_size)

policy = DeepNeuralNetworkPolicy(
    env, state_space=len(env.get_initial_state()), action_space=4
)


model_path = f"results/policy/pi_{policy_id}"
state_dict  = torch.load(model_path)
policy.policy_network.load_state_dict(state_dict)

lst_v = []
lst_est = []

for i in range(num_training_episode):
    state = env.get_initial_state()
    acc_reward = 0
    acc_rho = 1

    while not env.is_terminal(state):
        action = policy.select_action(state)
        next_state, reward = env.execute(state, action)
        
        acc_rho = acc_rho * policy.get_probability(state, action)/policy.get_probability(state, action) 
        acc_reward += reward * acc_rho

        state = next_state

    lst_v.append(acc_reward)
    if lst_est:
            lst_est.append((lst_est[-1]*len(lst_est)+acc_reward)/len(lst_est)  )
    else:
        lst_est.append(acc_reward)


    wandb.log({"estimate": lst_est[-1], "episodes": i}) 

