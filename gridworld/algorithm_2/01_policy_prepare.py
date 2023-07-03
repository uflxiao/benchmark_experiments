from environments.gridworld import GridWorld
from policy_gradient import PolicyGradient
from utils.deep_nn_policy import DeepNeuralNetworkPolicy
import wandb
import yaml

wandb_switch = True

if wandb_switch:
    with open('./01_config_sweep.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    run = wandb.init(config=config)

    env_size = wandb.config.env_size
    num_training_episode = wandb.config.num_train_episode

else:
    env_size = 5
    num_training_episode = 1000

gridworld = GridWorld(width=env_size, height=env_size)
gridworld.visualise()

policy = DeepNeuralNetworkPolicy(
    gridworld, state_space=len(gridworld.get_initial_state()), action_space=4
)
policy_gradient = PolicyGradient(gridworld, policy, alpha=0.1).execute(episodes=num_training_episode)
gridworld_image = gridworld.visualise_stochastic_policy(policy)

