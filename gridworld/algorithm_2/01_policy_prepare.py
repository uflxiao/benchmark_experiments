from environments.gridworld import GridWorld
from policy_gradient import PolicyGradient
from utils.deep_nn_policy import DeepNeuralNetworkPolicy

gridworld = GridWorld(width=3, height=3)
gridworld.visualise()

policy = DeepNeuralNetworkPolicy(
    gridworld, state_space=len(gridworld.get_initial_state()), action_space=4
)
policy_gradient = PolicyGradient(gridworld, policy, alpha=0.1).execute(episodes=1000)
gridworld_image = gridworld.visualise_stochastic_policy(policy)

