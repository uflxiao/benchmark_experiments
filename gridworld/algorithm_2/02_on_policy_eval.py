import pickle
from environments.gridworld import GridWorld
from utils.deep_nn_policy import DeepNeuralNetworkPolicy
import torch

env = GridWorld(width=3, height=3)

policy = DeepNeuralNetworkPolicy(
    env, state_space=len(env.get_initial_state()), action_space=4
)


model_path = f"results/policy/pi_850"
state_dict  = torch.load(model_path)
policy.policy_network.load_state_dict(state_dict)

lst_v = []
lst_est = []

for i in range(100):
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


    print(f"estimate: {lst_est[-1]} episodes: {i}")  

