import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_v2


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs]) # converts the list of numpy arrays into a single number array
    return state

if __name__ == "__main__":
    
    env = simple_v2.env(max_cycles=25, continuous_actions=True)
    env.reset()
    n_agents = env.num_agents
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space(f"agent_{i}").shape[0])
    critic_dims = sum(actor_dims)

    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, )
