import gym
from algos.ppo import Agent

env = gym.make('CartPole-v0', render_mode="human")
N = 20
batch_size = 5
n_epochs = 4
alpha = 0.0003
agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                alpha=alpha, n_epochs=n_epochs, 
                input_dims=env.observation_space.shape)

agent.load_models()

for _ in range(10):
        obs, info = env.reset()
        env.render()
        while True:
            action, _, _ = agent.choose_action(obs)
            print("state", obs)
            obs, _, done, _, _ = env.step(action)
            env.render()
            if done == True:
                break