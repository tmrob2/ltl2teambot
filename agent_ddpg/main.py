from asyncore import write
from fileinput import filename
from ddpg2 import Agent
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter()

env = gym.make('LunarLanderContinuous-v2')

agent = Agent(alpha=2.5e-5, beta=2.5e-4, input_dims=[8], 
    tau=0.001, env=env, batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)

np.random.seed(0)

score_history = []

for i in range(1000):
    done = False
    score = 0
    obs, info = env.reset()
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, truncated, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state

    score_history.append(score)
    writer.add_scalar("Score", score, i)
    print(f'epsisode {i}, score: {score}, 100 game ave: {np.mean(score_history[-100:])}')

    if i % 25 == 0:
        agent.save_models()

writer.flush()


