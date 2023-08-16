from audioop import avg
from collections import deque
import gymnasium as gym_
import numpy as np
from mappo.agent import Agent
from mappo.utils import FlatImageGrid
from torch.utils.tensorboard import SummaryWriter
import torch

if __name__ == "__main__":
    
    # make a simple minigrid environment
    env = gym_.make('MiniGrid-Empty-8x8-v0')
    env = FlatImageGrid(env)

    N = 128
    MODEL_SAVE = 500
    batch_size = 32
    n_epochs = 4
    alpha = 0.001
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
        alpha=alpha, beta=alpha, n_epochs=n_epochs, input_dim=env.observation_space.shape[0]
    )

    n_games = 300

    best_score = env.reward_range[0]
    score_history = deque(maxlen=100)

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    ahxs = torch.zeros(1, 1, 256, dtype=torch.float32).to(agent.actor.device)
    acxs = torch.zeros(1, 1, 256, dtype=torch.float32).to(agent.actor.device)

    chxs = torch.zeros(1, 1, 256, dtype=torch.float32).to(agent.actor.device)
    ccxs = torch.zeros(1, 1, 256, dtype=torch.float32).to(agent.actor.device)

    actor_rcell = (ahxs, acxs)
    critic_rcell = (chxs, ccxs)

    writer = SummaryWriter()

    for i in range(n_games):
        observation, info = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val, actor_rcell, critic_rcell = \
                agent.choose_action(observation, actor_rcell, critic_rcell)
            # TODO in a multiagent env the action will be a vector, 
            #  and its dimensions will need to be checked
            observation_, reward, done, _, info = env.step(action)
            #env.render()
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done, actor_rcell, critic_rcell)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
                

            observation = observation_
            
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        score_history.append(score)
        avg_score = np.mean(score_history)

        print('episode', i, 'score %.1f' % score, 'avg score %.3f' % avg_score,
        'time_steps', n_steps, 'learning_steps', learn_iters)

        writer.add_scalar("avg score", avg_score, n_steps)
        if best_score > 10:
            break


        writer.flush()