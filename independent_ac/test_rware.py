import gym
import rware
from rware import RewardType

env = gym.make("rware-tiny-2ag-v1", reward_type=RewardType.GLOBAL)

obs = env.reset()

print(f"reset obs: \n{obs}")

obs, reward, done, info = env.step(env.action_space.sample())

print(f"Rewards: {reward}") # each agent receives the same reward so
# we can just take the first one

# we don't need a vectorised environment because is is already 
# multiagent


