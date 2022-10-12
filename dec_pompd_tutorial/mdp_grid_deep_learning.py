# use gym minigrid and get image observation layers
from tkinter import Image
import gym
import minigrid
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
import random
import numpy as np
#from array2gif import write_gif
#import moviepy.editor as mpy

env = gym.make("MiniGrid-ObstructedMaze-1Dl-v0")
env = RGBImgObsWrapper(env)
env = ImgObsWrapper(env)

obs, _ = env.reset()


frames = []
random.seed(110)
np.random.seed(110)

for _ in range(10):
    obs, reward, done, info, _ = env.step(env.action_space.sample())
    frames.append(obs)

from moviepy.editor import ImageSequenceClip, VideoClip

def make_frame(t):
    return frames[t]

clip = ImageSequenceClip(frames, fps=5).resize(4.0)
clip.write_gif('test_animate.gif', fps=5)


