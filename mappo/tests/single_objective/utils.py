import numpy as np
from functools import reduce
import gymnasium as gym_
from minigrid.wrappers import ObservationWrapper
import operator
from functools import reduce

# todo test lstm layer with both the default image and the Flat Img Grid

class FlatImageGrid(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        imgSpace = env.observation_space.spaces['image']
        imgSize = reduce(operator.mul, imgSpace.shape, 1) # multiplies the shape together 
        # although an lstm expects our input shape to be a three dimensional vector. 
        # this is not necessarily bad through because we can use the trick from the POMDP
        # carpole tests
        self.observation_space = gym_.spaces.Box(
            low=0,
            high=255,
            shape=(imgSize + 1,),
            dtype="uint8"
        )

    def observation(self, obs):
        # flatten the image
        img_ = obs["image"].flatten()
        dir_ = obs["direction"]
        obs = np.append(img_, dir_)
        return obs