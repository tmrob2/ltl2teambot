from functools import reduce
import gym
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