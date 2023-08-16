from teamgrid.minigrid import *
from teamgrid.register import register

class Empty(MiniGridEnv):
    """
    Classical 4 rooms gridworld environment.
    """

    def __init__(self, num_agents=2, num_goals=2):
        self.num_agents = num_agents
        self.num_goals = num_goals
        size = 6
        super().__init__(
            grid_size=size,
            max_steps=8 * size**2
        )

        # Only allow turn left/turn right/forward movement actions
        self.action_space = spaces.Discrete(self.actions.forward+1)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Place the goal objects randomly
        self.goals = []
        for i in range(self.num_goals):
            obj = Ball('green')
            self.place_obj(obj)
            self.goals.append(obj)

        # Randomize the player start positions and orientations
        for _ in range(self.num_agents):
            self.place_agent()

register(
    id='TEAMGrid-Empty-v0',
    entry_point='teamgrid.envs:Empty'
)