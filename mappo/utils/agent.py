import numpy as np
import torch
from mappo.minigrid_copy.network import ACModel

class Agent:

    def __init__(
        self,
        obs_space, 
        action_space, 
        device,
        argmax=False, 
        num_envs=1,
        use_memory=False,
        preprocess_obs=None
        ) -> None:
        
        self.preprocess_obs = preprocess_obs
        self.model = ACModel(obs_space, action_space, use_memory=use_memory)
        self.model.load_models()
        
        self.device = device
        self.argmax = argmax
        self.num_envs = num_envs

        if self.model.use_memory:
            self.memories = torch.zeros(self.num_envs, self.model.memory_size, device=device)

        #self.model.load_state_dict(get_model_state(file, device))
        self.model.to(device)
        self.model.eval()


    def get_actions(self, obs):

        # process the observation
        
        obs = self.preprocess_obs(obs, device=self.device)

        with torch.no_grad():
            if self.model.use_memory:
                dist, _, self.memories = self.model(obs, self.memories)
            else: 
                dist, _ = self.model(obs)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()
        
        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    

    