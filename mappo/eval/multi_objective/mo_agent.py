import numpy as np
import torch
from mappo.networks.moltlnet import AC_MO_LTL_Model

class Agent:

    def __init__(
        self,
        action_space, 
        num_agents, 
        num_tasks,
        device,
        argmax=False, 
        num_envs=1,
        use_memory=False,
        preprocess_obs=None
        ) -> None:
        
        self.preprocess_obs = preprocess_obs if preprocess_obs is not None else self.default_preprocess_obss
        self.model = AC_MO_LTL_Model(action_space, num_tasks, use_memory=use_memory)
        self.model.load_models()
        
        self.device = device
        self.argmax = argmax
        self.num_envs = num_envs

        if self.model.use_memory:
            self.memories = torch.zeros(num_agents * self.num_envs, self.model.memory_size, device=device)

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

    def default_preprocess_obss(self, obss, device=None):
        return torch.tensor(np.array(obss), device=device, dtype=torch.float)

    

    