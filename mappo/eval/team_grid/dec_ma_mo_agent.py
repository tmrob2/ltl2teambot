import numpy as np
import torch
from mappo.networks.mo_ma_ltlnet import AC_MA_MO_LTL_Model

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
        self.models = []
        for i in range(num_agents):
            self.models.append(AC_MA_MO_LTL_Model(action_space, num_tasks, use_memory=use_memory, name=f"agent{i}"))
        for i in range(num_agents):
            self.models[i].load_models()
        
        self.device = device
        self.argmax = argmax
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.memories = {}

        for agent in range(num_agents):
            if self.models[0].use_memory:
                self.memories[agent] = torch.zeros(self.num_envs, self.models[agent].memory_size, device=device)

        #self.model.load_state_dict(get_model_state(file, device))
        for model in self.models:
            model.to(device)
            model.eval()


    def get_actions(self, obs):

        # process the observation
        
        obs = self.preprocess_obs(obs, device=self.device)

        with torch.no_grad():
            for agent in range(self.num_agents):
                if self.models[agent].use_memory:
                    dist, _, self.memories[agent] = self.models[agent](obs[agent], self.memories)
                else: 
                    dist, _ = self.model(obs)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample().reshape(self.num_envs, self.num_agents)
        
        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def default_preprocess_obss(self, obss, device=None):
        #return torch.tensor(np.array(obss), device=device, dtype=torch.float)
        t = torch.tensor(np.array(obss), device=device, dtype=torch.float)
        t = t.reshape(-1, *t.shape[2:])
        return t

    

    