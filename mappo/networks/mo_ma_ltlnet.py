import os
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from mappo.minigrid_copy.utils.base_model import RecurrentACModel
from copy import deepcopy


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class AC_MA_MO_LTL_Model(nn.Module, RecurrentACModel):
    def __init__(
        self, 
        #obs_space, 
        action_space, 
        num_tasks,
        partial_obs_dims = (7, 7),
        use_memory=False, 
        use_text=False,
        chkpt_file="tmp/ppo"
        ):
        super().__init__()

        self.checkpoint_file = \
            "/home/thomas/ai_projects/MAS_MT_RL/mappo/tmp/ppo/mo_actor_torch_ppo"
        #os.path.join(chkpt_file, 'mo_actor_torch_ppo')

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        #n = obs_space["image"][0]
        #m = obs_space["image"][1]
        self.n, self.m = partial_obs_dims
        self.image_embedding_size = ((self.n-1)//2-2)*((self.m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)
        else: 
            self.recurrent = False
        # Resize image embedding
        
        #TODO self.embedding_size = self.semi_memory_size

        self.ltl_embedder = nn.Embedding(20, 8, padding_idx=0)
        self.ltl_rnn = nn.GRU(8, 32, num_layers=2, bidirectional=True, batch_first=True)

        self.embedding_size = self.semi_memory_size + 32 * 2

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1 + num_tasks)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory=None):

        img = obs[:, :147]
        ltl = obs[: , 147:]

        #x = img.transpose(1, 3).transpose(2, 3)
        x = img.reshape(obs.shape[0], self.n, self.m, 3).permute(0, 3, 1, 2)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        embedded_formula = self.ltl_embedder(ltl.to(torch.long))
        _, h = self.ltl_rnn(embedded_formula)
        embedded_formula = h[:-2, :, :].transpose(0, 1).reshape(obs.shape[0], -1)

        composed_x = torch.cat([embedding, embedded_formula], dim=1)

        x = self.actor(composed_x)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(composed_x)
        value = x.squeeze(1)

        if self.use_memory:
            return dist, value, memory
        else:
            return dist, value


    def save_models(self):
        print('... saving models ...')
        torch.save(deepcopy(self.state_dict()), self.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.load_state_dict(torch.load(self.checkpoint_file))