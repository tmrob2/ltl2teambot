import torch
import numpy as np

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []  
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.ahx = []
        self.acx = []
        self.chx = []
        self.ccx = []

        self.batch_size = batch_size

    # TODO the great thing about creating batches is that it can easily be
    # applied to the multiagent setting
    def generate_batches(self):
        n_states = len(self.states)
        
        # We want to create the starting index for each of the batches
        batch_start = np.arange(0, n_states, self.batch_size)

        # we want to create and index for each of the states in current memory 
        indices = np.arange(n_states, dtype=np.int64)
        
        # Now randomly shuffle the indices so that we get random sampling from a 
        # batch
        np.random.shuffle(indices)

        # starting at the batch start index from above, select the next batch_size
        # random indices to go into a batch
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states),\
               np.array(self.actions),\
               np.array(self.probs),\
               np.array(self.vals),\
               np.array(self.rewards),\
               np.array(self.dones),\
               np.array(self.ahx), \
               np.array(self.acx), \
               np.array(self.chx), \
               np.array(self.ccx), \
               batches

    def store_memory(self, state, action, probs, vals, reward, done, actor_rcell, critic_rcell):
        (ah, ac) = actor_rcell
        (ch, cc) = critic_rcell

        (ah_, ac_) = (ah.cpu().numpy(), ac.cpu().numpy())
        (ch_, cc_) = (ch.cpu().numpy(), cc.cpu().numpy())
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        self.ahx.append(ah_)
        self.acx.append(ac_)
        self.chx.append(ch_)
        self.ccx.append(cc_)

    # TODO we can upgrade this to creating a tensor of zeroes
    # except in the instance of states where we just want Nones
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []    
        self.ahx = []
        self.acx = []
        self.chx = []
        self.ccx = []
        