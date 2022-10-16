import numpy
import torch
import torch.nn.functional as F
from mappo.algorithms.dec_ma_mo_base import BaseAlgorithm

class PPO(BaseAlgorithm):
    """The Proximal Policy Optimization algorithm"""

    def __init__(
        self, 
        envs, 
        acmodels, 
        num_agents,
        num_objectives,
        device=None, 
        num_frames_per_proc=None, 
        discount=0.99, 
        lr=0.001, 
        gae_lambda=0.95,
        entropy_coef=0.001, 
        value_loss_coef=0.5, 
        max_grad_norm=0.5, 
        recurrence=1,
        adam_eps=1e-8, 
        clip_eps=0.1, 
        epochs=4, 
        batch_size=256, 
        preprocess_obss=None,
        reshape_reward=None,
        mu=None,
        seed=None
        ):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodels, device, num_agents, num_objectives,
            num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
            value_loss_coef, max_grad_norm, recurrence, preprocess_obss, mu, seed)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        self.optimizers = {
            i: torch.optim.Adam(self.acmodels[i].parameters(), lr, eps=adam_eps)
            for i in range(self.num_agents)
        }
        self.batch_num = 0

    def update_parameters(self, exps):
        # Collect experiences

        # for each agent we have to update its model
        # therefore we will probably require some agent loops at this point

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for agent in range(self.num_agents):
            
                for inds in self._get_batches_starting_indexes():
                    # Initialize batch values

                    batch_entropy = 0
                    batch_value = 0
                    batch_policy_loss = 0
                    batch_value_loss = 0
                    batch_loss = 0

                    # Initialize memory

                    if self.acmodels[0].use_memory:
                        memory = exps.memory[agent][inds]

                    for i in range(self.recurrence):
                        # Create a sub-batch of experience

                        sb = exps[agent][inds + i]

                        # Compute loss

                        if self.acmodels[0].use_memory:
                            dist, value, memory = \
                                self.acmodels[agent](sb.obs, memory * sb.mask)
                        else:
                            dist, value = self.acmodel(sb.obs)

                        entropy = dist.entropy().mean()

                        ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                        surr1 = ratio * sb.advantage
                        surr2 = torch.clamp(
                            ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                        ) * sb.advantage
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # TODO the value from the model is it agent x (tasks + 1)
                        value_clipped = sb.value + torch.clamp(
                            value - sb.value, -self.clip_eps, self.clip_eps
                        )
                        surr1 = (value - sb.returnn).pow(2)
                        surr2 = (value_clipped - sb.returnn).pow(2)
                        value_loss = torch.max(surr1, surr2).mean()
                        #surr1 = F.huber_loss(value, sb.returnn)
                        #surr2 = F.huber_loss(value_clipped, sb.returnn)
                        #value_loss = torch.max(surr1, surr2)

                        loss = policy_loss - \
                            self.entropy_coef * entropy + \
                            self.value_loss_coef * value_loss

                        # Update batch values

                        batch_entropy += entropy.item()
                        batch_value += value.mean().item()
                        batch_policy_loss += policy_loss.item()
                        batch_value_loss += value_loss.item()
                        batch_loss += loss

                        # Update memories for next epoch

                        if self.acmodels[agent].use_memory and i < self.recurrence - 1:
                            exps.memory[agent][inds + i + 1] = memory.detach()

                    # Update batch values

                    batch_entropy /= self.recurrence
                    batch_value /= self.recurrence
                    batch_policy_loss /= self.recurrence
                    batch_value_loss /= self.recurrence
                    batch_loss /= self.recurrence

                    # Update actor-critic

                    self.optimizers[agent].zero_grad()
                    batch_loss.backward()
                    grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodels[agent].parameters()) ** 0.5
                    torch.nn.utils.clip_grad_norm_(self.acmodels[agent].parameters(), self.max_grad_norm)
                    self.optimizers[agent].step()

                    # Update log values

                    log_entropies.append(batch_entropy)
                    log_values.append(batch_value)
                    log_policy_losses.append(batch_policy_loss)
                    log_value_losses.append(batch_value_loss)
                    log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms)
        }

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.
        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.
        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
