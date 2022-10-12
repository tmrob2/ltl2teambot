from math import e
from os import fpathconf
from typing import Callable, Dict, List, Optional, Tuple, Union, Type

import gym
import torch
from torch import nn

from ltl_env import LTLBootcamp

from stable_baselines3 import ppo
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from gat import GA

DEVICE = "cuda"


class ACNetwork(nn.Module):
    """
    Network for policy and value function. 
    Receives the features extracted by the feature extractor

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi = 64,
        last_layer_dim_vf = 64
    ) -> None:
        super(ACNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # LTL module
        self.ltl_embedder = nn.Embedding(11, 16, padding_idx=0)
        self.ga = GA(seq_len=10, num_rel=8, num_heads=32, num_layers=3, device=DEVICE)

        # Policy network
        self.policy_net = nn.Linear(160, last_layer_dim_pi)
        # Value network
        self.value_net = nn.Linear(160, last_layer_dim_vf)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        batch_size = features.shape[0]
        formula = features[:, :10].to(torch.long)
        rels = features[:, 10:].reshape(batch_size, 10, 10).to(torch.long)
        
        # LTL module
        embedded_formula = self.ltl_embedder(formula)
        embedded_formula = self.ga(embedded_formula, rels) # [B,10,16]
        embedded_formula = embedded_formula.reshape(batch_size, -1) # [B,160]

        # RL module
        return self.policy_net(embedded_formula), self.value_net(embedded_formula)


class CustomActorCriticPolicy(ActorCriticPolicy): # for stable baselines3
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ACNetwork(self.features_dim)