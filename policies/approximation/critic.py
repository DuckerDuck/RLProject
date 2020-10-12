import torch
import torch.nn as nn
import torch.nn.functional as F

from policies.agent import Net

class Critic(Net):

    def __init__(self, policy, *args, **kwargs):

        super().__init__(
            *args,
            out_activation = None,
            n_outputs = 1,
            **kwargs,
        )

class VCritic(Critic):

    def __init__(self, *args, **kwargs):

        super().__init__(
            *args,
            **kwargs,
        )

    def V(self, obs):
        return self(obs)

