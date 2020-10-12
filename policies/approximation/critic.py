from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from policies.agent import Net

class Critic(ABC):

    @abstractmethod
    def V(self, obs):
        pass

class VCritic(Critic, Net):

    def __init__(self, *args, **kwargs):

        Net.__init__(
            self,
            *args,
            out_activation = None,
            n_outputs = 1,
            **kwargs,
        )

    def V(self, obs):
        return self(obs)
