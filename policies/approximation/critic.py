import torch
import torch.nn as nn
import torch.nn.functional as F

import policies.approximation.actor

from policies.agent import ApproximatingAgent
from utils.settings import build_cls

class Critic(ApproximatingAgent):

    def __init__(self, policy, *args, **kwargs):

        super().__init__(
            *args,
            out_activation = None,
            **kwargs,
        )

        self._policy = policy

    def sample_action(self, obs):
        return self.policy.sample_action(obs)

    @property
    def policy(self):
        return self._policy

class QCritic(Critic):

    def __init__(self, policy, *args, **kwargs):

        super().__init__(
            build_cls(policies.approximation.actor, **policy).set_q(lambda x : self(x)),
            *args,
            **kwargs,
        )

