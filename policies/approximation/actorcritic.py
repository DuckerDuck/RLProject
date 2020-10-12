from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from policies.agent import Agent

class ActorCritic(ABC, Agent):

    def __init__(self, actor, critic, discount_factor):

        Agent.__init__(self, discount_factor)

        self._actor = actor
        self._critic = critic

    def sample_action(self, obs):
        return self._actor.sample_action(obs)

    @property
    def actor(self):
        return self._actor

    @property
    def critic(self):
        return self._critic


class GAEAC(ActorCritic):

    def __init__(self, actor, critic, discount_factor, lambdapar):
        super().__init__(actor, critic, discount_factor)
        self._lambdapar = lambdapar

    @property
    def lambdapar(self):
        return self._lambdapar
