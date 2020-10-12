from abc import ABC, abstractmethod

from utils.settings import build_cls

import policies
from policies.agent import Agent
from policies.approximation.critic import Critic
from policies.approximation.actor import Actor

from itertools import chain

class ActorCritic(Actor, Critic):

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

    def V(self, obs):
        return self._critic.V(obs)

    def get_probs(self, obs, action):
        return self._actor.get_probs(obs, action)

    def parameters(self):
        return chain(self._actor.parameters(), self._critic.parameters())

def makeFactory(cls):

    def build(actor, critic, **kwargs):
        return cls(
            actor = build_cls(policies, **actor),
            critic = build_cls(policies, **critic),
            **kwargs,
        )

    return build

class GAEAC(ActorCritic):

    def __init__(self, actor, critic, discount_factor, lambdapar):
        super().__init__(actor, critic, discount_factor)
        self._lambdapar = lambdapar

    @property
    def lambdapar(self):
        return self._lambdapar

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        return self

makeGAEAC = makeFactory(GAEAC)
