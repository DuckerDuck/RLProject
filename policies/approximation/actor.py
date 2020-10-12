from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from policies.agent import Agent, ApproximatingAgent

# Actor interface

class Actor(ABC, Agent):

    @abstractmethod
    def get_probs(self, obs, actions):
        """
        This function takes a tensor of states and a tensor of actions and returns a tensor that contains 
        a probability of perfoming corresponding action in all states (one for every state action pair). 

        Args:
            obs: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: batch_size x 1

        Returns:
            A torch tensor filled with probabilities. Shape: batch_size x 1.
        """
        pass

# Differentiable Policy

class DifferentiableActor(Actor, ApproximatingAgent):

    def __init__(self, *args, **kwargs):
        ApproximatingAgent.__init__(
            self,
            *args,
            out_activation = 'Softmax',
            **kwargs,
        )

    def get_probs(self, obs, actions):
        return self.get_actor_probs(obs.float()).gather(1, actions.long())

    def sample_action(self, obs):
        with torch.no_grad():
            return torch.multinomial(self.get_actor_probs(obs), 1).item()

    def get_actor_probs(self, obs):
        return self(obs)