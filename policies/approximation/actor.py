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
        return self(obs.float()).gather(1, actions.long())

    def sample_action(self, obs):
        with torch.no_grad():
            return torch.multinomial(self(obs), 1).item()


# Non Differentiable Policies

class EGreedy(Actor):

    def __init__(self, get_q, epsilon):
        super().__init__()

        self._epsilon = epsilon
        self._get_q = get_q

    def get_probs(self, obs, actions): # Get values of each action for current observation
        return torch.squeeze(
            torch.Tensor((
                action == self.sample_action(observation)
                for action, observation in zip(actions, obs)
            )).float()
        )[:, None]

    def sample_action(self, obs):

        with torch.no_grad():
            action_vals = self._get_q(torch.Tensor(obs)).detach().numpy()

        # with probablity epsilon
        if random.random() >= self._epsilon:
            return torch.argmax(action_vals).item() # greedy action

        return torch.randint(0, len(action_vals), (1,)).item() # random action

class Boltzman(Actor):

    def __init__(self, get_q, epsilon):
        super().__init__()

        self._epsilon = epsilon
        self._get_q = get_q

    def _get_pdist(self, obs):
        return F.softmax(self._get_q(obs)/self._epsilon, dim=-1)

    def get_probs(self, obs, actions): # Get values of each action for current observation
        self_get_pdist(obs).gather(-1, actions.long())

    def sample_action(self, obs):
        with torch.no_grad():
            return torch.multinomial(self._get_pdist(obs), 1).item()
