import numpy as np

from policies.agent import Agent
from torch import nn

# pure random policy implementation

class RandomPolicy(Agent, nn.Module):

    def __init__(self, n_actions):
        nn.Module.__init__(self)
        self.n_actions = n_actions

    def sample_action(self, obs):
        return np.random.randint(self.n_actions)

    def forward(self, x):
        return 0
