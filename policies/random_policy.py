import numpy as np

# pure random policy implementation

class RandomPolicy(object):

    def __init__(self, n_possible_actions):
        self.n_possible_actions = n_possible_actions

    def sample_action(self, obs):
        return np.random.randint(self.n_possible_actions)