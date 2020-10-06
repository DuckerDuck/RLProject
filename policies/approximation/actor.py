import torch
import torch.nn as nn
import torch.nn.functional as F

from policies.agent import ApproximatingAgent

class NNPolicy(ApproximatingAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            out_activation = 'Softmax',
            **kwargs,
        )

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

        return self(obs.float()).gather(1, actions.long())

    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: state as a tensor. Shape: 1 x obs_dim or obs_dim

        Returns:
            An action (int).
        """

        with torch.no_grad():
            return torch.multinomial(self(obs), 1).item()
