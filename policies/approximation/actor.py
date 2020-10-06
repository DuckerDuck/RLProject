import torch
import torch.nn as nn
import torch.nn.functional as F

class NNPolicy(nn.Module):

    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Args:
            x: input tensor (first dimension is a batch dimension)

        Return:
            Probabilities of performing all actions in given input states x. Shape: batch_size x action_space_size
        """
        # YOUR CODE HERE
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)

        return F.softmax(x, dim=-1)


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
        # YOUR CODE HERE

        probs = self.forward(obs.float())
        action_probs = probs.gather(1, actions.long())

        return action_probs

    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: state as a tensor. Shape: 1 x obs_dim or obs_dim

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        with torch.no_grad():
            x = self.forward(obs)
            action = torch.multinomial(x, 1).item()
        return action
