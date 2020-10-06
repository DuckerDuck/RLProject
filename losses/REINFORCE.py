import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

def loss():
    def loss(policy, episode, discount_factor):
        """
        Computes reinforce loss for given episode.
        Args:
            policy: A policy which allows us to get probabilities of actions in states with its get_probs method.
        Returns:
            loss: reinforce loss
        """
        # Compute the reinforce loss
        # Make sure that your function runs in LINEAR TIME
        # Note that the rewards/returns should be maximized 
        # while the loss should be minimized so you need a - somewhere
        # YOUR CODE HERE

        # −∑𝑡log𝜋𝜃(𝑎𝑡|𝑠𝑡)𝐺𝑡
        states, actions, rewards, _ = episode
        log_probs = torch.log(policy.get_probs(states,actions))
        rtrn = 0
        loss = 0

        for t in range(len(states) - 1, -1, -1):
            loss += log_probs[t] * (rewards[t] + discount_factor * rtrn)
            rtrn = discount_factor * rtrn + rewards[t]
        loss *= -1
        return loss

    return loss
