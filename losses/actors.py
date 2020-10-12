import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

def makePolicyLoss(Psi):

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
            states, actions, _, _ = episode
            psi = Psi(policy=policy, episode=episode, discount_factor=discount_factor)

            log_probs = torch.log(policy.get_probs(states, actions))
            loss = 0

            for t in range(len(states) - 1, -1, -1):
                loss += log_probs[t] * next(psi)

            return -loss

        return loss

    return loss

@makePolicyLoss
class REINFORCE:
    def __init__(self, policy, episode, discount_factor):

        rewards = episode[2]

        self._rewards = rewards
        self._gamma = discount_factor

        self._t = len(rewards)
        self._G = 0

    def __next__(self):

        self._t -= 1

        if self._t < 0:
            return 0

        self._G = self._rewards[self._t] + self._gamma * self._G
        return self._G
