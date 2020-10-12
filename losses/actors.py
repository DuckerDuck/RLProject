import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

def makePolicyLoss(Psi):

    def loss():
        def loss(policy, episode):
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
            psi = Psi(policy=policy, episode=episode)

            log_probs = torch.log(policy.get_probs(states, actions))
            loss = 0

            for t in range(len(states) - 1, -1, -1):
                loss += log_probs[t] * next(psi)

            return -loss

        return loss

    return loss

@makePolicyLoss
class REINFORCE:
    def __init__(self, policy, episode):

        rewards = episode[2]

        self._rewards = rewards
        self._gamma = policy.discount_factor

        self._t = len(rewards)
        self._G = 0

    def __next__(self):

        self._t -= 1

        if self._t < 0:
            return 0

        self._G = self._rewards[self._t] + self._gamma * self._G
        return self._G

@makePolicyLoss
class GAE:
    def __init__(self, policy, episode):

        states, _, rewards, dones = episode

        self._rewards = rewards
        self._states = states
        self._dones = dones

        self._gamma = policy.discount_factor
        self._lambda = policy.lambdapar # ? improve this perhaps
        self._V = policy.V

        self._t = len(rewards)
        self._A = 0

    def __next__(self):

        self._t -= 1

        if self._t < 0:
            return 0

        next_val = self._V(self._states[self._t+1]) if not self._dones[self._t] else 0
        prev_val = self._V(self._states[self._t])

        delta_t = self._rewards[self._t] + self._gamma*next_val - prev_val

        self._A = delta_t + self._gamma*self._lambda*self._A
        return self._A
