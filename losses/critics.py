import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np

def MCValue():
    '''
    Return loss for ciritcs value function unbiased estimate
    '''

    def loss(policy, episode, discount_factor):

        # Get number of time-steps (train once for each time-step)
        states, _, rewards, _ = episode
        loss = 0
        rtrn = 0

        for t in range(len(states) - 1, -1, -1):
            loss = loss + F.smooth_l1_loss(rtrn, policy.V(states[t]))
            rtrn = discount_factor * rtrn + rewards[t]

        return loss

    return loss
