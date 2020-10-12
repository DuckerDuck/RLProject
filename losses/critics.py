import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np

def MCValue():
    '''
    Return loss for ciritcs value function unbiased estimate
    '''

    def loss(policy, episode):

        # Get number of time-steps (train once for each time-step)
        states, _, rewards, _ = episode
        rtrn = 0
        loss = 0

        for t in range(len(states) - 1, -1, -1):
            rtrn = policy.discount_factor * rtrn + rewards[t]
            loss = loss + F.smooth_l1_loss(rtrn, policy.V(states[t]))

        return loss

    return loss
