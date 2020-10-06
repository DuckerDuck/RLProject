import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np

def compute_q_vals(Q, states, actions, n, batch_size):
    """
    This method returns Q values for given state action pairs.

    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    return torch.gather(
        Q(
            states.reshape(batch_size, n, -1)[:, 0, :] # Get only initial states of sequences
        ),
        1,
        actions.reshape(batch_size, n, -1)[:, 0, :].long(), # Get only action performed in initial state
    )

def compute_targets(Q, rewards, states, dones, discount_factor, n, batch_size):
    """
    This method returns targets (values towards which Q-values should move).

    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """

    # Convert dones (in final states) to mask (to use to set terminal states' values to 0)
    mask = 1 - dones.reshape(batch_size, n, -1)[:, -1, :]*1

    # Get values for each action and each next state
    next_q_vals = Q(
        states.reshape(batch_size, n, -1)[:, -1, :]
    )

    # Get values for each next state according to off-policy Q-Learning rule (max per next state)
    next_vals = torch.max(next_q_vals, dim=1)[0][:, None] * mask

    # Compute MC estimate of return up to last state
    g_partial = (
        rewards.reshape(batch_size, n, -1) * # divide rewards into sequences of n consecutive transitions
        torch.pow(discount_factor, torch.arange(n))
    ).sum(1)

    # Return approximated targets
    return g_partial  + (discount_factor**n)*next_vals

def _sample_transitions(states, actions, rewards, dones, n=2, batch_size=1):
    '''
    Args:
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        n: an int, the length of the consecutive transitions to get
        batch_size: the number of transition sequences to sample
    Returns:
        for each tensor (states, actions, rewards, and dones) returns a (batch_size x n) x shape[-1] tensor of sampled random consectuive sequences of length n
    '''

    # Select batch_size random sequences of n consecutive indices along the first axis of states
    i = torch.arange(n).repeat(batch_size) + torch.repeat_interleave(torch.randint(0, states.shape[0]-n+1, (batch_size,)), n)

    # Return subselected
    return torch.index_select(states, 0, i), torch.index_select(actions, 0, i), torch.index_select(rewards, 0, i), torch.index_select(dones, 0, i)

def n_step_loss(n=2, batch_size=1):

    def loss(policy, episode, discount_factor):

        # Get number of time-steps (train once for each time-step)
        n_steps = episode[-1].shape[0]

        tot_loss = 0
        for _ in range(n_steps):

            # Sample n-steps transitions
            states, actions, rewards, dones = _sample_transitions(*episode, n=n, batch_size=batch_size)

            # Get error between prediction and target bootstrapping
            tot_loss = tot_loss + F.smooth_l1_loss(
                compute_q_vals(policy, states, actions, n, batch_size),
                compute_targets(policy, rewards, states, dones, discount_factor, n, batch_size),
            )

        return tot_loss

    return loss
