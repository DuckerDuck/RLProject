import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function and policy's sample_action function as lists.
    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.
    Returns:
        Tuple of lists (states, actions, rewards, dones). All lists should have same length. 
        Hint: Do not include the state after the termination in the list of states.
    """
    states = []
    actions = []
    rewards = []
    dones = []

    state = env.reset()
    done = False
    while not done:

        action = policy.sample_action(state)
        states.append(state)
        actions.append(action)

        state, r, done, _ = env.step(action)

        dones.append(done)
        rewards.append(r)
    return states, actions, rewards, dones


def sample_torch_episode(env, policy, device):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function as tensors.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of tensors (states, actions, rewards, dones). All tensors should have same first dimension and 
        should have dim=2. This means that vectors of length N (states, rewards, actions) should be Nx1.
        Hint: Do not include the state after termination in states.
    """
    states = []
    actions = []
    rewards = []
    dones = []

    done = False
    state = env.reset()

    while not done:
        state = torch.Tensor(state).float().to(device)
        action = policy.sample_action(state)
        new_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        state = new_state

    def make_tensor(l, from_tensor=False):
        if from_tensor:
            a = torch.cat(l).view(len(l), -1)
            t = torch.squeeze(a)
        else:
            t = torch.squeeze(torch.Tensor(l))
            t = t.to(device)
            t = torch.squeeze(t)

        if len(t.shape) < 2:
            return t[:, None]
        return t

    return make_tensor(states, True), make_tensor(actions), make_tensor(rewards), make_tensor(dones)
