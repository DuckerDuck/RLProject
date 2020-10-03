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


def sample_torch_episode(env, policy):
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
    
    # YOUR CODE HERE
    done = False
    state = env.reset()
    action = policy.sample_action(torch.tensor(state).float())
    states.append(state)

    while not done:
        state, R, done, _ = env.step(action)
        if not done:
            states.append(state)
        actions.append(action)
        rewards.append(R)
        dones.append(done)

        action = policy.sample_action(torch.tensor(state).float())

    return torch.tensor(states), torch.tensor(actions).reshape(-1, 1), torch.tensor(rewards).reshape(-1, 1), torch.tensor(dones).reshape(-1, 1)