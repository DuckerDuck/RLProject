import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

def compute_reinforce_loss(policy, episode, discount_factor):
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

def run_episodes_policy_gradient(policy, env, num_episodes, discount_factor, learn_rate, sampling_function, render = None):
    optimizer = optim.Adam(policy.parameters(), learn_rate)
    
    episode_durations = []
    for i in range(num_episodes):
        

        optimizer.zero_grad()
        
        episode = sampling_function(env, policy)
        loss = compute_reinforce_loss(policy, episode, discount_factor)
        
        loss.backward()
        optimizer.step()
                           
        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))
            if render is not None and i%200 == 0:
                render(env, policy)
        episode_durations.append(len(episode[0]))
        
    return episode_durations