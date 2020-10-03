import gym
import time
import torch

from policies.random_policy import RandomPolicy
from policies.CartPole.nnpolicy import NNPolicy
from utils.sample_episode import sample_episode, sample_torch_episode
from trainer.CartPole.REINFORCE import run_episodes_policy_gradient
from utils.rendering import render_torch_environment

# parameters
env_name = 'CartPole-v1'
#n_possible_actions = 2
#policy = RandomPolicy(n_possible_actions)
#n_samples = 10

# REINFORCE TRAINER
num_hidden = 128
num_episodes = 5000
discount_factor = 0.99
learn_rate = 0.001
policy = NNPolicy(num_hidden)



env = gym.envs.make(env_name)
env.max_episode_steps=3000,


episode_duration_policy_gradients = run_episodes_policy_gradient(
    policy, env, num_episodes, discount_factor, learn_rate, sample_torch_episode, render_torch_environment
)

