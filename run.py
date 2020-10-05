import time

import gym
import torch
from torch.optim import Adam

import policies
import policies.approximation
import policies.random_policy

from losses.REINFORCE import loss as loss_fn

from utils.sample_episode import sample_episode, sample_torch_episode
from utils.rendering import render_torch_environment
from utils.settings import SettingsParser, DictArgs, get_mod_attr

# REINFORCE TRAINER
# num_hidden = 128
# num_episodes = 5000
# discount_factor = 0.99
# learn_rate = 0.001
# policy = NNPolicy(num_hidden)

def main(args):

    # Create env
    env = gym.envs.make(**args.env)

    # Create agent
    policy = get_mod_attr(policies, args.policy['cls'])(
        **args.policy['parameters']
    )

    # Training and Evalutation
    optimizer = Adam(policy.parameters(), args.policy['learn_rate'])

    episode_durations = []
    for i in range(args.num_episodes):

        # Reset gradients
        optimizer.zero_grad()

        # Run episode with current policy
        episode = sample_torch_episode(env, policy)

        # Compute loss
        loss = loss_fn(policy, episode, args.policy['discount_factor'])

        # Update parameters
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))
            if args.render and i%200 == 0:
                render_torch_environment(env, policy)
        episode_durations.append(len(episode[0]))

if __name__ == '__main__':

    parser = SettingsParser()

    parser.add_argument(
        '--policy',
        action = DictArgs,
        nargs = '+',
        help = 'settings for policy',
    )

    parser.add_argument(
        '--num_episodes',
        type = int,
        default = 100,
        help = 'number of episodes to run the traininr for',
    )

    parser.add_argument(
        '--env',
        action = DictArgs,
        nargs = '+',
        help = 'settings for environment',
    )

    parser.add_argument(
        '--render',
        default = False,
        action = 'store_true',
        help = "If this fleg is given, rendering in evaluation is performed",
    )

    parser.add_argument(
        '--device',
        default = 'cuda:0' if torch.cuda.is_available() else 'cpu',
        help = 'device (cuda or cpu) in which to run model',
    )

    args = parser.parse_args()
    args.device = torch.device(args.device)

    main(args)

