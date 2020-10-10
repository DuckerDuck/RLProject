import time

import gym
import torch
from torch.optim import Adam

import policies
import policies.approximation
import policies.approximation.actor
import policies.approximation.critic
import policies.random_policy

import losses
import losses.REINFORCE
import losses.critics
import losses.noloss

from utils.sample_episode import sample_episode, sample_torch_episode
from utils.rendering import render_torch_environment
from utils.settings import SettingsParser, DictArgs, get_mod_attr, build_cls
from utils.evaluation import ResultsManager

def main(args):

    # Create env
    env = gym.envs.make(**args.env)

    # Create agent
    policy = build_cls(policies, **args.policy)

    # Create losses
    loss_fn = build_cls(losses, **args.loss)

    # Create Result Writer
    writer = ResultsManager.setup_writer('results/output.json', args)

    # Training and Evalutation
    optimizer = Adam(policy.parameters(), args.policy['learn_rate']) if next(policy.parameters(), None) is not None else None

    episode_durations = []
    for i in range(args.num_episodes):

        # Reset gradients
        if optimizer is not None:
            optimizer.zero_grad()

        # Run episode with current policy
        episode = sample_torch_episode(env, policy)

        # Write to results
        writer.add_value('episode', i)
        writer.add_value('episode_length', len(episode[0]))

        # Compute loss
        loss = loss_fn(policy, episode, args.policy['discount_factor'])

        # Update parameters
        if optimizer is not None:
            loss.backward()
            optimizer.step()

        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))
            if args.render and i%200 == 0:
                render_torch_environment(env, policy)
        episode_durations.append(len(episode[0]))

    # Save Results
    writer.save()

if __name__ == '__main__':

    parser = SettingsParser()

    parser.add_argument(
        '--policy',
        action = DictArgs,
        nargs = '+',
        help = 'settings for policy',
    )

    parser.add_argument(
        '--loss',
        help = 'loss function to use',
    )

    parser.add_argument(
        '--env',
        action = DictArgs,
        nargs = '+',
        help = 'settings for environment',
    )

    parser.add_argument(
        '--num_episodes',
        type = int,
        default = 100,
        help = 'number of episodes to run the training for',
    )

    parser.add_argument(
        '--render',
        default = False,
        action = 'store_true',
        help = "If this flag is given, rendering in evaluation is performed",
    )

    parser.add_argument(
        '--device',
        default = 'cuda:0' if torch.cuda.is_available() else 'cpu',
        help = 'device (cuda or cpu) in which to run model',
    )

    args = parser.parse_args()
    args.device = torch.device(args.device)

    main(args)

