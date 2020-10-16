import os
import time

import gym
import torch
import numpy as np
from datetime import datetime
from torch.optim import Adam

import policies
import policies.approximation
import policies.approximation.actor
import policies.approximation.critic
import policies.actorcritic
import policies.random_policy

import losses
import losses.actors
import losses.critics
import losses.noloss

from utils.sample_episode import sample_episode, sample_torch_episode
from utils.rendering import render_torch_environment
from utils.settings import SettingsParser, DictArgs, get_mod_attr, build_cls
from utils.evaluation import ResultsManager

import pickle

SEED = 42

def main(args):
    timestamp = datetime.now().strftime('%m_%d_%H_%M_%S')
    directory = "results/exp_{}_{}".format(args.name, timestamp)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(args.num_runs):
        run(args, SEED + i, directory)


def run(args, seed, directory):
    # Create env
    env = gym.envs.make(**args.env)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create agent
    policy = build_cls(policies, **args.policy).to(args.device)

    # Create losses
    loss_fns = [build_cls(losses, **loss_desc) for loss_desc in (args.loss if isinstance(args.loss, list) else [args.loss])]

    # Create Result Writer
    writer = ResultsManager.setup_writer(f'{directory}/output_{seed}.json', args)

    # Training and Evalutation
    optimizer = Adam(policy.parameters(), args.policy['learn_rate']) if next(policy.parameters(), None) is not None else None

    episode_durations = []
    for i in range(args.num_episodes):

        # Reset gradients
        if optimizer is not None:
            optimizer.zero_grad()

        # Run episode with current policy
        episode = sample_torch_episode(env, policy, args.device)

        # Write to results
        writer.add_value('episode', i)
        writer.add_value('episode_length', len(episode[0]))
        writer.add_value('return', episode[2].sum().item())

        # Compute loss
        loss = [loss_fn(policy, episode) for loss_fn in loss_fns]

        # Update parameters
        if optimizer is not None:
            for l in loss:
                l.backward()
            optimizer.step()

        if i % 10 == 0:
            print(f"Episode {i} finished after {len(episode[0])} steps")
            if args.render and i % 200 == 0:
                render_torch_environment(env, policy)
        episode_durations.append(len(episode[0]))

    # Save Results
    writer.save()
    with open(f"{directory}/model_{seed}.pt", "wb") as f:
        pickle.dump(policy, f)

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

    parser.add_argument(
        '--num_runs',
        type = int,
        default = 10,
        help = 'number of runs of the same experiments, but with different seeds',
    )

    parser.add_argument(
        '--name',
        type = str,
        default = "",
        help = 'number of runs of the same experiments, but with different seeds',
    )

    args = parser.parse_args()
    args.device = torch.device(args.device)
    print('Running on device', args.device)
    main(args)
