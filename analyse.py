import argparse
import json
import os
from glob import glob

import numpy as np

SEED = 42

def aggregate(input_name, **kwargs):

    files = glob(input_name)

    ep_lengths = []
    for file in files:
        with open(file) as f:
            data = json.load(f)
            ep_lengths.append(data['episode_length'])

    return dict(
        data,
        episode_length = list(np.mean(ep_lengths, axis=0)),
        episode_std = list(np.std(ep_lengths, axis=0)),
    )

def main(config):

    with open(config.output_name, 'wt') as f:
        json.dump(
            globals()[config.function](**config.__dict__),
            f
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_name',
        type=str,
        required=True,
        help = 'regexp of file(s) to analyze'
    )

    parser.add_argument(
        '--output_name',
        type=str,
        required=True,
        help = "Name of file in which to dump results"
    )

    parser.add_argument(
        "--function",
        type=str,
        choices = ["aggregate"],
        required=True,
        help = "Function to be performed on data given"
    )

    config = parser.parse_args()
    main(config)
