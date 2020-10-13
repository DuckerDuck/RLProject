import argparse
import json
import os

import numpy as np

SEED = 42


def average_result_same_setting(path):
    files = [f for f in os.listdir(path) if '.json' in f]
    filepath = path + "/{}"

    ep_lengths = []
    for file in files:
        with open(filepath.format(file)) as f:
            data = json.load(f)
            ep_lengths.append(data['episode_length'])
    return np.mean(ep_lengths, axis=0), np.std(ep_lengths, axis=0)


def main(config):
    mean, std = average_result_same_setting(config.path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, required=True)
    config = parser.parse_args()
    main(config)
