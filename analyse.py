import argparse
import json
import os
from glob import glob
from sklearn import metrics

import numpy as np

SEED = 42

def auc(input_name, targets, lower_bnd, **kwargs):
    """Area under Curve + Asymptotic
        Assumes that results are already aggregated!
    """
    files = glob(input_name + '*')

    results = {}
    for file in files:
        with open(file) as f:
            data = json.load(f)
            for target in targets:
                y = np.array(data[target])

                # Asymptotic
                upper_bnd = y[-1]

                # Normalize returns
                y = (y - lower_bnd) / (upper_bnd - lower_bnd)

                num_episodes = len(data['episode'])

                results[file] = {
                    'auc': metrics.auc(data['episode'], y) / num_episodes,
                    'upper_bnd': upper_bnd
                }
    return results


def aggregate(input_name, targets, do_filter, **kwargs):

    files = glob(input_name)

    vals = {}
    for file in files:
        with open(file) as f:

            data = json.load(f)

            for target in targets:
                if not do_filter or np.any(np.array(data['episode_length'])!=data['episode_length'][0]):
                    vals[target] = vals.get(target, []) + [data[target]]

    return dict(
        data,
        **{
            target : list(np.mean(vals[target], axis=0))
            for target in targets
        },
        **{
            f"{target}_std" : list(np.std(vals[target], axis=0))
            for target in targets
        }
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
        help = 'regexp of file(s) to analyze',
    )

    parser.add_argument(
        '--output_name',
        type=str,
        required=True,
        help = "Name of file in which to dump results",
    )

    parser.add_argument(
        "--function",
        type=str,
        choices = ["aggregate", "auc"],
        required=True,
        help = "Function to be performed on data given",
    )

    parser.add_argument(
        "--targets",
        type=str,
        nargs = "+",
        choices = ["episode_length", "return"],
        default = ["episode_length", "return"],
        help = "Value on which to perform analytics",
    )

    parser.add_argument(
        "--do_filter",
        action = "store_true",
        default = False,
        help = "Set this flag to filter uncoverged runs",
    )

    parser.add_argument(
        "--lower_bnd",
        type=float,
        default = -500,
        help = "For AUC only: theoretical lower bound of returns",
    )

    config = parser.parse_args()
    main(config)
