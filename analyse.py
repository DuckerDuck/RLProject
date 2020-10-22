import argparse
import json
import os
from glob import glob
from sklearn import metrics

import numpy as np

SEED = 42

def auc(input_name, targets, **kwargs):
    """Area under Curve + Asymptotic
        Assumes that results are already aggregated!
    """
    files = glob(input_name + '*')

    results = dict(per_file = dict(), total = dict())
    for file in files:
        with open(file) as f:
            data = json.load(f)
            for target in targets:
                y = np.array(data[target])

                # Asymptotic
                asymptote = y[-1]

                upper_bnd = np.max(y)
                lower_bnd = np.min(y)

                # Normalize returns
                if upper_bnd != lower_bnd:
                    y = (y - lower_bnd) / (upper_bnd - lower_bnd)

                num_episodes = len(data['episode'])

                results['per_file' ][file] = {
                    'auc': metrics.auc(data['episode'], y) / num_episodes if upper_bnd != lower_bnd else 1,
                    'asymptote': asymptote
                }

    results['total']['auc'] = dict(
        mean = np.mean(np.array([x['auc'] for x in results['per_file'].values()])),
        std = np.std(np.array([x['auc'] for x in results['per_file'].values()])),
    )

    results['total']['asymptote'] = dict(
        mean = np.mean(np.array([x['asymptote'] for x in results['per_file'].values()])),
        std = np.std(np.array([x['asymptote'] for x in results['per_file'].values()])),
    )

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

    config = parser.parse_args()
    ret = main(config)
