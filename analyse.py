import argparse
import json
import os
from glob import glob

import numpy as np

SEED = 42

def aggregate(input_name, targets, **kwargs):

    files = glob(input_name)

    vals = {}
    for file in files:
        with open(file) as f:

            data = json.load(f)

            for target in targets:
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
        choices = ["aggregate"],
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

    config = parser.parse_args()
    main(config)
