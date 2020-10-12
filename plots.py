import json
import numpy as np

from argparse import ArgumentParser
from matplotlib import pyplot as plt

from utils.evaluation import ResultsManager

plt.rcParams.update({'font.size': 20})

def e_length(results, args):
    """Episode versus episode length"""
    x = np.array(results['episode'])
    y = np.array(results['episode_length'])

    plt.scatter(x, y)
    plt.xlabel('Episode number')
    plt.ylabel('Episode length')
    

def main(args):
    if args.plot not in globals().keys():
        print('Unknown plot type ', args.plot)
        return

    plt.figure(figsize=(12, 8))

    if len(args.labels) != len(args.results_files):
        print('Not the same amount of labels and result files given!')
        return

    for results_file in args.results_files:
        with open(results_file) as f:
            results = json.load(f)

        globals()[args.plot](results, args)

    plt.legend(args.labels)
    plt.savefig(f'results/{args.plot}.png')

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument(
        'results_files',
        help = 'Results files to draw data from.',
        nargs='*'
    )

    parser.add_argument(
        '--plot',
        help = 'Which plot to generate. [e_length]',
    )

    parser.add_argument(
        '--labels',
        help = 'Is using multiple results files, add label to the axis.',
        nargs='*'
    )

    args = parser.parse_args()
    print(args)
    main(args)