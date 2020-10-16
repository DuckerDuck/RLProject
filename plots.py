import json
import numpy as np

from argparse import ArgumentParser
from matplotlib import pyplot as plt

from utils.evaluation import ResultsManager

plt.rcParams.update({'font.size': 20})

def e_length(results, args, label=None):
    """Episode versus episode length"""
    x = np.array(results['episode'])
    y = np.array(results['episode_length'])

    # running_avg = np.cumsum(y)/(np.arange(y.shape[0])+1)

    plt.plot(x, y, label=label)

    if 'episode_std' in results:
        std = np.array(results['episode_std'])
        plt.fill_between(x, y-std, y+std, alpha=0.5, label = f"{label+'_' if label is not None else ''}std")

    # plt.plot(x, running_avg, label = f"{label+'_' if label is not None else ''}avg")
    plt.xlabel('Episode number')
    plt.ylabel('Episode length')

def main(args):
    if args.plot not in globals().keys():
        print('Unknown plot type ', args.plot)
        return

    plt.figure(figsize=(12, 8))

    if args.labels is not None and len(args.labels) != len(args.results_files):
        print('Not the same amount of labels and result files given!')
        return

    for i, results_file in enumerate(args.results_files):
        with open(results_file) as f:
            results = json.load(f)

        globals()[args.plot](results, args, label=args.labels[i] if args.labels is not None else None)

    plt.legend()

    if args.save is not None:
        plt.savefig(args.save)

    if args.show:
        plt.show()

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument(
        '--results_files',
        help = 'Results files to draw data from.',
        nargs='*',
        required=True,
    )

    parser.add_argument(
        '--plot',
        choices = ['e_length'],
        help = 'Which plot to generate.',
        required = True,
    )

    parser.add_argument(
        '--labels',
        help = 'Is using multiple results files, add label to the axis.',
        default = None,
        nargs='*'
    )

    parser.add_argument(
        '--show',
        action='store_true',
        default = False,
        help = 'Set to show figure'
    )

    parser.add_argument(
        '--save',
        help = 'Set to name in which to save figure. If not set figure is not saved',
        default = None,
    )

    args = parser.parse_args()
    main(args)
