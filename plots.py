import json
import numpy as np

from argparse import ArgumentParser
from matplotlib import pyplot as plt

from utils.evaluation import ResultsManager

plt.rcParams.update({'font.size': 20})

def e_length(results, args):
    plt.figure(figsize=(12, 8))
    
    x = np.array(results['episode'])
    y = np.array(results['episode_length'])

    plt.scatter(x, y)
    plt.xlabel('Episode number')
    plt.ylabel('Episode length')
    plt.savefig('results/e_length.png')


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument(
        'results_file',
        help = 'Results file to draw data from.',
    )

    parser.add_argument(
        '--plot',
        help = 'Which plot to generate [e_length]',
    )

    args = parser.parse_args()

    with open(args.results_file) as f:
        results = json.load(f)

    if args.plot == 'e_length':
        e_length(results, args)
    else:
        print('Unknown plot type ', args.plot)