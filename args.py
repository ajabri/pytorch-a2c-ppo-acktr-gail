import argparse

import torch
from pdb import set_trace as st

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--obs-interval', type=int, default=1)
    parser.add_argument(
        '--predict-interval', type=int, default=1)
    parser.add_argument(
        '--seed', type=int, default=123)
    parser.add_argument('--no-op', default=False, action='store_true')
    parser.add_argument('--delta-pos', default=False, action='store_true')
    parser.add_argument('--ops', default=False, action='store_true')
    parser.add_argument(
        '--rl-num-iter', type=int, default=100)

    parser.add_argument(
        '--env-name',
        type=str,
        default='relocate-v0')

    args = parser.parse_args()

    return args
