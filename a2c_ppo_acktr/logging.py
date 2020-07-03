import glob
import os

import torch
import torch.nn as nn

from a2c_ppo_acktr.envs import VecNormalize

import sys
import pdb
import wandb

def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        traceback.print_exception(type, value, tb)
        print
        pdb.post_mortem(tb) # more "modern"
sys.excepthook = info


def make_histograms(obs):
    x = np.histogram2d(obs[:, 0], obs[:, 1], bins=50, density=True)[0].transpose(1, 0)[::-1]
    x = (x-x.min())/ (x-x.min()).max()
    v = np.histogram2d(obs[:, 2], obs[:, 3], bins=50, density=True)[0].transpose(1, 0)[::-1]
    v = (v-v.min())/ (v-v.min()).max()
    
    # ang = np.histogram(obs[:, 4], bins=100, density=True) 
    # ang_v = np.histogram(obs[:, 5], bins=100, density=True)
    
    ang = obs[: 4]
    ang_v = obs[:, 5]

    return x, v, ang, ang_v

def wandb_lunarlander(capt, pred):
    wandb.log({
        # "xhist %s" % j: [wandb.Image(cm.jet(_x*255)) for _x in (x1, x2)],
        # "vhist %s" % j: [wandb.Image(cm.jet(_v*255)) for _v in (v1, v2)],
        "cap x": wandb.Histogram(capt[:, 0]),
        "cap y": wandb.Histogram(capt[:, 1]),
        "cap vx": wandb.Histogram(capt[:, 2]),
        "cap vy": wandb.Histogram(capt[:, 3]),

        "pred x": wandb.Histogram(pred[:, 0]),
        "pred y": wandb.Histogram(pred[:, 1]),
        "pred vx": wandb.Histogram(pred[:, 2]),
        "pred vy": wandb.Histogram(pred[:, 3]),

        "cap ang": wandb.Histogram(capt[:, 4]),
        "cap ang_v": wandb.Histogram(capt[:, 5]),
        "pred ang": wandb.Histogram(pred[:, 4]),
        "pred ang_v": wandb.Histogram(pred[:, 5]),
    })

def wandb_minigrid(capt, pred):
    if capt.ndim > 3:
        wandb.log({
            # "xhist %s" % j: [wandb.Image(cm.jet(_x*255)) for _x in (x1, x2)],
            # "vhist %s" % j: [wandb.Image(cm.jet(_v*255)) for _v in (v1, v2)],
            "capt": wandb.Images(capt[:, 0]),
            "pred": wandb.Images(capt[:, 0]),
            "capt mean": wand.Image(capt.mean(0)),
            "pred mean": wand.Image(pred.mean(0)),
        })
    else:
        pass