"""
utils/visualize.py
Sample grids and training curves.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image


def save_grid(images, path, nrow=10):
    """images in [-1,1]; 100 images -> 10x10 grid."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(images.clamp(-1, 1), path, nrow=nrow, normalize=True, value_range=(-1, 1))


@torch.no_grad()
def save_fixed_z_grid(G, z_fixed, path):
    """Same z every call -> track one set of samples across training."""
    was_training = G.training
    G.eval()
    save_grid(G(z_fixed).cpu(), path)
    G.train(was_training)


def plot_curves(log, path, keys=("w_dist", "gp", "loss_G")):
    """log: dict key -> list of (iter, value)."""
    keys = [k for k in keys if log.get(k)]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.5))
    for ax, k in zip(axes if len(keys) > 1 else [axes], keys):
        it, v = zip(*log[k])
        ax.plot(it, v, lw=0.8)
        ax.set_title(k)
        ax.set_xlabel("generator iter")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
