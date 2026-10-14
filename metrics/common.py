"""
metrics/common.py
Shared helpers: load the frozen CIFAR classifier, generate samples, batch over images.
"""
import torch

from models import InceptionCIFAR


def load_classifier(path, device):
    clf = InceptionCIFAR().to(device)
    clf.load_state_dict(torch.load(path, map_location=device))
    clf.eval()
    for p in clf.parameters():
        p.requires_grad_(False)
    return clf


@torch.no_grad()
def generate(G, n, device, batch_size=500):
    """n samples in [-1,1] on CPU. G in eval mode (BN running stats)."""
    was_training = G.training
    G.eval()
    out = [G.sample(min(batch_size, n - i), device).cpu() for i in range(0, n, batch_size)]
    G.train(was_training)
    return torch.cat(out)


@torch.no_grad()
def batched(fn, images, device, batch_size=500):
    """Apply fn to images in batches; returns concatenated CPU tensor."""
    return torch.cat([fn(images[i:i + batch_size].to(device)).cpu()
                      for i in range(0, len(images), batch_size)])
