"""
data/dataloader.py
CIFAR-10 loaders for GAN training, Inception-classifier training, and FID/IS evaluation.
Expects: data/cifar-10-python.tar.gz (already downloaded). Extracts once if needed.
"""
import os
import tarfile

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))  # .../data
ARCHIVE = os.path.join(DATA_ROOT, "cifar-10-python.tar.gz")
EXTRACTED = os.path.join(DATA_ROOT, "cifar-10-batches-py")


def _ensure_extracted():
    if os.path.isdir(EXTRACTED):
        return
    if not os.path.isfile(ARCHIVE):
        raise FileNotFoundError(f"Missing {ARCHIVE}")
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(DATA_ROOT)


# ---------- transforms ----------
# GAN: images in [-1, 1] to match generator Tanh output
GAN_TF = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3, (0.5,) * 3),
])

# Classifier: same [-1, 1] range so it can score generator outputs directly
CLS_TRAIN_TF = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3, (0.5,) * 3),
])
CLS_TEST_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3, (0.5,) * 3),
])


def _dataset(train: bool, tf):
    _ensure_extracted()
    return datasets.CIFAR10(root=DATA_ROOT, train=train, transform=tf, download=False)


# ---------- GAN ----------
def get_gan_loader(batch_size=128, num_workers=4):
    """Train split, shuffled, drop_last so every step has exactly batch_size."""
    return DataLoader(
        _dataset(True, GAN_TF), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )


def infinite_loader(loader):
    """Iteration-based training (100K iters) instead of epochs. Yields images only."""
    while True:
        for x, _ in loader:
            yield x


# ---------- Inception classifier ----------
def get_classifier_loaders(batch_size=128, num_workers=4):
    train = DataLoader(
        _dataset(True, CLS_TRAIN_TF), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test = DataLoader(
        _dataset(False, CLS_TEST_TF), batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train, test


# ---------- Evaluation (FID reference stats) ----------
def get_real_images(n=10000, train=True, batch_size=500, num_workers=4):
    """Returns up to n real images as a [n,3,32,32] tensor in [-1,1], no augmentation."""
    loader = DataLoader(_dataset(train, CLS_TEST_TF), batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)
    out, count = [], 0
    for x, _ in loader:
        out.append(x)
        count += x.size(0)
        if count >= n:
            break
    return torch.cat(out)[:n]