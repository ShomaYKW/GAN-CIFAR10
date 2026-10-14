"""
metrics/inception_score.py
IS = exp( E_x [ KL( p(y|x) || p(y) ) ] ), following openai/improved-gan:
50K samples, 10 splits, p(y) estimated per split, report mean ± std over splits.
Classifier = InceptionCIFAR trained from scratch (not ImageNet Inception-v3).
"""
import numpy as np
import torch.nn.functional as F

from .common import batched, generate


def inception_score_from_images(clf, images, device, splits=10, eps=1e-12):
    probs = batched(lambda x: F.softmax(clf(x), dim=1), images, device).double().numpy()
    scores = []
    for part in np.array_split(probs, splits):
        py = part.mean(axis=0, keepdims=True)                       # marginal p(y)
        kl = (part * (np.log(part + eps) - np.log(py + eps))).sum(1)  # KL per sample
        scores.append(np.exp(kl.mean()))
    return float(np.mean(scores)), float(np.std(scores))


def inception_score(G, clf, device, n=50000, splits=10):
    return inception_score_from_images(clf, generate(G, n, device), device, splits)
