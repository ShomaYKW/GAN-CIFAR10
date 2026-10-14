"""
metrics/fid.py
FID = ||mu_r - mu_g||^2 + Tr(S_r + S_g - 2 (S_r S_g)^{1/2})
on 336-d penultimate features of InceptionCIFAR.
Mean ± std: real stats fixed; FID recomputed over `repeats` independent draws of generated samples.
"""
import numpy as np
from scipy import linalg

from .common import batched, generate


def feature_stats(clf, images, device):
    f = batched(clf.features, images, device).double().numpy()
    return f.mean(0), np.cov(f, rowvar=False)


def frechet_distance(mu1, s1, mu2, s2, eps=1e-6):
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():  # near-singular product: regularize
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
    covmean = covmean.real  # drop tiny imaginary parts from numerical error
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def fid(G, clf, real_stats, device, n=10000, repeats=5):
    """real_stats = feature_stats(clf, real_images, device), computed once and reused."""
    mu_r, s_r = real_stats
    vals = [frechet_distance(mu_r, s_r, *feature_stats(clf, generate(G, n, device), device))
            for _ in range(repeats)]
    return float(np.mean(vals)), float(np.std(vals))
