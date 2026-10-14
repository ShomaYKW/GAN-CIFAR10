"""
utils/lr_schedule.py
Linear decay α_t = α_0 (1 - t / T), T = 100K generator iterations, for both G and D.
Stepped once per GENERATOR iteration (D's n_critic inner steps share the same LR).
"""
from torch.optim.lr_scheduler import LambdaLR


def linear_decay(optimizer, total_iters=100_000):
    return LambdaLR(optimizer, lambda it: max(0.0, 1.0 - it / total_iters))
