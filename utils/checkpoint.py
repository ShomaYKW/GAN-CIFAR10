"""
utils/checkpoint.py
Save / resume full training state: G, D, optimizers, schedulers, iteration.
"""
import os

import torch


def save_checkpoint(path, it, G, D, opt_G, opt_D, sch_G, sch_D):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "iter": it,
        "G": G.state_dict(), "D": D.state_dict(),
        "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
        "sch_G": sch_G.state_dict(), "sch_D": sch_D.state_dict(),
    }, path)


def load_checkpoint(path, G, D=None, opt_G=None, opt_D=None, sch_G=None, sch_D=None,
                    device="cpu"):
    """Returns the iteration to resume from. Pass only G for evaluation."""
    ck = torch.load(path, map_location=device)
    G.load_state_dict(ck["G"])
    for obj, key in [(D, "D"), (opt_G, "opt_G"), (opt_D, "opt_D"),
                     (sch_G, "sch_G"), (sch_D, "sch_D")]:
        if obj is not None:
            obj.load_state_dict(ck[key])
    return ck["iter"]


def latest_checkpoint(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
    return os.path.join(ckpt_dir, files[-1]) if files else None
