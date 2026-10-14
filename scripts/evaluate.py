"""
scripts/evaluate.py
Step 3: 100-sample grid, Inception Score (50K, 10 splits), FID (mean ± std over repeats).
Usage: python scripts/evaluate.py --variant wgangp_g256_bs128 [--ckpt path] [--no_fid]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml

from data.dataloader import get_real_images
from metrics import (feature_stats, fid, inception_score, inception_score_from_images,
                     load_classifier)
from models import Generator
from utils import latest_checkpoint, load_checkpoint, save_grid

CLF_PATH = "outputs/classifier/inception_cifar.pt"
REAL_CACHE = "outputs/classifier/real_stats.npz"


def real_reference(clf, device, n_real):
    """Real-data FID stats + real-data IS, cached once and shared by all variants."""
    if os.path.exists(REAL_CACHE):
        d = np.load(REAL_CACHE)
        return (d["mu"], d["sigma"]), (float(d["is_m"]), float(d["is_s"]))
    real = get_real_images(n_real, train=True)
    mu, sigma = feature_stats(clf, real, device)
    is_m, is_s = inception_score_from_images(clf, real, device)
    np.savez(REAL_CACHE, mu=mu, sigma=sigma, is_m=is_m, is_s=is_s)
    return (mu, sigma), (is_m, is_s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--ckpt", default=None, help="default: latest in outputs/<variant>/checkpoints")
    p.add_argument("--n_is", type=int, default=50000)
    p.add_argument("--n_fid", type=int, default=10000)
    p.add_argument("--fid_repeats", type=int, default=5)
    p.add_argument("--n_real", type=int, default=50000)
    p.add_argument("--no_fid", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out_dir = os.path.join("outputs", args.variant)
    cfg = yaml.safe_load(open(os.path.join(out_dir, "config.yaml")))

    G = Generator(cfg["model"]["z_dim"], cfg["model"]["g_ch"]).to(device)
    ckpt = args.ckpt or latest_checkpoint(os.path.join(out_dir, "checkpoints"))
    it = load_checkpoint(ckpt, G, device=device)
    G.eval()
    clf = load_classifier(CLF_PATH, device)
    print(f"{args.variant} @ iter {it}  ({ckpt})")

    # ---- 100 samples ----
    save_grid(G.sample(100, device).cpu(), os.path.join(out_dir, "samples", "final_100.png"))

    # ---- IS ----
    is_m, is_s = inception_score(G, clf, device, n=args.n_is)
    res = {"variant": args.variant, "iter": it, "IS_mean": is_m, "IS_std": is_s}
    print(f"IS  = {is_m:.3f} ± {is_s:.3f}")

    # ---- FID (bonus) + real-data reference ----
    real_stats, (ris_m, ris_s) = real_reference(clf, device, args.n_real)
    res.update({"IS_real_mean": ris_m, "IS_real_std": ris_s})
    print(f"IS(real) = {ris_m:.3f} ± {ris_s:.3f}")
    if not args.no_fid:
        f_m, f_s = fid(G, clf, real_stats, device, n=args.n_fid, repeats=args.fid_repeats)
        res.update({"FID_mean": f_m, "FID_std": f_s})
        print(f"FID = {f_m:.3f} ± {f_s:.3f}  ({args.fid_repeats}×{args.n_fid} samples)")

    json.dump(res, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
