"""
scripts/train_gan.py
Step 2: train one WGAN-GP variant. Resumes automatically from the latest checkpoint.
Usage: python scripts/train_gan.py --config configs/wgangp_g256_bs128.yaml
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from data.dataloader import get_gan_loader, infinite_loader
from losses import real_fake_penalty, wgan_gp
from metrics import inception_score, load_classifier
from models import Discriminator, Generator
from utils import (latest_checkpoint, linear_decay, load_checkpoint, plot_curves,
                   save_checkpoint, save_fixed_z_grid)

CLF_PATH = "outputs/classifier/inception_cifar.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    m, t, l, lg = cfg["model"], cfg["train"], cfg["loss"], cfg["log"]

    out_dir = os.path.join("outputs", cfg["name"])
    ckpt_dir, sample_dir = os.path.join(out_dir, "checkpoints"), os.path.join(out_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(out_dir, "config.yaml"))

    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # ---- models / optim ----
    G = Generator(m["z_dim"], m["g_ch"]).to(device)
    D = Discriminator(m["d_ch"]).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=t["lr"], betas=tuple(t["betas"]))
    opt_D = torch.optim.Adam(D.parameters(), lr=t["lr"], betas=tuple(t["betas"]))
    sch_G, sch_D = linear_decay(opt_G, t["decay_iters"]), linear_decay(opt_D, t["decay_iters"])

    # ---- resume ----
    start, log_path = 0, os.path.join(out_dir, "log.json")
    log = {}
    ck = latest_checkpoint(ckpt_dir)
    if ck:
        start = load_checkpoint(ck, G, D, opt_G, opt_D, sch_G, sch_D, device)
        log = json.load(open(log_path)) if os.path.exists(log_path) else {}
        print(f"Resumed from {ck} @ iter {start}")

    # ---- loss dispatch ----
    if l["type"] == "wgan_gp":
        d_loss = lambda real, fake, it: wgan_gp.critic_loss(D, real, fake, l["lam"])
        g_loss = lambda fake, it: wgan_gp.generator_loss(D, fake)
    elif l["type"] == "real_fake":
        kw = dict(sigma0=l["sigma0"], anneal_iters=l["anneal_iters"])
        d_loss = lambda real, fake, it: real_fake_penalty.critic_loss(
            D, real, fake, it, l["lam"], l["mode"], **kw)
        g_loss = lambda fake, it: real_fake_penalty.generator_loss(D, fake, it, **kw)
    else:
        raise ValueError(l["type"])

    data = infinite_loader(get_gan_loader(t["batch_size"], t["num_workers"]))
    z_fixed = torch.randn(100, m["z_dim"], generator=torch.Generator().manual_seed(123)).to(device)
    clf = load_classifier(CLF_PATH, device) if lg["is_every"] and os.path.exists(CLF_PATH) else None
    B = t["batch_size"]

    def rec(k, it, v):
        log.setdefault(k, []).append((it, v))

    t0 = time.time()
    for it in range(start, t["train_iters"]):
        # ---- critic: n_critic steps ----
        for _ in range(t["n_critic"]):
            real = next(data).to(device, non_blocking=True)
            with torch.no_grad():
                fake = G(torch.randn(B, m["z_dim"], device=device))
            lossD, stats = d_loss(real, fake, it)
            opt_D.zero_grad(set_to_none=True)
            lossD.backward()
            opt_D.step()

        # ---- generator: 1 step (D frozen) ----
        for prm in D.parameters():
            prm.requires_grad_(False)
        lossG = g_loss(G(torch.randn(B, m["z_dim"], device=device)), it)
        opt_G.zero_grad(set_to_none=True)
        lossG.backward()
        opt_G.step()
        for prm in D.parameters():
            prm.requires_grad_(True)

        sch_G.step()
        sch_D.step()
        n = it + 1

        # ---- logging ----
        if n % lg["log_every"] == 0:
            stats["loss_G"] = lossG.item()
            for k, v in stats.items():
                rec(k, n, v)
            print(f"[{n:6d}] " + " ".join(f"{k} {v:.4f}" for k, v in stats.items())
                  + f" lr {sch_G.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s")
        if n % lg["sample_every"] == 0:
            save_fixed_z_grid(G, z_fixed, os.path.join(sample_dir, f"iter_{n:06d}.png"))
        if clf is not None and n % lg["is_every"] == 0:
            is_m, is_s = inception_score(G, clf, device, n=5000)
            rec("IS", n, is_m)
            print(f"[{n:6d}] IS(5K) {is_m:.3f} ± {is_s:.3f}")
        if n % lg["ckpt_every"] == 0 or n == t["train_iters"]:
            save_checkpoint(os.path.join(ckpt_dir, f"iter_{n:06d}.pt"),
                            n, G, D, opt_G, opt_D, sch_G, sch_D)
            json.dump(log, open(log_path, "w"))
            plot_curves(log, os.path.join(out_dir, "curves.png"),
                        keys=("w_dist", "gp", "loss_G", "IS"))

    print(f"Done: {cfg['name']}")


if __name__ == "__main__":
    main()
