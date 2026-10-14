"""
scripts/train_classifier.py
Step 1: train InceptionCIFAR from scratch on CIFAR-10. Shared by IS and FID for every GAN variant.
Usage: python scripts/train_classifier.py [--epochs 60]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from data.dataloader import get_classifier_loaders
from models import InceptionCIFAR


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--out", default="outputs/classifier/inception_cifar.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    train_loader, test_loader = get_classifier_loaders(args.batch_size)

    model = InceptionCIFAR().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = nn.CrossEntropyLoss()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        run_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.item()
        sched.step()

        acc = evaluate(model, test_loader, device)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), args.out)
        print(f"ep {ep + 1:3d} | loss {run_loss / len(train_loader):.4f} | "
              f"test acc {acc:.4f} | best {best:.4f}")

    print(f"Saved best classifier ({best:.4f}) -> {args.out}")


if __name__ == "__main__":
    main()
