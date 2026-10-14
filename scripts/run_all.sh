#!/usr/bin/env bash
# Full pipeline. Run from project root: bash scripts/run_all.sh
set -e

VARIANTS=(wgangp_g256_bs128 wgangp_g128_bs256 wgangp_noise_r1r2)

# 1. classifier (skip if trained)
[ -f outputs/classifier/inception_cifar.pt ] || python scripts/train_classifier.py

# 2. GANs
for v in "${VARIANTS[@]}"; do
  python scripts/train_gan.py --config configs/$v.yaml
done

# 3. evaluation
for v in "${VARIANTS[@]}"; do
  python scripts/evaluate.py --variant $v
done
