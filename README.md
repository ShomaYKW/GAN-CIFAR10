# GAN on CIFAR-10 — WGAN-GP Variants

SN-GAN ResNet architecture trained with WGAN-GP on CIFAR-10. 

## Architecture
| Generator | Discriminator |
|---|---|
| z ∈ R¹²⁸ ~ N(0, I) | x ∈ R^{32×32×3} |
| dense → 4×4×C | ResBlock down 128 |
| ResBlock up C ×3 | ResBlock down 128 |
| BN, ReLU, 3×3 conv → 3, Tanh | ResBlock 128 ×2 |
| | ReLU, global sum pool, dense → 1 |

C = 256 (default) or 128 (variant 2).

## Hyperparameters
Adam (α=2e-4, β₁=0, β₂=0.9) · λ=10 · n_critic=5 · LR decayed linearly to 0 over 100K iterations · ≥50K generator iterations.

## Variants
| Config | G filters | Batch | Penalty |
|---|---|---|---|
| `wgangp_g256_bs128.yaml` | 256 | 128 | GP on interpolates |
| `wgangp_g128_bs256.yaml` | 128 | 256 | GP on interpolates |
| `wgangp_noise_r1r2.yaml` (bonus) | 256 | 128 | Instance noise + GP on real & fake samples [3] |

## Pipeline
```bash
# 1. Train the Inception classifier (shared by all variants)
python scripts/train_classifier.py

# 2. Train the GANs
python scripts/train_gan.py --config configs/wgangp_g256_bs128.yaml
python scripts/train_gan.py --config configs/wgangp_g128_bs256.yaml
python scripts/train_gan.py --config configs/wgangp_noise_r1r2.yaml

# 3. Evaluate: 100 samples, IS, FID (mean ± std)
python scripts/evaluate.py --variant <variant_name>

# Or run everything
bash scripts/run_all.sh
```

## Outputs
```
outputs/
├── classifier/            # classifier weights
└── <variant_name>/
    ├── checkpoints/
    ├── samples/           # 10×10 grid (100 samples)
    └── metrics.json       # IS, FID
```

## Metrics
- **Inception Score:** exp(E_x[KL(p(y|x) ‖ p(y))]), computed on 50K samples over 10 splits and reported as mean ± std. Follows the OpenAI improved-gan protocol, with the custom CIFAR-10 classifier replacing Inception-v3.
- **FID (bonus):** ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2(Σ_r Σ_g)^{1/2}), computed on penultimate classifier features. Reported as mean ± std over repeated sample draws.

##folder structure
# `gan_cifar10/`

```
gan_cifar10/
├── configs/
│   ├── wgangp_g256_bs128.yaml      # variant 1
│   ├── wgangp_g128_bs256.yaml      # variant 2
│   └── wgangp_noise_r1r2.yaml      # bonus variant 3
├── models/
│   ├── __init__.py
│   ├── blocks.py                   # ResBlock up / down / plain
│   ├── generator.py                # SN-GAN ResNet G (ch=256 or 128)
│   ├── discriminator.py            # SN-GAN ResNet D (ch=128)
│   └── inception_cifar.py          # Inception-style classifier (from scratch)
├── losses/
│   ├── __init__.py
│   ├── wgan_gp.py                  # interpolated GP (λ=10)
│   └── real_fake_penalty.py        # GP on real + fake only, instance noise
├── data/
│   └── cifar10.py                  # loader, [-1,1] normalization
├── metrics/
│   ├── inception_score.py          # IS w/ custom classifier (10 splits)
│   └── fid.py                      # FID: mean ± std over N runs
├── utils/
│   ├── lr_schedule.py              # linear decay → 0 over 100K
│   ├── checkpoint.py
│   └── visualize.py                # 10×10 sample grid
├── scripts/
│   ├── train_classifier.py         # step 1: train Inception on CIFAR-10
│   ├── train_gan.py                # step 2: --config <yaml>
│   ├── evaluate.py                 # step 3: samples + IS + FID
│   └── run_all.sh
├── outputs/
│   ├── classifier/                 # classifier weights
│   └── <variant_name>/
│       ├── checkpoints/
│       ├── samples/                # 100-sample grid
│       └── metrics.json            # IS, FID
├── requirements.txt
└── README.md
```

## Results
| Variant | IS ↑ | FID ↓ |
|---|---|---|
| G256, bs128 | – | – |
| G128, bs256 | – | – |
| Noise + real/fake GP | – | – |

## References
[2] Gulrajani et al., *Improved Training of Wasserstein GANs*, 2017.
[3] Mescheder et al., *Which Training Methods for GANs do actually Converge?*, 2018.
[4] Miyato et al., *Spectral Normalization for GANs*, 2018.
