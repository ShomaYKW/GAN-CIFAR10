"""
losses/real_fake_penalty.py
Bonus variant (Mescheder et al. 2018): instance noise + gradient penalties evaluated
ON real and generated samples only (no interpolates).

Penalty modes:
  "zero" (default, [3]): R1 + R2 = (γ/2) E_real||∇D||^2 + (γ/2) E_fake||∇D||^2
                         -> pushes D toward zero gradient on the data/generator manifolds,
                            which gives local convergence (Mescheder Thm 4.1).
  "one":                 λ E_{real∪fake}[(||∇D|| - 1)^2]   (WGAN-GP target, different points)

Instance noise: x <- x + σ_t n, n ~ N(0, I), applied to both real and fake inputs of D.
σ_t anneals linearly σ_0 -> 0, smoothing both distributions so their supports overlap early on.
"""
import torch


def instance_noise_sigma(it, sigma0=0.1, anneal_iters=50000):
    return sigma0 * max(0.0, 1.0 - it / anneal_iters)


def add_noise(x, sigma):
    return x + sigma * torch.randn_like(x) if sigma > 0 else x


def _grad_sq_norm(D, x):
    x = x.detach().requires_grad_(True)
    out = D(x)
    grad, = torch.autograd.grad(out.sum(), x, create_graph=True)
    return out, grad.flatten(1).pow(2).sum(1)  # ||∇D||^2 per sample


def critic_loss(D, real, fake, it, lam=10.0, mode="zero", sigma0=0.1, anneal_iters=50000):
    sigma = instance_noise_sigma(it, sigma0, anneal_iters)
    real_n, fake_n = add_noise(real, sigma), add_noise(fake.detach(), sigma)

    d_real, g2_real = _grad_sq_norm(D, real_n)
    d_fake, g2_fake = _grad_sq_norm(D, fake_n)

    if mode == "zero":
        pen = 0.5 * lam * (g2_real.mean() + g2_fake.mean())
    elif mode == "one":
        g = torch.cat([g2_real, g2_fake]).add(1e-12).sqrt()
        pen = lam * ((g - 1) ** 2).mean()
    else:
        raise ValueError(mode)

    w_dist = d_real.mean() - d_fake.mean()
    return -w_dist + pen, {"w_dist": w_dist.item(), "gp": pen.item(), "sigma": sigma,
                           "gnorm_real": g2_real.sqrt().mean().item(),
                           "gnorm_fake": g2_fake.sqrt().mean().item()}


def generator_loss(D, fake, it, sigma0=0.1, anneal_iters=50000):
    return -D(add_noise(fake, instance_noise_sigma(it, sigma0, anneal_iters))).mean()
