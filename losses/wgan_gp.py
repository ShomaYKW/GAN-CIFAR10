"""
losses/wgan_gp.py
WGAN-GP (Gulrajani et al. 2017).
Critic:    L_D = E[D(x_fake)] - E[D(x_real)] + λ E_x̂[(||∇_x̂ D(x̂)||_2 - 1)^2]
           x̂ = ε x_real + (1-ε) x_fake,  ε ~ U[0,1] per sample
Generator: L_G = -E[D(G(z))]
"""
import torch


def gradient_penalty(D, real, fake, lam=10.0):
    eps = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    x_hat = (eps * real + (1 - eps) * fake.detach()).requires_grad_(True)
    grad, = torch.autograd.grad(D(x_hat).sum(), x_hat, create_graph=True)
    gnorm = grad.flatten(1).norm(2, dim=1)
    return lam * ((gnorm - 1) ** 2).mean(), gnorm.mean().detach()


def critic_loss(D, real, fake, lam=10.0):
    d_real, d_fake = D(real), D(fake.detach())
    gp, gnorm = gradient_penalty(D, real, fake, lam)
    w_dist = d_real.mean() - d_fake.mean()  # Wasserstein estimate (for logging)
    return -w_dist + gp, {"w_dist": w_dist.item(), "gp": gp.item(), "gnorm": gnorm.item()}


def generator_loss(D, fake):
    return -D(fake).mean()
