import os
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from utils.transforms import *

def edm_sampler_multistep(
    init_x, net, diffusion_steps, num_steps=18, class_labels=None, randn_like=torch.randn_like,
    sigma_min=0.002, sigma_max=80, rho=7, S_churn=0, S_min=0, S_max=float('inf'), S_noise=1
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=init_x.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    # Main sampling loop.
    n = torch.randn_like(init_x) * t_steps[diffusion_steps]
    x_next = (init_x + n).to(torch.float64)
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[diffusion_steps:-1], t_steps[diffusion_steps+1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

    return x_next.to(torch.float32)

def purify_x_edm_multistep(X, score, args):
    transform_diff = raw_to_diff(args.dataset)
    transform_raw = diff_to_raw(args.dataset)
    X = transform_diff(X)
    purif_X_re = torch.clamp(transform_raw(edm_sampler_multistep(X, score, int(args.forward_steps), args.total_steps)),0.0,1.0)
    
    return purif_X_re

def edm_sampler_one_shot(init_x, net, t, num_steps, class_labels=None):

    t = torch.tensor(t, dtype=torch.float64, device=init_x.device)
    n = torch.randn_like(init_x) * t
    x_t = (init_x + n).to(torch.float64)
    denoised = net(x_t, t, class_labels)

    return denoised

def purify_x_edm_one_shot(X, score, args):
    transform_diff = raw_to_diff(args.dataset)
    transform_raw = diff_to_raw(args.dataset)
    X = transform_diff(X)
    purif_X_re = torch.clamp(transform_raw(edm_sampler_one_shot(X, score, args.forward_steps, args.total_steps)),0.0,1.0)
    
    return purif_X_re

class ImageModel(nn.Module):
    def __init__(self, init_x):
        super().__init__()
        self.img = nn.Parameter(init_x.clone())
        self.img.requires_grad = True

    def encode(self):
        return self.img
    
def sampler_opt_x0(
    init_x, net, init_t, num_steps, args, class_labels=None, randn_like=torch.randn_like,
    sigma_min=0.002, sigma_max=80, rho=7
):
    # # Use all noise levels

    # # Adjust noise levels based on what's supported by the network.
    # sigma_min = max(sigma_min, net.sigma_min)
    # sigma_max = min(sigma_max, net.sigma_max, t)
    # # forward_steps = int(forward_steps)

    # # Time step discretization.
    # step_indices = torch.arange(num_steps, dtype=torch.float64, device=init_x.device)
    # t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    # t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    # # Main sampling loop.
    # # n = torch.randn_like(init_x) * t_steps[diffusion_steps]
    # # x_next = (init_x + n).to(torch.float64)

    model = ImageModel(init_x).to(init_x.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for i in range(args.purify_iter):
        sample_img = model.encode()
        t = random.uniform(init_t - 0.1, init_t + 0.1)
        t = torch.tensor(t, dtype=torch.float64, device=init_x.device)
        eps = torch.randn_like(init_x)
        sample_img_t = sample_img + eps * t
        denoised = net(sample_img_t, t, class_labels)
        # eps_hat = (sample_img_t - denoised) / t

        # eps_1 = torch.randn_like(init_x)
        # sample_img_t_1 = init_x + eps_1 * t
        # denoised_1 = net(sample_img_t_1, t, class_labels)
        # eps_hat_1 = (sample_img_t_1 - denoised) / t

        # loss = F.mse_loss(denoised, sample_img) + args.loss_lambda * F.mse_loss(denoised, denoised_1) # SR
        # loss = F.mse_loss(denoised, sample_img) + args.loss_lambda * F.mse_loss(sample_img, init_x) # MSE
        loss = F.mse_loss(denoised, sample_img) # Diff
        opt.zero_grad()
        loss.backward()
        opt.step()

    sample_img = model.encode().detach()

    return sample_img

def sampler_opt_xt(
    init_x, net, t, num_steps, args, class_labels=None
):
    t = torch.tensor(t, dtype=torch.float64, device=init_x.device)
    eps = torch.randn_like(init_x)
    x_t = init_x + eps * t

    model = ImageModel(x_t).to(init_x.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for i in range(args.purify_iter):

        sample_img = model.encode()
        denoised = net(sample_img, t, class_labels)
        eps_hat = (sample_img - denoised) / t
        # loss = F.mse_loss(denoised, sample_img)
        loss = F.mse_loss(eps_hat, eps)
        opt.zero_grad()
        loss.backward()
        opt.step()

    sample_img = model.encode().detach()
    denoised = net(sample_img, t, class_labels)

    return denoised

def purify_x_opt_x0(X, score, args):
    transform_diff = raw_to_diff(args.dataset)
    transform_raw = diff_to_raw(args.dataset)
    X = transform_diff(X)
    purif_X_re = torch.clamp(transform_raw(sampler_opt_x0(X, score, args.forward_steps, args.total_steps, args)),0.0,1.0)
    return purif_X_re

def purify_x_opt_xt(X, score, args):
    transform_diff = raw_to_diff(args.dataset)
    transform_raw = diff_to_raw(args.dataset)
    X = transform_diff(X)
    purif_X_re = torch.clamp(transform_raw(sampler_opt_xt(X, score, args.forward_steps, args.total_steps, args)),0.0,1.0)
    return purif_X_re