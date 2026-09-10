import torch
import torch.nn.functional as F
import numpy as np


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)


class Diffusion:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        self.device = device
        betas = linear_beta_schedule(T).to(device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        """Forward process: q(x_t | x_0) = N(sqrt(alpha_bar_t)*x_0, (1-alpha_bar_t)*I)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_ac * x0 + sqrt_om * noise

    @torch.no_grad()
    def p_sample_loop(self, model, n, img_size=28, steps=None, verbose=True):
        """Ancestral sampling from pure noise, optionally with fewer steps."""
        steps = steps or self.T
        # Subsample timestep indices uniformly from T (descending: T-1 -> 0)
        time_seq = list(np.linspace(self.T - 1, 0, steps, dtype=int))
        model.eval()
        x = torch.randn(n, 1, img_size, img_size, device=self.device)
        iterator = range(len(time_seq))
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"Sampling ({steps} steps)", leave=False)
        for i in iterator:
            t = time_seq[i]
            batch_t = torch.full((n,), t, device=self.device, dtype=torch.long)
            pred_noise = model(x, batch_t)
            alpha_bar = self.alphas_cumprod[t]
            # Predict x0 from noise prediction
            x0_pred = (x - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            if i < len(time_seq) - 1:
                t_prev = time_seq[i + 1]
                alpha_bar_prev = self.alphas_cumprod[t_prev]
                sigma = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev - sigma**2) * pred_noise + sigma * noise
            else:
                x = x0_pred
        return torch.clamp(x, -1.0, 1.0)

    @torch.no_grad()
    def ddim_sample_loop(self, model, n, img_size=28, steps=50, eta=0.0, verbose=True):
        """DDIM deterministic sampling with fewer steps."""
        time_seq = list(np.linspace(0, self.T - 1, steps, dtype=int))[::-1]
        model.eval()
        x = torch.randn(n, 1, img_size, img_size, device=self.device)
        iterator = range(len(time_seq))
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"DDIM ({steps} steps)", leave=False)
        for i in iterator:
            t = time_seq[i]
            batch_t = torch.full((n,), t, device=self.device, dtype=torch.long)
            pred_noise = model(x, batch_t)
            alpha_bar = self.alphas_cumprod[t]
            x0_pred = (x - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            if i < len(time_seq) - 1:
                t_prev = time_seq[i + 1]
                alpha_bar_prev = self.alphas_cumprod[t_prev]
                sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev))
                noise = torch.randn_like(x) if eta > 0 else 0.0
                x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev - sigma**2) * pred_noise + sigma * noise
            else:
                x = x0_pred
        return x
