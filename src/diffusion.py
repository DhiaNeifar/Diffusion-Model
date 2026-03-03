import math
import torch

def make_linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)

def precompute_ddpm_terms(betas: torch.Tensor, device: torch.device):
    betas = betas.to(device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": sqrt_alpha_bar,
        "sqrt_one_minus_alpha_bar": sqrt_one_minus_alpha_bar,
    }

def extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    b = t.shape[0]
    out = a.gather(0, t)
    return out.view(b, 1, 1, 1).expand(x_shape)

def q_sample(x0: torch.Tensor, t: torch.Tensor, sqrt_alpha_bar: torch.Tensor, sqrt_one_minus_alpha_bar: torch.Tensor, noise: torch.Tensor):
    s1 = extract(sqrt_alpha_bar, t, x0.shape)
    s2 = extract(sqrt_one_minus_alpha_bar, t, x0.shape)
    return s1 * x0 + s2 * noise

@torch.no_grad()
def p_sample_step_ddpm(x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor, model, ddpm_terms: dict, w: float, null_label: int):
    betas = ddpm_terms["betas"]
    alphas = ddpm_terms["alphas"]
    alpha_bar = ddpm_terms["alpha_bar"]

    eps_cond = model(x_t, t, y)
    y_null = torch.full_like(y, null_label)
    eps_uncond = model(x_t, t, y_null)
    eps_hat = (1.0 + w) * eps_cond - w * eps_uncond

    beta_t = extract(betas, t, x_t.shape)
    alpha_t = extract(alphas, t, x_t.shape)
    alpha_bar_t = extract(alpha_bar, t, x_t.shape)

    mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_hat)

    if t[0].item() == 0:
        return mean

    noise = torch.randn_like(x_t)
    sigma = torch.sqrt(beta_t)
    return mean + sigma * noise

@torch.no_grad()
def sample_loop(model, ddpm_terms: dict, device: torch.device, T: int, label: int, n: int, w: float, img_size: int, null_label: int):
    x = torch.randn(n, 1, img_size, img_size, device=device)
    y = torch.full((n,), int(label), device=device, dtype=torch.long)
    for ti in reversed(range(T)):
        t = torch.full((n,), ti, device=device, dtype=torch.long)
        x = p_sample_step_ddpm(x, t, y, model, ddpm_terms, w=w, null_label=null_label)
    return x.clamp(-1, 1)
