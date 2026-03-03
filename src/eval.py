import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

from .config import TrainConfig
from .utils import get_device, print_device_info, set_seed, find_latest_checkpoint, load_checkpoint
from .data import make_datasets, make_loaders
from .model import SimpleEpsModel
from .diffusion import make_linear_beta_schedule, precompute_ddpm_terms, q_sample

@torch.no_grad()
def eval_loss(model, loader, ddpm_terms, cfg: TrainConfig, device: torch.device, t_bins: int = 5):
    model.eval()
    sqrt_alpha_bar = ddpm_terms["sqrt_alpha_bar"]
    sqrt_one_minus_alpha_bar = ddpm_terms["sqrt_one_minus_alpha_bar"]

    total = 0.0
    count = 0
    bin_sum = torch.zeros(t_bins, device=device)
    bin_count = torch.zeros(t_bins, device=device)
    bin_edges = torch.linspace(0, cfg.T, t_bins + 1, device=device)
    for x0, y in tqdm(loader, desc="eval", leave=False):
        x0 = x0.to(device)
        y = y.to(device)
        t = torch.randint(0, cfg.T, (x0.size(0),), device=device).long()
        noise = torch.randn_like(x0)
        xt = q_sample(x0, t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, noise)

        pred = model(xt, t, y)
        mse_per_sample = (pred - noise).pow(2).flatten(1).mean(dim=1)
        total += mse_per_sample.mean().item()
        count += 1
        bin_idx = torch.bucketize(t.float(), bin_edges[1:-1], right=False)
        for bi in range(t_bins):
            m = (bin_idx == bi)
            if m.any():
                bin_sum[bi] += mse_per_sample[m].sum()
                bin_count[bi] += m.sum()

    bin_loss = []
    for bi in range(t_bins):
        left = int(bin_edges[bi].item())
        right = int(bin_edges[bi + 1].item()) - 1
        denom = max(bin_count[bi].item(), 1.0)
        bin_loss.append((left, right, (bin_sum[bi] / denom).item()))
    return total / max(1, count), bin_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ckpt_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--t_bins", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    cfg = TrainConfig(T=args.T, batch_size=args.batch_size, num_workers=args.num_workers, ckpt_dir=args.ckpt_dir, seed=args.seed)

    device = get_device()
    print_device_info(device)
    set_seed(cfg.seed)

    ckpt_path = args.ckpt_path.strip()
    if not ckpt_path:
        ckpt_path = find_latest_checkpoint(cfg.ckpt_dir)
    if not ckpt_path:
        raise FileNotFoundError("No checkpoint found. Train first or pass --ckpt_path.")

    train_ds, val_ds = make_datasets()
    _, val_loader = make_loaders(train_ds, val_ds, cfg.batch_size, cfg.num_workers)

    model = SimpleEpsModel(num_classes=cfg.num_classes + 1).to(device)
    try:
        load_checkpoint(ckpt_path, model, optimizer=None, map_location=device)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture mismatch. Retrain from scratch with the current model "
            "or pass a compatible --ckpt_path."
        ) from exc

    betas = make_linear_beta_schedule(cfg.T)
    ddpm_terms = precompute_ddpm_terms(betas, device=device)

    vloss, t_bin_loss = eval_loss(model, val_loader, ddpm_terms, cfg, device, t_bins=args.t_bins)
    print("checkpoint:", ckpt_path)
    print("val_loss:", f"{vloss:.6f}")
    print("loss_by_t_bin:")
    for left, right, bl in t_bin_loss:
        print(f"  t in [{left:4d}, {right:4d}] -> {bl:.6f}")

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
