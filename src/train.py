import argparse
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .config import TrainConfig
from .utils import get_device, print_device_info, set_seed, ensure_dir, save_checkpoint, save_json
from .data import make_datasets, make_loaders
from .model import SimpleEpsModel
from .diffusion import make_linear_beta_schedule, precompute_ddpm_terms, q_sample

def drop_labels(y: torch.Tensor, p_drop: float, null_label: int) -> torch.Tensor:
    mask = (torch.rand_like(y.float()) < p_drop)
    y2 = y.clone()
    y2[mask] = null_label
    return y2

def train_one_epoch(model, optimizer, loader, ddpm_terms, cfg: TrainConfig, device: torch.device, global_step: int):
    model.train()
    betas = ddpm_terms["betas"]
    sqrt_alpha_bar = ddpm_terms["sqrt_alpha_bar"]
    sqrt_one_minus_alpha_bar = ddpm_terms["sqrt_one_minus_alpha_bar"]

    running = 0.0
    for x0, y in tqdm(loader, desc="train", leave=False):
        x0 = x0.to(device)
        y = y.to(device)
        t = torch.randint(0, cfg.T, (x0.size(0),), device=device).long()
        noise = torch.randn_like(x0)

        xt = q_sample(x0, t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, noise)
        y_prime = drop_labels(y, cfg.p_drop, cfg.null_label)

        pred = model(xt, t, y_prime)
        loss = F.mse_loss(pred, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += loss.item()
        global_step += 1

    return running / max(1, len(loader)), global_step


@torch.no_grad()
def eval_one_epoch(model, loader, ddpm_terms, cfg: TrainConfig, device: torch.device):
    model.eval()
    sqrt_alpha_bar = ddpm_terms["sqrt_alpha_bar"]
    sqrt_one_minus_alpha_bar = ddpm_terms["sqrt_one_minus_alpha_bar"]
    total = 0.0
    count = 0
    for x0, y in tqdm(loader, desc="val", leave=False):
        x0 = x0.to(device)
        y = y.to(device)
        t = torch.randint(0, cfg.T, (x0.size(0),), device=device).long()
        noise = torch.randn_like(x0)
        xt = q_sample(x0, t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, noise)
        pred = model(xt, t, y)
        loss = F.mse_loss(pred, noise, reduction="mean").item()
        total += loss
        count += 1
    return total / max(1, count)


def save_loss_curve(train_losses, val_losses, out_path: str) -> None:
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_losses, label="train_loss", marker="o", markersize=3)
    plt.plot(epochs, val_losses, label="val_loss", marker="o", markersize=3)
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("DDPM Noise Prediction Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--p_drop", type=float, default=0.1)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ckpt_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    cfg = TrainConfig(
        T=args.T,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        p_drop=args.p_drop,
        schedule=args.schedule,
        num_workers=args.num_workers,
        ckpt_dir=args.ckpt_dir,
        seed=args.seed,
    )

    device = get_device()
    print_device_info(device)
    set_seed(cfg.seed)
    ensure_dir(cfg.ckpt_dir)
    ensure_dir("outputs/figures")

    train_ds, val_ds = make_datasets()
    train_loader, val_loader = make_loaders(train_ds, val_ds, cfg.batch_size, cfg.num_workers)

    model = SimpleEpsModel(num_classes=cfg.num_classes + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    betas = make_linear_beta_schedule(cfg.T)
    ddpm_terms = precompute_ddpm_terms(betas, device=device)

    meta = {
        "course": "University of Arizona, ECE 636 Information Theory, Project 1, Spring 2026",
        "T": cfg.T,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "p_drop": cfg.p_drop,
        "schedule": cfg.schedule,
        "device": str(device),
    }
    save_json(os.path.join(cfg.ckpt_dir, "run_meta.json"), meta)

    global_step = 0
    train_losses = []
    val_losses = []
    for epoch in range(1, cfg.epochs + 1):
        train_loss, global_step = train_one_epoch(model, optimizer, train_loader, ddpm_terms, cfg, device, global_step)
        val_loss = eval_one_epoch(model, val_loader, ddpm_terms, cfg, device)
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        print(f"epoch {epoch:03d} train_loss {train_loss:.6f} val_loss {val_loss:.6f}")

        ckpt_path = os.path.join(cfg.ckpt_dir, f"epoch_{epoch:03d}.pt")
        save_checkpoint(ckpt_path, model, optimizer, epoch=epoch, step=global_step, meta=meta)

        last_path = os.path.join(cfg.ckpt_dir, "last.pt")
        save_checkpoint(last_path, model, optimizer, epoch=epoch, step=global_step, meta=meta)

        history = {
            "epochs": list(range(1, epoch + 1)),
            "train_loss": train_losses,
            "val_loss": val_losses,
        }
        save_json(os.path.join(cfg.ckpt_dir, "loss_history.json"), history)
        save_loss_curve(train_losses, val_losses, os.path.join("outputs/figures", "fig_train_val_loss.png"))

    print("done. last checkpoint:", os.path.join(cfg.ckpt_dir, "last.pt"))

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
