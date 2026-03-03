import argparse
import os

import torch
import matplotlib.pyplot as plt

from .config import TrainConfig
from .utils import get_device, print_device_info, ensure_dir, find_latest_checkpoint, load_checkpoint
from .model import SimpleEpsModel
from .diffusion import make_linear_beta_schedule, precompute_ddpm_terms, sample_loop

def to_0_1(x):
    return (x + 1.0) / 2.0


def save_labeled_grid(imgs: torch.Tensor, row_labels, n_cols: int, out_path: str, title: str) -> None:
    imgs = to_0_1(imgs).cpu()
    n_rows = len(row_labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.1, n_rows * 1.1))
    if n_rows == 1:
        axes = [axes]
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            ax = axes[r][c] if n_cols > 1 else axes[r]
            ax.imshow(imgs[idx, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(str(row_labels[r]), rotation=0, labelpad=12, fontsize=9, va="center")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def make_conditional_grid(model, ddpm_terms, cfg: TrainConfig, device, w: float, samples_per_class: int, out_path: str):
    imgs = []
    for y in range(cfg.num_classes):
        x = sample_loop(model, ddpm_terms, device, cfg.T, label=y, n=samples_per_class, w=w, img_size=cfg.img_size, null_label=cfg.null_label)
        imgs.append(x.cpu())
    imgs = torch.cat(imgs, dim=0)
    row_labels = [f"y={y}" for y in range(cfg.num_classes)]
    save_labeled_grid(
        imgs,
        row_labels=row_labels,
        n_cols=samples_per_class,
        out_path=out_path,
        title=f"Conditional samples by label, w={w}",
    )
    print("saved:", out_path)

@torch.no_grad()
def make_cfg_grid(model, ddpm_terms, cfg: TrainConfig, device, label: int, w_list, n: int, out_path: str):
    rows = []
    for w in w_list:
        x = sample_loop(model, ddpm_terms, device, cfg.T, label=label, n=n, w=float(w), img_size=cfg.img_size, null_label=cfg.null_label)
        rows.append(x.cpu())
    imgs = torch.cat(rows, dim=0)
    row_labels = [f"w={float(w):g}" for w in w_list]
    save_labeled_grid(
        imgs,
        row_labels=row_labels,
        n_cols=n,
        out_path=out_path,
        title=f"CFG sweep for label y={label}",
    )
    print("saved:", out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["cond_grid", "cfg_grid"], default="cond_grid")
    parser.add_argument("--ckpt_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--w", type=float, default=3.0)
    parser.add_argument("--samples_per_class", type=int, default=8)
    parser.add_argument("--label", type=int, default=3)
    parser.add_argument("--w_list", type=float, nargs="*", default=[0.0, 1.0, 3.0, 5.0])
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--out_dir", type=str, default="outputs/figures")
    args = parser.parse_args()

    cfg = TrainConfig(T=args.T, figures_dir=args.out_dir)

    device = get_device()
    print_device_info(device)
    ensure_dir(cfg.figures_dir)

    ckpt_path = args.ckpt_path.strip()
    if not ckpt_path:
        ckpt_path = find_latest_checkpoint(args.ckpt_dir)
    if not ckpt_path:
        raise FileNotFoundError("No checkpoint found. Train first or pass --ckpt_path.")

    model = SimpleEpsModel(num_classes=cfg.num_classes + 1).to(device)
    try:
        load_checkpoint(ckpt_path, model, optimizer=None, map_location=device)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture mismatch. Retrain from scratch with the current model "
            "or pass a compatible --ckpt_path."
        ) from exc
    model.eval()

    betas = make_linear_beta_schedule(cfg.T)
    ddpm_terms = precompute_ddpm_terms(betas, device=device)

    if args.mode == "cond_grid":
        out_path = os.path.join(cfg.figures_dir, "fig_cond_grid.png")
        make_conditional_grid(model, ddpm_terms, cfg, device, w=args.w, samples_per_class=args.samples_per_class, out_path=out_path)
    else:
        out_path = os.path.join(cfg.figures_dir, "fig_cfg_grid.png")
        make_cfg_grid(model, ddpm_terms, cfg, device, label=args.label, w_list=args.w_list, n=args.n, out_path=out_path)

    print("checkpoint:", ckpt_path)

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
