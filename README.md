# Project 1: Conditional Diffusion Models on MNIST

University of Arizona  
ECE 636 Information Theory  
Project 1, Spring 2026

## Goal
Train a simple conditional diffusion model on MNIST. Given a label y in {0,1,2,3,4,5,6,7,8,9}, the model generates that digit. The model uses classifier-free guidance during sampling.

## Setup
Activate the conda environment:
conda activate cpm

Install dependencies:
pip install -r requirements.txt

Check Apple GPU support:
python -c "import torch; print('mps:', torch.backends.mps.is_available())"

## Method
Forward process:
q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)

Reparameterization:
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps, eps ~ N(0, I)

Training objective:
L = E || eps - eps_theta(x_t, t, y') ||^2

Classifier-free guidance:
eps_hat = (1 + w) eps_theta(x_t, t, y) - w eps_theta(x_t, t, null)

## Train
This trains and saves checkpoints in outputs/checkpoints.

python -m src.train --epochs 30 --batch_size 128 --lr 1e-4 --p_drop 0.1 --T 1000 --num_workers 2

## Evaluate last checkpoint
This loads the latest checkpoint and reports validation loss.

python -m src.eval --batch_size 256 --T 1000 --num_workers 2

## Sample
This loads the latest checkpoint by default and saves figures in outputs/figures.

Conditional grid for labels 0 to 9:
python -m src.sample --mode cond_grid --w 3.0 --samples_per_class 8

Guidance grid for a fixed label with multiple w values:
python -m src.sample --mode cfg_grid --label 3 --w_list 0 1 3 5 --n 12

## Debugging poor samples
If figures look like random strokes, run these checks:

1. Ensure you are sampling from the checkpoint you just trained:
python -m src.sample --mode cond_grid --ckpt_path outputs/checkpoints/last.pt --w 1.0 --samples_per_class 8

2. Run evaluation with timestep bins:
python -m src.eval --ckpt_path outputs/checkpoints/last.pt --t_bins 5

3. Sweep guidance to inspect CFG behavior:
python -m src.sample --mode cfg_grid --ckpt_path outputs/checkpoints/last.pt --label 3 --w_list 0 0.5 1 2 3 --n 12

4. If you changed model code, retrain from scratch so old checkpoints do not mix with new architecture.

## Outputs
Checkpoints:
outputs/checkpoints/last.pt
outputs/checkpoints/epoch_001.pt and so on

Figures:
outputs/figures/fig_cond_grid.png
outputs/figures/fig_cfg_grid.png
outputs/figures/fig_train_val_loss.png

Training history:
outputs/checkpoints/loss_history.json
