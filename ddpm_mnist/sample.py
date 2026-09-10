import os
import time
import argparse
import torch
from torchvision.utils import save_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import SmallUNet
from diffusion import Diffusion


def denormalize(x):
    return (x + 1.0) / 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="results/checkpoints/latest.pth")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = SmallUNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    T = ckpt["args"]["T"]
    diffusion = Diffusion(T=T, device=device)
    os.makedirs(args.out_dir, exist_ok=True)

    # Generate 8x8 grid with full 1000-step DDPM
    print("Generating 8x8 sample grid (1000 steps)...")
    t0 = time.time()
    samples = diffusion.p_sample_loop(model, args.n, steps=1000, verbose=True)
    t1000 = time.time() - t0
    save_image(denormalize(samples), os.path.join(args.out_dir, "samples_8x8.png"), nrow=8)
    print(f"  1000 steps: {t1000:.2f}s")

    # Step comparison: 1000 / 200 / 50
    results = {}
    for steps in [1000, 200, 50]:
        print(f"Sampling with {steps} steps...")
        t0 = time.time()
        s = diffusion.p_sample_loop(model, args.n, steps=steps, verbose=True)
        elapsed = time.time() - t0
        results[steps] = {"samples": s, "time": elapsed}
        save_image(denormalize(s), os.path.join(args.out_dir, f"samples_{steps}.png"), nrow=8)
        print(f"  {steps} steps: {elapsed:.2f}s")

    # DDIM bonus: 50 steps deterministic
    print("DDIM sampling (50 steps, eta=0)...")
    t0 = time.time()
    s_ddim = diffusion.ddim_sample_loop(model, args.n, steps=50, eta=0.0, verbose=True)
    t_ddim = time.time() - t0
    save_image(denormalize(s_ddim), os.path.join(args.out_dir, "samples_ddim50.png"), nrow=8)
    print(f"  DDIM 50 steps: {t_ddim:.2f}s")

    # Create step comparison figure (4 rows: 1000/200/50 DDPM + DDIM 50)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ["DDPM 1000 steps", "DDPM 200 steps", "DDPM 50 steps", "DDIM 50 steps"]
    imgs = [results[1000]["samples"], results[200]["samples"], results[50]["samples"], s_ddim]
    for ax, img, title in zip(axes, imgs, titles):
        grid = img[:16]  # show first 16 for compactness
        grid = denormalize(grid)
        # Manual grid layout 4x4
        grid_np = grid.squeeze(1).numpy()
        canvas = np.zeros((4*28+3*2, 4*28+3*2))
        for i in range(16):
            r, c = divmod(i, 4)
            canvas[r*(28+2):r*(28+2)+28, c*(28+2):c*(28+2)+28] = grid_np[i]
        ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.suptitle("Sampling Steps Comparison", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "step_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Save timing results
    with open(os.path.join(args.out_dir, "timing.txt"), "w") as f:
        f.write(f"DDPM 1000 steps: {t1000:.2f}s\n")
        for steps in [200, 50]:
            f.write(f"DDPM {steps} steps: {results[steps]['time']:.2f}s\n")
        f.write(f"DDIM 50 steps: {t_ddim:.2f}s\n")

    # Forward noising visualization
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    x0 = ds[0][0].unsqueeze(0).to(device)
    timesteps = [0, 100, 300, 500, 700, 999]
    fig, axes = plt.subplots(1, len(timesteps), figsize=(15, 2.5))
    for ax, t_val in zip(axes, timesteps):
        t = torch.tensor([t_val], device=device)
        noise = torch.randn_like(x0)
        x_t = diffusion.q_sample(x0, t, noise)
        ax.imshow(denormalize(x_t).squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t={t_val}")
        ax.axis("off")
    plt.suptitle("Forward Noising Process q(x_t|x_0)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "forward_noising.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("All sampling done.")


import numpy as np


if __name__ == "__main__":
    main()
