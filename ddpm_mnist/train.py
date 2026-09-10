import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import SmallUNet
from diffusion import Diffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--ckpt-dir", default="results/checkpoints")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=False, drop_last=True)

    model = SmallUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    diffusion = Diffusion(T=args.T, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    start_epoch = 0
    all_losses = []
    latest_ckpt = os.path.join(args.ckpt_dir, "latest.pth")
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        all_losses = ckpt["all_losses"]
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        for x0, _ in pbar:
            x0 = x0.to(device)
            B = x0.size(0)
            t = torch.randint(0, args.T, (B,), device=device)
            noise = torch.randn_like(x0)
            x_t = diffusion.q_sample(x0, t, noise)
            pred_noise = model(x_t, t)
            loss = criterion(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{epoch_loss/n_batches:.4f}")
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        all_losses.append(avg_loss)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1} done | avg_loss={avg_loss:.4f} | time={elapsed:.1f}s")

        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "all_losses": all_losses,
            "args": vars(args),
        }
        torch.save(ckpt, latest_ckpt)
        torch.save(ckpt, os.path.join(args.ckpt_dir, f"epoch_{epoch+1}.pth"))

        # Save loss curve every epoch
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(all_losses)+1), all_losses, "b-o", markersize=3)
        plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.title("Training Loss Curve")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=150)
        plt.close()

    print("Training complete.")


if __name__ == "__main__":
    main()
