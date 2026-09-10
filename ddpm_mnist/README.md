# MNIST DDPM — 从零训练迷你扩散模型

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

从零实现 DDPM（Denoising Diffusion Probabilistic Model），在 MNIST 上训练小型 U-Net 生成手写数字，支持 DDPM / DDIM 双通道采样与多步数对比。

---

## 示例结果

### 8×8 生成样本（DDPM 1000 步）
![samples](results/samples_8x8.png)

### 采样步数对比
![comparison](results/step_comparison.png)

### 训练损失曲线
![loss](results/loss_curve.png)

---

## 方法简述

| 模块 | 说明 |
|------|------|
| **前向加噪** | 线性噪声调度 β ∈ [1e-4, 0.02]，T = 1000；一步采样 q(x_t\|x_0) = N(√ᾱ_t·x_0, (1−ᾱ_t)I) |
| **去噪网络** | 小型 U-Net（~1.69M 参数），3 级下采样 + 自注意力瓶颈层 + 正弦时间嵌入 |
| **反向采样** | Ancestral Sampling（DDPM η=1）与确定性采样（DDIM η=0） |
| **训练目标** | MSE：预测噪声 ε_θ(x_t, t) ≈ ε |

## 环境要求

- Python ≥ 3.10
- PyTorch ≥ 2.0（CPU 或 GPU 均可）
- torchvision, matplotlib, tqdm, numpy

## 快速开始

```bash
cd ddpm_mnist
pip install -r requirements.txt

# 1. 训练（CPU 约 80 分钟，GPU 更快）
python train.py --epochs 8 --batch-size 128 --T 1000

# 2. 生成 8×8 样本网格 + 步数对比 + DDIM 加速
python sample.py --ckpt results/checkpoints/latest.pth --n 64

# 3. 生成 3 页 PDF 实验报告
python generate_report.py
```

## 命令行参数

### train.py
| 参数 | 默认 | 说明 |
|------|------|------|
| `--epochs` | 15 | 训练轮数 |
| `--batch-size` | 128 | 批大小 |
| `--lr` | 2e-4 | 学习率（AdamW） |
| `--T` | 1000 | 扩散总步数 |
| `--ckpt-dir` | `results/checkpoints` | 检查点保存目录 |

### sample.py
| 参数 | 默认 | 说明 |
|------|------|------|
| `--ckpt` | `results/checkpoints/latest.pth` | 模型权重路径 |
| `--n` | 64 | 生成样本数（8×8=64） |
| `--device` | cpu | 推理设备 |

## 项目结构

```
mnist-ddpm/
├── model.py              # 小型 U-Net（~1.69M 参数）
├── diffusion.py          # 噪声调度 / 前向加噪 / DDPM & DDIM 采样
├── train.py              # 训练（tqdm 进度条 + checkpoint）
├── sample.py             # 采样 + 步数对比 + DDIM 加速
├── generate_report.py    # 生成 3 页 PDF 报告
├── report.pdf            # 实验报告
├── requirements.txt      # 依赖
├── results/
│   ├── loss_curve.png
│   ├── samples_8x8.png
│   ├── step_comparison.png
│   ├── forward_noising.png
│   └── checkpoints/      # 训练权重（.pth）
└── README.md
```

## 采样性能对比

| 方法 | 步数 | 耗时 (s) | 质量 |
|------|:----:|:-------:|:----:|
| DDPM | 1000 | 256.16 | ★★★★★ |
| DDPM |  200 |  54.11 | ★★★★ |
| DDPM |   50 |  12.47 | ★★★ |
| **DDIM** | **50** | **12.34** | **★★★★★** |

> DDIM 50 步（η=0）在质量与速度间取得最佳平衡。

## 遇到的问题与解决

| 问题 | 解决方案 |
|------|----------|
| GroupNorm 通道数不整除（in_ch=1） | 使用 `min(8, in_ch)` 作为 group 数 |
| 非连续时间步后验方差错误 → 200/50 步退化为噪声 | 改用 DDIM (η=1) 广义后验分布 |
| CPU 训练慢 | 减少 epochs 至 8，启用 16 线程并行 |

## License

MIT
