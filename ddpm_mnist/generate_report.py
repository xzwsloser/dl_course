import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
import os

FONT_PATH = "/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc"
font_manager.fontManager.addfont(FONT_PATH)
prop = font_manager.FontProperties(fname=FONT_PATH)
plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

PAGE_W, PAGE_H = 8.27, 11.69  # A4
MARGIN = 0.9  # inches
CONTENT_W = PAGE_W - 2 * MARGIN


def new_page(pdf, title="", subtitle=""):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    y = 1.0 - 0.8 / PAGE_H
    if title:
        fig.text(MARGIN / PAGE_W, y, title, fontsize=16, fontweight="bold",
                 va="top", transform=fig.transFigure)
        y -= 0.35 / PAGE_H
    if subtitle:
        fig.text(MARGIN / PAGE_W, y, subtitle, fontsize=9, color="#555",
                 va="top", transform=fig.transFigure)
        y -= 0.3 / PAGE_H
    return fig, y


def section(fig, y, text):
    y -= 0.15 / PAGE_H
    fig.text(MARGIN / PAGE_W, y, text, fontsize=13, fontweight="bold",
             va="top", transform=fig.transFigure)
    y -= 0.08 / PAGE_H
    fig.add_artist(plt.Line2D([MARGIN / PAGE_W, 1 - MARGIN / PAGE_W],
                              [y, y], color="#cccccc", lw=0.5,
                              transform=fig.transFigure))
    y -= 0.1 / PAGE_H
    return y


def body(fig, y, lines, fontsize=10, spacing=0.016):
    for line in lines:
        fig.text(MARGIN / PAGE_W, y, line, fontsize=fontsize, va="top",
                 transform=fig.transFigure)
        y -= spacing
    return y


def image(fig, y, path, height_frac):
    """Place image centered. y is the top edge in figure coords (0-1)."""
    img = plt.imread(path)
    ih, iw = img.shape[:2]
    aspect = ih / iw
    w = CONTENT_W
    h = w * aspect
    max_h = PAGE_H * height_frac
    if h > max_h:
        h = max_h
        w = h / aspect
    h_frac = h / PAGE_H
    w_frac = w / PAGE_W
    x0 = (PAGE_W - w) / 2
    ax = fig.add_axes([x0 / PAGE_W, y - h_frac, w_frac, h_frac])
    ax.imshow(img)
    ax.axis("off")
    return y - h_frac - 0.15 / PAGE_H


def table(fig, y, headers, rows, col_widths=None, fontsize=10):
    if col_widths is None:
        col_widths = [CONTENT_W / len(headers)] * len(headers)
    x_start = MARGIN
    # header
    x = x_start
    for j, h in enumerate(headers):
        fig.text(x / PAGE_W, y, h, fontsize=fontsize, fontweight="bold",
                 va="top", transform=fig.transFigure)
        x += col_widths[j]
    y -= 0.18 / PAGE_H
    fig.add_artist(plt.Line2D([MARGIN / PAGE_W, 1 - MARGIN / PAGE_W],
                              [y, y], color="#999", lw=0.8,
                              transform=fig.transFigure))
    y -= 0.05 / PAGE_H
    for row in rows:
        x = x_start
        for j, cell in enumerate(row):
            fig.text(x / PAGE_W, y, cell, fontsize=fontsize, va="top",
                     transform=fig.transFigure)
            x += col_widths[j]
        y -= 0.18 / PAGE_H
    y -= 0.1 / PAGE_H
    return y


def main():
    results_dir = "results"
    pdf_path = "report.pdf"

    with PdfPages(pdf_path) as pdf:
        # ========== Page 1: Method & Setup ==========
        fig, y = new_page(pdf, "从零训练迷你扩散模型（MNIST DDPM）",
                          "深度学习课程实验")

        y = section(fig, y, "1  方法简述")
        y = body(fig, y, [
            "1.1  前向加噪过程",
            "   线性噪声调度：beta 从 1e-4 到 0.02，总步数 T = 1000。",
            "   一步加噪：q(x_t|x_0) = N( sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I )。",
            "",
            "1.2  去噪网络",
            "   小型 U-Net：3 级下采样 (32 -> 64 -> 128) + 自注意力瓶颈层 + 3 级上采样。",
            "   正弦时间嵌入 + GroupNorm + SiLU；参数量约 1.69M（满足 <= 5M 限制）。",
            "   训练目标：MSE 预测噪声 eps_theta(x_t, t) 约等于 eps。",
            "",
            "1.3  反向采样",
            "   DDPM Ancestral Sampling：从纯噪声逐步去噪至 x_0。",
            "   非连续时间步使用 DDIM (eta=1) 广义后验分布，保证较少步数下的质量。",
            "   DDIM 加速采样（选做加分）：确定性采样 (eta=0)，50 步生成高质量样本。",
        ], fontsize=10)

        y = section(fig, y, "2  实验设置")
        y = table(fig, y,
                  ["项目", "参数"],
                  [
                      ["数据集", "MNIST (60,000 张 28×28 灰度图)"],
                      ["训练轮数", "8 epochs"],
                      ["批大小", "128"],
                      ["优化器", "AdamW + CosineAnnealingLR"],
                      ["学习率", "2e-4"],
                      ["硬件", "CPU (20 核, 16 线程并行)"],
                      ["进度条 / Checkpoint", "tqdm / 每 epoch 保存 latest + epoch_N"],
                      ["训练耗时", "约 85 分钟"],
                  ],
                  col_widths=[0.25 * CONTENT_W, 0.75 * CONTENT_W])

        y = section(fig, y, "3  训练损失曲线")
        y = image(fig, y, os.path.join(results_dir, "loss_curve.png"), 0.22)
        pdf.savefig(fig)
        plt.close(fig)

        # ========== Page 2: Results ==========
        fig, y = new_page(pdf, "实验结果")

        y = section(fig, y, "4  前向加噪可视化")
        y = image(fig, y, os.path.join(results_dir, "forward_noising.png"), 0.18)
        y = body(fig, y, ["从 t=0 到 t=999，图像逐步被高斯噪声淹没，验证前向过程正确性。"])

        y = section(fig, y, "5  生成样本网格 (8×8, DDPM 1000 步)")
        y = image(fig, y, os.path.join(results_dir, "samples_8x8.png"), 0.35)
        y = body(fig, y, ["模型成功生成可辨识的 MNIST 手写数字，风格多样，无明显模式坍塌。"])
        pdf.savefig(fig)
        plt.close(fig)

        # ========== Page 3: Comparison & Discussion ==========
        fig, y = new_page(pdf, "步数对比分析与讨论")

        y = section(fig, y, "6  采样步数对比")
        y = image(fig, y, os.path.join(results_dir, "step_comparison.png"), 0.25)

        y = table(fig, y,
                  ["方法", "步数", "耗时(s)", "质量"],
                  [
                      ["DDPM", "1000", "256.16", "优秀"],
                      ["DDPM", "200", "54.11", "良好"],
                      ["DDPM", "50", "12.47", "一般"],
                      ["DDIM", "50", "12.34", "优秀"],
                  ],
                  col_widths=[0.25 * CONTENT_W, 0.15 * CONTENT_W,
                              0.25 * CONTENT_W, 0.35 * CONTENT_W])

        y = section(fig, y, "7  分析")
        y = body(fig, y, [
            "- 步数减少使采样耗时近似线性下降 (1000 -> 50 步约 20 倍加速)。",
            "- DDPM 200 步时质量有所下降，50 步时出现明显模糊。",
            "- DDIM 50 步 (eta=0) 凭借确定性映射，在极少步数下保持较高生成质量。",
            "- 综合来看，DDIM 50 步在质量与速度间取得最佳平衡。",
        ], fontsize=10)

        y = section(fig, y, "8  遇到的问题与解决")
        y = table(fig, y,
                  ["问题", "解决方案"],
                  [
                      ["GroupNorm 通道不整除 (in_ch=1)", "使用 min(8, in_ch) 作为 group 数"],
                      ["非连续时间步采样退化为噪声", "改用 DDIM (eta=1) 广义后验分布"],
                      ["CPU 训练慢", "减少 epochs 至 8，启用 16 线程并行"],
                  ],
                  col_widths=[0.55 * CONTENT_W, 0.45 * CONTENT_W])

        pdf.savefig(fig)
        plt.close(fig)

    print(f"Report saved to {pdf_path}")


if __name__ == "__main__":
    main()
