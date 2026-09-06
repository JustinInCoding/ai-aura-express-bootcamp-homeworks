"""汇总 runs/ 下的训练结果：打印/保存对比表，绘制准确率与 loss 曲线图。

用法：
    python compare.py                                      # 汇总全部 run
    python compare.py baseline-alexnet baseline-vgg16 ...  # 只对比指定 tag
    python compare.py resnet18-lr001 resnet18-noaug --title "ResNet18 调参" --out ablation

输出：runs/curves-<slug>.png（测试准确率）、runs/losses-<slug>.png（测试 loss）、
runs/compare-<slug>.md（markdown 表格片段），同时把表格打印到终端。
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# macOS 自带的中文字体，避免标题/图例变成方框
plt.rcParams["font.sans-serif"] = [
    "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 下面这套颜色 / 墨色来自一套经过无障碍校验的图表配色（浅色模式），
# 类别色按固定顺序取用，不做循环
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
SURFACE = "#fcfcfb"   # 图表底色
INK = "#0b0b0b"       # 标题
INK2 = "#52514e"      # 正文 / 图例文字
MUTED = "#898781"     # 坐标轴刻度
GRID = "#e1e0d9"      # 网格线
SPINE = "#c3c2b7"     # 坐标轴线


def parse_args():
    parser = argparse.ArgumentParser(description="汇总训练结果")
    parser.add_argument("tags", nargs="*", help="要对比的 run tag，默认全部")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--title", default="CIFAR-10 模型对比")
    parser.add_argument("--out", default="baseline", help="输出文件名的前缀")
    return parser.parse_args()


def load_runs(out_dir, tags):
    paths = sorted(out_dir.glob("*.json")) if not tags else [
        out_dir / f"{t}.json" for t in tags]
    runs = []
    for path in paths:
        if not path.exists():
            print(f"!! 找不到 {path}，跳过")
            continue
        runs.append(json.loads(path.read_text()))
    if not runs:
        raise SystemExit("没有可汇总的结果")
    return runs


def fmt_config(cfg):
    parts = [cfg["optimizer"], f"lr={cfg['lr']:g}"]
    if cfg["scheduler"] != "none":
        parts.append("cosine")
    if cfg.get("no_aug"):
        parts.append("无增强")
    return " ".join(parts)


def build_table(runs):
    rows = []
    for run in sorted(runs, key=lambda r: -r["best_acc"]):
        hist = run["history"]
        last = hist[-1]
        rows.append({
            "tag": run["tag"],
            "model": run["model"],
            "params": f"{run['params'] / 1e6:.2f}M",
            "config": fmt_config(run["config"]),
            "best_acc": f"{run['best_acc']:.4f} (@{run['best_epoch']})",
            "final_acc": f"{last['test_acc']:.4f}",
            "final_train_acc": f"{last['train_acc']:.4f}",
            "sec_per_epoch": f"{sum(h['sec'] for h in hist) / len(hist):.1f}s",
            "total_min": f"{run['total_sec'] / 60:.1f}",
        })
    return rows


def print_table(rows):
    header = ["tag", "model", "params", "config", "best_acc", "final_acc",
              "train_acc", "s/epoch", "total"]
    keys = ["tag", "model", "params", "config", "best_acc", "final_acc",
            "final_train_acc", "sec_per_epoch", "total_min"]
    widths = [max(len(header[i]), *(len(str(r[k])) for r in rows))
              for i, k in enumerate(keys)]

    def fmt(values):
        return "  ".join(str(v).ljust(w) for v, w in zip(values, widths))

    print(fmt(header))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        print(fmt([r[k] for k in keys]))


def save_markdown(rows, path):
    header = ("| run | 模型 | 参数量 | 配置 | 最优测试精度 | 末轮测试精度 | "
              "末轮训练精度 | 秒/epoch | 总耗时(min) |\n"
              "|---|---|---|---|---|---|---|---|---|\n")
    lines = [header]
    for r in rows:
        lines.append(f"| {r['tag']} | {r['model']} | {r['params']} | {r['config']} "
                     f"| {r['best_acc']} | {r['final_acc']} | {r['final_train_acc']} "
                     f"| {r['sec_per_epoch']} | {r['total_min']} |\n")
    path.write_text("".join(lines))


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def _non_overlapping(values, min_gap):
    """把线尾标签的 y 值从上到下推开至少 min_gap，避免互相压住。"""
    ordered = sorted(values, key=lambda v: -v[1])
    ys = []
    for _, y in ordered:
        if ys and ys[-1] - y < min_gap:
            y = ys[-1] - min_gap
        ys.append(y)
    return {name: y for (name, _), y in zip(ordered, ys)}


def plot_metric(runs, key, ylabel, out_png, title, legend_loc="lower right"):
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    style_axes(ax)

    finals = []
    for i, run in enumerate(runs):
        hist = run["history"]
        xs = [h["epoch"] for h in hist]
        ys = [h[key] for h in hist]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(xs, ys, color=color, linewidth=2, label=run["tag"])
        finals.append((run["tag"], ys[-1]))

    # 图例 + 线尾直接标注（浅色下黄色/青色对比度不足，靠文字标签兜底）
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8, loc=legend_loc)
    x_last = runs[0]["history"][-1]["epoch"]
    # 标签最小间距按 y 轴高度的 6% 估（约等于 8pt 文字的高度），避免互相压住
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    label_ys = _non_overlapping(finals, min_gap=y_span * 0.06)
    for tag, _ in finals:
        ax.text(x_last + 0.5, label_ys[tag], tag, fontsize=8, color=INK2,
                va="center")
    ax.margins(x=0.18)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {out_png}")


def main():
    args = parse_args()
    runs = load_runs(args.out_dir, args.tags)
    rows = build_table(runs)
    print_table(rows)
    save_markdown(rows, args.out_dir / f"compare-{args.out}.md")

    plot_metric(runs, "test_acc", "test accuracy",
                args.out_dir / f"curves-{args.out}.png",
                f"{args.title} — 测试集准确率")
    plot_metric(runs, "test_loss", "test loss",
                args.out_dir / f"losses-{args.out}.png",
                f"{args.title} — 测试集损失", legend_loc="upper right")


if __name__ == "__main__":
    main()
