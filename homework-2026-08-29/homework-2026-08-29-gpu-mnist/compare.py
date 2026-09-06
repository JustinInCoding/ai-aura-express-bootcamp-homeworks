"""汇总对比：读 results/{tag}_*.json，输出对比表格 + 图表。

分进程训练后每个模型的结果各自落一个 JSON，需要这一步统一汇总。
同时生成 markdown 表格，直接贴进 README。
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境，直接存文件，不要弹窗
import matplotlib.pyplot as plt

from models import MODELS

ORDER = list(MODELS)  # 按提出年份排序展示：AlexNet -> VGG -> GoogLeNet -> ResNet


def setup_cjk_font():
    """让图里的中文能正常显示，否则 matplotlib 会画成一串豆腐块方框。

    Ubuntu 服务器镜像默认不带任何中文字体，CPU 版里那几个（PingFang HK 等）
    是 macOS 专有的，在这里一个都找不到。先列 Linux 常见的，macOS 的留在
    后面做兜底，方便同一份脚本两边都能用。缺字体时装法见 setup.sh。
    """
    from matplotlib import font_manager as fm
    installed = {f.name for f in fm.fontManager.ttflist}
    candidates = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans CN",   # fonts-noto-cjk
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",                     # fonts-wqy-*
        "SimHei", "Microsoft YaHei",
        "PingFang HK", "Heiti TC", "Songti SC", "Arial Unicode MS",     # macOS 兜底
    ]
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            # 中文字体普遍缺少 U+2212（真正的减号），不改的话负号会渲染成方框
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None


def load(tag, out):
    results = []
    for name in ORDER:
        p = Path(out) / f"{tag}_{name}.json"
        if p.exists():
            results.append(json.load(open(p)))
        else:
            print(f"[跳过] 缺少 {p}")
    if not results:
        raise SystemExit(f"{out} 下没有找到任何 {tag}_*.json")
    return results


def print_table(results):
    print(f"\n{'=' * 78}")
    print("四网络对比")
    print(f"{'=' * 78}")
    head = f"{'model':<12}{'params':>10}{'best_acc':>11}{'final_acc':>11}{'s/epoch':>10}{'total':>9}"
    print(head)
    print("-" * 78)
    for r in results:
        print(f"{r['model']:<12}{r['params'] / 1e6:>9.2f}M{r['best_test_acc']:>10.2f}%"
              f"{r['final_test_acc']:>10.2f}%{r['sec_per_epoch']:>9.1f}s{r['total_min']:>8.1f}m")

    print("\n各类别准确率（取自最佳 epoch）")
    print(f"{'model':<12}" + "".join(f"{d:>7}" for d in range(10)))
    print("-" * 82)
    for r in results:
        print(f"{r['model']:<12}" + "".join(f"{a:>6.1f}%" for a in r["best_per_class_acc"]))
    # 最难的类别对所有模型是否一致，能说明这是数据本身的难度而非某个架构的弱点
    worst = [(r["model"], min(range(10), key=lambda c: r["best_per_class_acc"][c]))
             for r in results]
    print("\n各模型最难类别：" + "  ".join(f"{m}→{c}" for m, c in worst))


def write_markdown(results, path):
    lines = ["| 模型 | 参数量 | 最佳测试准确率 | 末轮准确率 | 单 epoch 耗时 | 总耗时 |",
             "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['params'] / 1e6:.2f}M | "
                     f"**{r['best_test_acc']:.2f}%** | {r['final_test_acc']:.2f}% | "
                     f"{r['sec_per_epoch']:.0f}s | {r['total_min']:.1f} 分 |")
    lines += ["", "| 模型 | " + " | ".join(str(d) for d in range(10)) + " |",
              "|---" * 11 + "|"]
    for r in results:
        lines.append(f"| {r['model']} | "
                     + " | ".join(f"{a:.1f}" for a in r["best_per_class_acc"]) + " |")
    path.write_text("\n".join(lines) + "\n")
    print(f"\nMarkdown 表格已写入 {path}")


def plot_curves(results, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in results:
        ep = [h["epoch"] for h in r["history"]]
        ax1.plot(ep, [h["test_acc"] for h in r["history"]], marker="o", label=r["model"])
        ax2.plot(ep, [h["train_loss"] for h in r["history"]], marker="o", label=r["model"])

    ax1.set(xlabel="epoch", ylabel="测试准确率 (%)", title="测试准确率收敛曲线")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.set(xlabel="epoch", ylabel="训练损失", title="训练损失下降曲线", yscale="log")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"图已保存 {path}")


def plot_tradeoff(results, path):
    """准确率 / 参数量 / 耗时三者的权衡 —— 作业结论主要靠这张图说话。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    names = [r["model"] for r in results]
    colors = plt.cm.tab10.colors[:len(names)]

    for ax, vals, title, unit in [
        (axes[0], [r["best_test_acc"] for r in results], "最佳测试准确率", "%"),
        (axes[1], [r["params"] / 1e6 for r in results], "参数量", "M"),
        (axes[2], [r["sec_per_epoch"] for r in results], "单 epoch 耗时", "s"),
    ]:
        bars = ax.bar(names, vals, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        # 准确率都在 98%+，从 0 起画会看不出差异，故把 y 轴收紧到数据范围
        if unit == "%":
            lo = min(vals)
            ax.set_ylim(lo - (max(vals) - lo + 0.1) * 1.5, max(vals) + 0.15)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}{unit}",
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"图已保存 {path}")


def plot_per_class(results, path):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    width = 0.8 / len(results)
    for i, r in enumerate(results):
        xs = [c + i * width - 0.4 + width / 2 for c in range(10)]
        ax.bar(xs, r["best_per_class_acc"], width, label=r["model"])
    lo = min(min(r["best_per_class_acc"]) for r in results)
    ax.set(xlabel="数字类别", ylabel="准确率 (%)", title="各类别准确率对比",
           xticks=range(10), ylim=(lo - 1, 100.4))
    ax.grid(alpha=0.3, axis="y")
    ax.legend(ncol=len(results))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"图已保存 {path}")


def main():
    p = argparse.ArgumentParser(description="汇总四网络对比结果并出图")
    p.add_argument("--tag", default="baseline")
    p.add_argument("--out", default="./results")
    p.add_argument("--figdir", default="./results/figs")
    args = p.parse_args()

    font = setup_cjk_font()
    if not font:
        print("[警告] 没找到中文字体，图中中文可能显示为方框")

    results = load(args.tag, args.out)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    print_table(results)
    write_markdown(results, Path(args.out) / f"{args.tag}_table.md")
    plot_curves(results, figdir / f"{args.tag}_curves.png")
    plot_tradeoff(results, figdir / f"{args.tag}_tradeoff.png")
    plot_per_class(results, figdir / f"{args.tag}_per_class.png")


if __name__ == "__main__":
    main()
