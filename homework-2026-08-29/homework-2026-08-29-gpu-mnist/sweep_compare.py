"""超参数单变量对照汇总：读 results/{tag}_{model}.json，输出对比表 + 图。

与 compare.py 的区别：compare.py 横向比"不同架构"，这里纵向比"同一架构下
不同超参数"。基准是 baseline_{model}.json。

"改动"一列不是写死的文字，而是把每组的 hparams 与基准逐项 diff 出来的 ——
这样改 sweep.sh 里的配置后，这里不用同步修改，也不会出现描述与实际不符。
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare import setup_cjk_font

# 顺序即展示顺序：基准在最前，便于逐行对比
SWEEP_TAGS = ["baseline", "lr001", "lr1", "adam", "bs256", "cosine", "augment"]

# 这些字段属于运行环境，不是要研究的超参数，diff 时忽略。
# GPU 版比 CPU 版多了 device / cudnn_benchmark，少了 threads（CPU 专有）。
IGNORE = {"tag", "out", "model", "resume", "limit_batches",
          "num_workers", "seed", "epochs",
          "device", "cudnn_benchmark"}


def load(tags, model, out, base_tag="baseline"):
    results = []
    for tag in tags:
        p = Path(out) / f"{tag}_{model}.json"
        if p.exists():
            r = json.load(open(p))
            r["tag"] = tag
            results.append(r)
        else:
            print(f"[跳过] 缺少 {p}")
    if not results:
        raise SystemExit(f"{out} 下没找到任何 *_{model}.json")
    base = next((r for r in results if r["tag"] == base_tag), None)
    if base is None:
        raise SystemExit(f"缺少 {base_tag} 结果，无法做对照。")
    # 把基准挪到第一项，这样后续逻辑不变
    results.remove(base)
    results.insert(0, base)
    return results


def describe(r, base):
    """把这一组相对基准改了什么，压成一句话。"""
    diff = [f"{k}={v}" for k, v in r["hparams"].items()
            if k not in IGNORE and base["hparams"].get(k) != v]
    return ", ".join(diff) if diff else "基准"


def print_table(results):
    base = results[0]
    print(f"\n{'=' * 88}")
    print(f"超参数单变量对照（{base['model']}, {base['hparams']['epochs']} epochs）")
    print(f"{'=' * 88}")
    print(f"{'tag':<10}{'改动':<34}{'best':>9}{'Δbest':>8}{'final':>9}{'s/epoch':>10}")
    print("-" * 88)
    for r in results:
        d = r["best_test_acc"] - base["best_test_acc"]
        # 基准自己那行不显示 Δ，否则一个 +0.00 容易被误读成"某个改动没效果"
        delta = "—" if r is base else f"{d:+.2f}"
        print(f"{r['tag']:<10}{describe(r, base):<34}{r['best_test_acc']:>8.2f}%"
              f"{delta:>8}{r['final_test_acc']:>8.2f}%{r['sec_per_epoch']:>9.1f}s")

    best = max(results, key=lambda r: r["best_test_acc"])
    print(f"\n最佳配置：{best['tag']}（{describe(best, base)}）"
          f"  {best['best_test_acc']:.2f}%")
    # 训练末尾 train_acc 与 test_acc 的差距是过拟合的直接指标
    print("\n过拟合观察（末轮 train_acc - test_acc，越大越过拟合）")
    for r in results:
        h = r["history"][-1]
        print(f"  {r['tag']:<10}train {h['train_acc']:>6.2f}%  test {h['test_acc']:>6.2f}%"
              f"  gap {h['train_acc'] - h['test_acc']:+.2f}")


def write_markdown(results, path):
    base = results[0]
    lines = ["| 配置 | 相对基准的改动 | 最佳准确率 | 相对基准 | 末轮准确率 | 单 epoch 耗时 |",
             "|---|---|---|---|---|---|"]
    for r in results:
        d = r["best_test_acc"] - base["best_test_acc"]
        delta = "—" if r is base else f"{d:+.2f}"
        lines.append(f"| `{r['tag']}` | {describe(r, base)} | "
                     f"**{r['best_test_acc']:.2f}%** | {delta} | "
                     f"{r['final_test_acc']:.2f}% | {r['sec_per_epoch']:.0f}s |")
    path.write_text("\n".join(lines) + "\n")
    print(f"\nMarkdown 表格已写入 {path}")


def plot_curves(results, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in results:
        ep = [h["epoch"] for h in r["history"]]
        # 基准画粗黑线做参照，其余用默认色循环
        style = dict(color="black", lw=2.5) if r is results[0] else {}
        ax1.plot(ep, [h["test_acc"] for h in r["history"]], marker="o",
                 label=r["tag"], **style)
        ax2.plot(ep, [h["train_loss"] for h in r["history"]], marker="o",
                 label=r["tag"], **style)

    ax1.set(xlabel="epoch", ylabel="测试准确率 (%)", title="不同超参数下的收敛曲线")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    ax2.set(xlabel="epoch", ylabel="训练损失", title="训练损失（对数轴）", yscale="log")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"图已保存 {path}")


def plot_delta(results, path):
    """相对基准的准确率变化 —— 一眼看出哪个超参数真的有影响。"""
    base = results[0]
    others = [r for r in results if r is not base]
    if not others:
        return
    tags = [r["tag"] for r in others]
    deltas = [r["best_test_acc"] - base["best_test_acc"] for r in others]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(tags, deltas, color=["#2a9d8f" if d >= 0 else "#e76f51" for d in deltas])
    ax.axhline(0, color="black", lw=1)
    ax.set(ylabel="准确率变化 (百分点)",
           title=f"相对基准的准确率变化（基准 {base['best_test_acc']:.2f}%）")
    ax.grid(alpha=0.3, axis="y")
    for i, d in enumerate(deltas):
        # 正值标在柱顶、负值标在柱底，避免文字压在柱子上
        ax.text(i, d, f"{d:+.2f}", ha="center",
                va="bottom" if d >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"图已保存 {path}")


def main():
    p = argparse.ArgumentParser(description="汇总超参数单变量对照结果")
    p.add_argument("--model", default="vgg")
    p.add_argument("--tags", nargs="*", default=SWEEP_TAGS)
    p.add_argument("--base", default="baseline", help="基准实验的 tag（默认 baseline）")
    p.add_argument("--prefix", default="sweep", help="输出文件名前缀（默认 sweep）")
    p.add_argument("--out", default="./results")
    p.add_argument("--figdir", default="./results/figs")
    args = p.parse_args()

    if not setup_cjk_font():
        print("[警告] 没找到中文字体，图中中文可能显示为方框")

    results = load(args.tags, args.model, args.out, base_tag=args.base)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    print_table(results)
    write_markdown(results, Path(args.out) / f"{args.prefix}_{args.model}_table.md")
    plot_curves(results, figdir / f"{args.prefix}_{args.model}_curves.png")
    plot_delta(results, figdir / f"{args.prefix}_{args.model}_delta.png")


if __name__ == "__main__":
    main()
