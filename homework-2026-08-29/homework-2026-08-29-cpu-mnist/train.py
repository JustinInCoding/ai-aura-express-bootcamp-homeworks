"""MNIST 四网络对比训练脚本。

用法见文件末尾 __main__ 的 --help，或 README。
所有超参数都通过命令行传入，便于后续做调参对比而无需改代码。
"""

import argparse
import csv
import faulthandler
import json
import os
import sys
import time
from pathlib import Path

# 必须在 import torch（会加载 libiomp5）之前设置，否则 OpenMP 运行时已初始化，改不动了。
# 本机实测 SIGSEGV 的根因是 OpenMP 并行区嵌套 fork 失控：inter-op 线程池派发算子，
# 算子内部再开 OMP 并行区，libiomp5 逐层 __kmp_fork_call 递归 500+ 层，撑爆 8MB 主线程栈。
# 限制活动层级为 1 = 只允许一层并行区，嵌套请求退化为串行执行。
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
os.environ.setdefault("KMP_MAX_ACTIVE_LEVELS", "1")

# 万一再崩，faulthandler 会把 Python 层调用栈打到 stderr，而不是只留一句 segmentation fault
faulthandler.enable()

import torch

# inter-op 默认 8（= 逻辑核数），与 4 线程的 intra-op 并行区叠乘正是嵌套的来源。
# MNIST 这种单分支模型本来就没有可并行的算子分支，置 1 不损失性能。
torch.set_num_interop_threads(1)
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import MODELS, build, count_params

# 本机实测的 MNIST 统计量（见 README 3.1）
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def get_loaders(args):
    """构建训练/测试 DataLoader。

    注意归一化只用训练集统计量，测试集沿用同一组值 —— 测试集的统计量
    属于"未来信息"，用它归一化是一种数据泄露。
    """
    norm = transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))

    train_tf = [transforms.ToTensor(), norm]
    if args.augment:
        # 手写数字的合理扰动：小幅旋转+平移。不做水平翻转，因为翻转后的数字不再是该类别
        train_tf = [transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                    transforms.ToTensor(), norm]

    train_set = datasets.MNIST(args.data, train=True, download=True,
                               transform=transforms.Compose(train_tf))
    test_set = datasets.MNIST(args.data, train=False, download=True,
                              transform=transforms.Compose([transforms.ToTensor(), norm]))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False,
                             num_workers=args.num_workers)
    return train_loader, test_loader


def make_optimizer(model, args):
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr,
                               momentum=args.momentum, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    raise ValueError(f"未知优化器 {args.optimizer}")


def make_scheduler(opt, args):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 3), gamma=0.1)
    raise ValueError(f"未知调度器 {args.scheduler}")


@torch.no_grad()          # 评估不建计算图，省内存也更快
def evaluate(model, loader):
    """返回 (整体准确率, 每类准确率列表)。"""
    model.eval()          # BN 改用滑动统计量、Dropout 关闭；调用方之后要记得切回 train()
    correct = torch.zeros(10)
    total = torch.zeros(10)
    for x, y in loader:
        pred = model(x).argmax(1)
        for c in range(10):
            mask = y == c
            total[c] += mask.sum()
            correct[c] += (pred[mask] == c).sum()
    per_class = (correct / total * 100).tolist()
    return (correct.sum() / total.sum() * 100).item(), per_class


def train_one(name, args, train_loader, test_loader):
    """训练单个模型，返回一份结果 dict（供 save_results / compare 使用）。"""
    torch.manual_seed(args.seed)          # 每个模型用同一 seed 起步，保证对比公平

    model = build(name)
    n_params = count_params(model)
    opt = make_optimizer(model, args)     # 必须在 model 之后建，要拿 model.parameters()
    sched = make_scheduler(opt, args)     # --scheduler none 时是 None，后面都要判空
    # CrossEntropyLoss 内部自带 softmax，所以各网络最后一层是裸 Linear，别再手动加
    lossf = nn.CrossEntropyLoss()

    # 本机 GoogLeNet 的 backward 会间发性触发 libiomp5 段错误（见 README 5.4）。
    # 每个 epoch 存一次断点，崩溃后最多只丢一个 epoch，而不是整个模型的训练成果。
    ckpt_path = Path(args.out) / f"ckpt_{args.tag}_{name}.pt"
    start_ep, history, best_acc, best_per_class, elapsed = 1, [], 0.0, None, 0.0

    if args.resume and ckpt_path.exists():
        # 下面 5 类状态缺一不可：少 opt 则 momentum 缓冲清零、少 sched 则 lr 跳回初始值
        ck = torch.load(ckpt_path)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if sched is not None and ck["sched"] is not None:
            sched.load_state_dict(ck["sched"])
        # 恢复 RNG 状态，否则续训后的数据 shuffle 顺序与不中断的一次跑不一致，结果不可复现
        torch.set_rng_state(ck["rng"])
        start_ep = ck["epoch"] + 1
        history, best_acc = ck["history"], ck["best_acc"]
        best_per_class, elapsed = ck["best_per_class"], ck["elapsed"]
        print(f"\n[续训] {name} 已完成 {ck['epoch']}/{args.epochs} epochs，"
              f"当前最佳 {best_acc:.2f}%，从第 {start_ep} epoch 继续")

    n_batches = args.limit_batches or len(train_loader)   # or 取第一个真值，仅用于表头
    print(f"\n{'=' * 72}")
    print(f"训练 {name}  |  参数量 {n_params / 1e6:.2f}M  |  "
          f"{args.epochs} epochs x {n_batches} batches")
    print(f"{'=' * 72}")
    print(f"{'epoch':<7}{'train_loss':>12}{'train_acc':>11}{'test_acc':>10}"
          f"{'lr':>10}{'time':>9}")
    print("-" * 72)

    for ep in range(start_ep, args.epochs + 1):
        # 不是多余的：上一轮结尾的 evaluate 把模型切成了 eval，不切回来 BN 的滑动
        # 统计量会停止更新、Dropout 会一直关闭 —— 训练悄悄退化且不报错
        model.train()
        t0 = time.perf_counter()
        run_loss = run_correct = run_seen = 0

        for i, (x, y) in enumerate(train_loader):
            if args.limit_batches and i >= args.limit_batches:
                break                     # 冒烟测试：跑够 N 个 batch 就撤
            # 下面五步是 PyTorch 训练的骨架，顺序不能动
            opt.zero_grad()               # backward 是累加到 p.grad 而非覆盖，必须手动清
            out = model(x)
            loss = lossf(out, y)
            loss.backward()               # 只算梯度
            opt.step()                    # 才更新权重（分开是为了能在中间插梯度裁剪等）

            # 乘 y.size(0)：loss 是 batch 均值，而 drop_last=False 时末尾 batch 更小，
            # 还原成总损失再除 run_seen 才是真正的样本级平均
            run_loss += loss.item() * y.size(0)
            run_correct += (out.argmax(1) == y).sum().item()   # .item() 避免留住计算图
            run_seen += y.size(0)

        cur_lr = opt.param_groups[0]["lr"]   # 先读：step 之后 lr 就是下一轮的值了
        if sched:
            sched.step()

        # train_acc 是边训边算的，同轮前面的 batch 用的是更差的权重，加上 augment 会
        # 扰动训练图 —— 所以 train_acc < test_acc 是正常现象，不是 bug
        train_loss = run_loss / run_seen
        train_acc = run_correct / run_seen * 100
        test_acc, per_class = evaluate(model, test_loader)
        dt = time.perf_counter() - t0
        elapsed += dt

        # 注意 best 是按测试集挑的，报出的数字略偏乐观；四个模型待遇一致故对比仍公平
        if test_acc > best_acc:
            best_acc, best_per_class = test_acc, per_class

        history.append({"epoch": ep, "train_loss": train_loss, "train_acc": train_acc,
                        "test_acc": test_acc, "lr": cur_lr, "sec": dt})
        # flush：长任务输出重定向到 logs/ 时，不 flush 进度会卡在缓冲区里看不见
        print(f"{ep:<7}{train_loss:>12.4f}{train_acc:>10.2f}%{test_acc:>9.2f}%"
              f"{cur_lr:>10.5f}{dt:>8.1f}s", flush=True)

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict() if sched else None,
                    "rng": torch.get_rng_state(), "epoch": ep, "history": history,
                    "best_acc": best_acc, "best_per_class": best_per_class,
                    "elapsed": elapsed}, ckpt_path)

    total_min = elapsed / 60
    print("-" * 72)
    print(f"{name} 完成：最佳测试准确率 {best_acc:.2f}%  |  总耗时 {total_min:.1f} 分钟")

    ckpt_path.unlink(missing_ok=True)   # 正常跑完就删掉，避免下次同 tag 实验误用旧断点

    return {
        "model": name,
        "params": n_params,
        # best 与 final 差距大 = 后期过拟合或震荡，compare.py 里值得看的信号
        "best_test_acc": best_acc,
        "final_test_acc": history[-1]["test_acc"],
        "best_per_class_acc": best_per_class,
        "total_min": total_min,
        "sec_per_epoch": total_min * 60 / args.epochs,
        "history": history,
        # 全量存参数（剔掉 data 这个本机路径），让每份 json 自带完整实验配置
        "hparams": {k: v for k, v in vars(args).items() if k != "data"},
    }


def save_results(results, args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for r in results:
        with open(out / f"{args.tag}_{r['model']}.json", "w") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)

    csv_path = out / "summary.csv"
    fields = ["tag", "model", "params", "epochs", "batch_size", "lr", "optimizer",
              "momentum", "weight_decay", "scheduler", "augment",
              "best_test_acc", "final_test_acc", "sec_per_epoch", "total_min"]
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for r in results:
            w.writerow({
                "tag": args.tag, "model": r["model"], "params": r["params"],
                "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                "optimizer": args.optimizer, "momentum": args.momentum,
                "weight_decay": args.weight_decay, "scheduler": args.scheduler,
                "augment": args.augment,
                "best_test_acc": round(r["best_test_acc"], 4),
                "final_test_acc": round(r["final_test_acc"], 4),
                "sec_per_epoch": round(r["sec_per_epoch"], 1),
                "total_min": round(r["total_min"], 2),
            })
    print(f"\n结果已保存：{out}/{args.tag}_*.json 与 {csv_path}")


def print_comparison(results):
    print(f"\n{'=' * 72}")
    print("四网络对比")
    print(f"{'=' * 72}")
    print(f"{'model':<12}{'params':>10}{'best_acc':>11}{'final_acc':>11}"
          f"{'s/epoch':>10}{'总耗时':>10}")
    print("-" * 72)
    for r in sorted(results, key=lambda r: -r["best_test_acc"]):
        print(f"{r['model']:<12}{r['params'] / 1e6:>9.2f}M{r['best_test_acc']:>10.2f}%"
              f"{r['final_test_acc']:>10.2f}%{r['sec_per_epoch']:>9.1f}s"
              f"{r['total_min']:>9.1f}m")

    print(f"\n{'各类别准确率（取自最佳 epoch）':<40}")
    print(f"{'model':<12}" + "".join(f"{d:>7}" for d in range(10)))
    print("-" * 82)
    for r in results:
        print(f"{r['model']:<12}" + "".join(f"{a:>6.1f}%" for a in r["best_per_class_acc"]))


def parse_args():
    p = argparse.ArgumentParser(
        description="MNIST 上对比 AlexNet / VGG / GoogLeNet / ResNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="all",
                   choices=list(MODELS) + ["all"], help="要训练的模型")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    p.add_argument("--momentum", type=float, default=0.9, help="仅 SGD 使用")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--scheduler", default="none", choices=["none", "cosine", "step"])
    p.add_argument("--augment", action="store_true", help="启用随机旋转+平移增强")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0,
                   help="macOS 上进程启动开销大，MNIST 建议保持 0")
    p.add_argument("--threads", type=int, default=0,
                   help="intra-op(OMP) 线程数，0=用默认(4)。调小可规避 OpenMP 段错误，代价是变慢")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="存在断点时自动续训。--no-resume 可强制从头开始")
    p.add_argument("--limit-batches", type=int, default=0,
                   help="每 epoch 只跑前 N 个 batch，0 表示全部。用于冒烟测试")
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="./results")
    p.add_argument("--tag", default="baseline", help="本次实验名，用于结果文件命名")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.threads:
        torch.set_num_threads(args.threads)

    print(f"torch {torch.__version__} | intra-op={torch.get_num_threads()} "
          f"inter-op={torch.get_num_interop_threads()} "
          f"| mkldnn={torch.backends.mkldnn.is_available()} | device=cpu")
    print(f"配置：{vars(args)}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = get_loaders(args)
    names = list(MODELS) if args.model == "all" else [args.model]

    results = [train_one(n, args, train_loader, test_loader) for n in names]

    if len(results) > 1:
        print_comparison(results)
    save_results(results, args)


if __name__ == "__main__":
    main()
    # 结果已全部落盘，但解释器退出阶段的 GC 回收 autograd 图时仍会踩到 torch 2.2.2
    # CPU 后端的段错误（README 5.5），退出码变成 139，外层脚本会误判训练失败并重跑。
    # os._exit 立即结束进程，跳过 GC 与解释器清理 —— 此时已无任何待落盘的数据。
    # 因为它同时跳过缓冲区刷写，必须先手动 flush，否则最后几行输出会丢。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
