"""MNIST 四网络对比训练脚本（CUDA 版）。

与 CPU 版（homework-2026-08-29/train.py）的关系：
  - `models.py` 逐字节相同，四个网络结构未做任何改动
  - 实验口径（epochs / batch / lr / optimizer / scheduler / augment / seed）完全一致
  - 只改与硬件相关的部分，改动清单见 计划.md 第 3 节

用法见文件末尾 __main__ 的 --help，或 CLAUDE.md。
所有超参数都通过命令行传入，改配置不需要动代码。
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import MODELS, build, count_params

# CPU 版实测的 MNIST 统计量。GPU 版沿用同一组值，保证归一化口径一致
# （用 compute_stats.py 可以在服务器上重新对账一遍）
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def pick_device(arg):
    if arg != "auto":
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_loaders(args, device):
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

    # pin_memory 把 batch 放进锁页内存，H2D 拷贝才能走 DMA 且与计算重叠；仅 CUDA 有意义。
    # persistent_workers 避免每个 epoch 重启 worker 进程 —— 本任务一轮只要几秒，
    # 反复 fork 的开销占比会很显眼。
    common = dict(num_workers=args.num_workers,
                  pin_memory=(device.type == "cuda"))
    if args.num_workers > 0:
        common["persistent_workers"] = True

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, **common)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, **common)
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


def sync(device):
    """CUDA 算子是异步下发的，不同步就只测到了「把 kernel 塞进队列」的时间。

    CPU 版没有这一步，因为 CPU 算子是同步执行的。少了它，s/epoch 会严重偏小，
    整个 CPU↔GPU 耗时对比和 s/MFLOP 指标就都是错的。
    """
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()          # 评估不建计算图，省显存也更快
def evaluate(model, loader, device):
    """返回 (整体准确率, 每类准确率列表)。"""
    model.eval()          # BN 改用滑动统计量、Dropout 关闭；调用方之后要记得切回 train()
    # 计数器建在 device 上，避免每个 batch 都把结果搬回 CPU 同步一次
    correct = torch.zeros(10, device=device)
    total = torch.zeros(10, device=device)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        for c in range(10):
            mask = y == c
            total[c] += mask.sum()
            correct[c] += (pred[mask] == c).sum()
    per_class = (correct / total * 100).tolist()
    return (correct.sum() / total.sum() * 100).item(), per_class


def warmup(names, args, device):
    """用随机数据把 cudnn 的算法搜索跑一遍，再开始正式计时。

    cudnn.benchmark=True 会在每种新 (层配置, 输入形状) 上试跑多种算法挑最快的。
    这个搜索只发生一次，但会让第 1 个 epoch 的耗时虚高，污染逐轮耗时数据。
    这里提前用同样的形状触发一次，代价是几秒。

    必须在 train_one 的 manual_seed 之前调用 —— 建模型和造随机输入都会消耗 RNG，
    train_one 开头的重新播种会把状态盖回去，所以不影响可复现性。
    """
    if device.type != "cuda":
        return
    t0 = time.perf_counter()
    lossf = nn.CrossEntropyLoss()
    for name in names:
        m = build(name).to(device)
        x = torch.randn(args.batch_size, 1, 28, 28, device=device)
        y = torch.randint(0, 10, (args.batch_size,), device=device)
        for _ in range(3):
            m.train()
            m.zero_grad(set_to_none=True)
            lossf(m(x), y).backward()
        m.eval()
        with torch.no_grad():
            m(torch.randn(512, 1, 28, 28, device=device))   # 评估用的 batch 形状
        del m, x, y
    sync(device)
    torch.cuda.empty_cache()
    print(f"[warmup] cudnn 算法搜索完成，耗时 {time.perf_counter() - t0:.1f}s")


def train_one(name, args, train_loader, test_loader, device):
    """训练单个模型，返回一份结果 dict（供 save_results / compare 使用）。"""
    torch.manual_seed(args.seed)          # 每个模型用同一 seed 起步，保证对比公平
    if device.type == "cuda":
        # Dropout（AlexNet / GoogLeNet 里有）在 GPU 上走的是 CUDA 的 RNG，
        # 只播 CPU 的种子不足以让这两个网络可复现
        torch.cuda.manual_seed_all(args.seed)

    model = build(name).to(device)
    n_params = count_params(model)
    opt = make_optimizer(model, args)     # 必须在 model 之后建，要拿 model.parameters()
    sched = make_scheduler(opt, args)     # --scheduler none 时是 None，后面都要判空
    # CrossEntropyLoss 内部自带 softmax，所以各网络最后一层是裸 Linear，别再手动加
    lossf = nn.CrossEntropyLoss()

    # GPU 上单组实验只要几分钟，断点不像 CPU 版那样是刚需，但保留它有两个用处：
    # 云服务器 SSH 断连、以及按量计费实例被回收时，不至于从头再来。
    ckpt_path = Path(args.out) / f"ckpt_{args.tag}_{name}.pt"
    start_ep, history, best_acc, best_per_class, elapsed = 1, [], 0.0, None, 0.0

    if args.resume and ckpt_path.exists():
        # 下面几类状态缺一不可：少 opt 则 momentum 缓冲清零、少 sched 则 lr 跳回初始值
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if sched is not None and ck["sched"] is not None:
            sched.load_state_dict(ck["sched"])
        # 恢复 RNG 状态，否则续训后的数据 shuffle 顺序与不中断的一次跑不一致，结果不可复现。
        # CPU RNG 决定 shuffle 顺序，CUDA RNG 决定 Dropout 掩码，两个都要恢复。
        torch.set_rng_state(ck["rng"].cpu())
        if device.type == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        start_ep = ck["epoch"] + 1
        history, best_acc = ck["history"], ck["best_acc"]
        best_per_class, elapsed = ck["best_per_class"], ck["elapsed"]
        print(f"\n[续训] {name} 已完成 {ck['epoch']}/{args.epochs} epochs，"
              f"当前最佳 {best_acc:.2f}%，从第 {start_ep} epoch 继续")

    n_batches = args.limit_batches or len(train_loader)   # or 取第一个真值，仅用于表头
    print(f"\n{'=' * 72}")
    print(f"训练 {name}  |  参数量 {n_params / 1e6:.2f}M  |  "
          f"{args.epochs} epochs x {n_batches} batches  |  {device}")
    print(f"{'=' * 72}")
    print(f"{'epoch':<7}{'train_loss':>12}{'train_acc':>11}{'test_acc':>10}"
          f"{'lr':>10}{'time':>9}")
    print("-" * 72)

    for ep in range(start_ep, args.epochs + 1):
        # 不是多余的：上一轮结尾的 evaluate 把模型切成了 eval，不切回来 BN 的滑动
        # 统计量会停止更新、Dropout 会一直关闭 —— 训练悄悄退化且不报错
        model.train()
        sync(device)                      # 起点也要同步，否则会把上一轮的尾巴算进来
        t0 = time.perf_counter()
        run_loss = run_correct = run_seen = 0

        for i, (x, y) in enumerate(train_loader):
            if args.limit_batches and i >= args.limit_batches:
                break                     # 冒烟测试：跑够 N 个 batch 就撤
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # 下面五步是 PyTorch 训练的骨架，顺序不能动
            opt.zero_grad(set_to_none=True)   # backward 是累加到 p.grad 而非覆盖，必须先清
            out = model(x)
            loss = lossf(out, y)
            loss.backward()               # 只算梯度
            opt.step()                    # 才更新权重（分开是为了能在中间插梯度裁剪等）

            # 乘 y.size(0)：loss 是 batch 均值，而 drop_last=False 时末尾 batch 更小，
            # 还原成总损失再除 run_seen 才是真正的样本级平均。
            # 注意 .item() 会强制等待 GPU 算完（隐式同步），是这个循环里最大的停顿点；
            # 保留它是为了与 CPU 版口径完全一致，不因统计方式差异污染对比。
            run_loss += loss.item() * y.size(0)
            run_correct += (out.argmax(1) == y).sum().item()
            run_seen += y.size(0)

        cur_lr = opt.param_groups[0]["lr"]   # 先读：step 之后 lr 就是下一轮的值了
        if sched:
            sched.step()

        # train_acc 是边训边算的，同轮前面的 batch 用的是更差的权重，加上 augment 会
        # 扰动训练图 —— 所以 train_acc < test_acc 是正常现象，不是 bug
        train_loss = run_loss / run_seen
        train_acc = run_correct / run_seen * 100
        test_acc, per_class = evaluate(model, test_loader, device)
        sync(device)                      # 与 CPU 版一致：s/epoch 含评估时间
        dt = time.perf_counter() - t0
        elapsed += dt

        # 注意 best 是按测试集挑的，报出的数字略偏乐观；各组待遇一致故对比仍公平
        if test_acc > best_acc:
            best_acc, best_per_class = test_acc, per_class

        history.append({"epoch": ep, "train_loss": train_loss, "train_acc": train_acc,
                        "test_acc": test_acc, "lr": cur_lr, "sec": dt})
        # flush：长任务输出重定向到 logs/ 时，不 flush 进度会卡在缓冲区里看不见
        print(f"{ep:<7}{train_loss:>12.4f}{train_acc:>10.2f}%{test_acc:>9.2f}%"
              f"{cur_lr:>10.5f}{dt:>8.1f}s", flush=True)

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict() if sched else None,
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "epoch": ep, "history": history,
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
        # 硬件信息一并存下：GPU 版结果要与 CPU 版对照，得知道每份数据出自什么机器
        "env": env_info(),
    }


def env_info():
    info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info.update({"gpu": torch.cuda.get_device_name(0),
                     "cuda": torch.version.cuda,
                     "cudnn": torch.backends.cudnn.version()})
    return info


def save_results(results, args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for r in results:
        with open(out / f"{args.tag}_{r['model']}.json", "w") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)

    csv_path = out / "summary.csv"
    fields = ["tag", "model", "params", "epochs", "batch_size", "lr", "optimizer",
              "momentum", "weight_decay", "scheduler", "augment",
              "best_test_acc", "final_test_acc", "sec_per_epoch", "total_min", "gpu"]
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
                "gpu": r["env"].get("gpu", "cpu"),
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
        description="MNIST 上对比 AlexNet / VGG / GoogLeNet / ResNet（CUDA 版）",
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
    # --- 以下为运行环境参数，不属于被研究的超参数 ---
    p.add_argument("--device", default="auto", help="auto / cuda / cuda:0 / cpu")
    p.add_argument("--num-workers", type=int, default=4,
                   help="Linux 上 fork 便宜，开 worker 是净赚。augment 组的仿射变换在 "
                        "CPU 上做，worker 少了会让 GPU 饿着。改这个值会改变 augment "
                        "组的随机流，同一批对比实验中必须保持一致")
    p.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True,
                   help="让 cudnn 自动挑最快算法。输入尺寸固定（本任务恒为 28x28）时是净收益")
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
    device = pick_device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} "
              f"| CUDA {torch.version.cuda} | cudnn {torch.backends.cudnn.version()} "
              f"| benchmark={args.cudnn_benchmark}")
    else:
        # 显式提醒：如果本该用 GPU 却掉回了 CPU，跑完再发现就浪费了机时
        print(f"torch {torch.__version__} | device=cpu "
              f"| cuda_available={torch.cuda.is_available()}")
        if args.device == "auto":
            print("[警告] 没有检测到可用的 CUDA 设备，本次将在 CPU 上训练")
    print(f"配置：{vars(args)}")

    torch.manual_seed(args.seed)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = get_loaders(args, device)
    names = list(MODELS) if args.model == "all" else [args.model]

    warmup(names, args, device)           # 必须在 train_one 播种之前

    results = [train_one(n, args, train_loader, test_loader, device) for n in names]

    if len(results) > 1:
        print_comparison(results)
    save_results(results, args)


if __name__ == "__main__":
    main()
