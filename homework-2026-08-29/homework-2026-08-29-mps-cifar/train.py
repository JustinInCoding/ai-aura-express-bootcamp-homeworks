"""CIFAR-10 训练脚本。

用法示例：
    python train.py --model resnet18 --tag baseline-resnet18
    python train.py --model vgg16 --epochs 20 --lr 0.01 --tag vgg16-lr001
    python train.py --model alexnet --epochs 1 --limit-batches 5 --tag smoke   # 冒烟测试

每个 run 的结果（配置 + 逐 epoch 指标 + 最优精度）写到 runs/<tag>.json，
最优权重写到 runs/<tag>.pth，compare.py 会汇总所有 json 生成对比报告。
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from models import MODEL_NAMES, build_model

# CIFAR-10 全训练集的通道均值 / 标准差
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# GoogLeNet 论文中辅助分类器的 loss 权重
AUX_WEIGHT = 0.3


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 训练")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--tag", default=None, help="本次实验的名字，默认用模型名")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率；不传则 SGD 用 0.1，Adam 用 0.001")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--no-aug", action="store_true", help="关闭数据增强")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-batches", type=int, default=0,
                        help="只训练/验证前 N 个 batch（冒烟测试用），0 = 不限制")
    return parser.parse_args()


def build_loaders(args):
    norm = T.Normalize(CIFAR_MEAN, CIFAR_STD)
    if args.no_aug:
        train_tf = T.Compose([T.ToTensor(), norm])
    else:
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),   # 随机平移
            T.RandomHorizontalFlip(),      # 随机镜像
            T.ToTensor(),
            norm,
        ])
    test_tf = T.Compose([T.ToTensor(), norm])

    train_set = torchvision.datasets.CIFAR10(
        args.data_dir, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(
        args.data_dir, train=False, download=True, transform=test_tf)

    common = dict(batch_size=args.batch_size, num_workers=args.workers,
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)

    if args.limit_batches:
        train_loader = _truncate(train_loader, args.limit_batches)
        test_loader = _truncate(test_loader, args.limit_batches)
    return train_loader, test_loader


def _truncate(loader, n):
    """取前 n 个 batch 拼成一个假 loader，冒烟测试用。"""
    batches = []
    for i, batch in enumerate(loader):
        if i >= n:
            break
        batches.append(batch)

    class _Sliced:
        def __iter__(self):
            return iter(batches)

        def __len__(self):
            return len(batches)

    return _Sliced()


def compute_loss(criterion, logits, targets):
    """GoogLeNet 训练时返回 (主, 辅1, 辅2)，按论文权重合成总 loss。"""
    if isinstance(logits, tuple):
        return criterion(logits[0], targets) + sum(
            AUX_WEIGHT * criterion(aux, targets) for aux in logits[1:])
    return criterion(logits, targets)


def num_correct(logits, targets):
    main = logits[0] if isinstance(logits, tuple) else logits
    return (main.argmax(dim=1) == targets).sum().item()


def run_epoch(model, loader, device, criterion, optimizer=None):
    """跑一个 epoch。传 optimizer 是训练，不传是验证。返回 (loss, acc)。"""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = compute_loss(criterion, logits, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * targets.size(0)
            correct += num_correct(logits, targets)
            total += targets.size(0)
    return total_loss / total, correct / total


def main():
    args = parse_args()
    tag = args.tag or args.model

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[{tag}] device={device} model={args.model} "
          f"epochs={args.epochs} bs={args.batch_size}")

    train_loader, test_loader = build_loaders(args)
    model = build_model(args.model, num_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] params={n_params / 1e6:.2f}M "
          f"train={len(train_loader)}batches test={len(test_loader)}batches")

    if args.optimizer == "sgd":
        lr = args.lr if args.lr is not None else 0.1
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        lr = args.lr if args.lr is not None else 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)

    criterion = nn.CrossEntropyLoss()
    args_dict = {k: str(v) if isinstance(v, Path) else v
                 for k, v in vars(args).items()} | {"lr": lr, "device": str(device)}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    history, best_acc, best_epoch = [], 0.0, -1
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = run_epoch(
            model, train_loader, device, criterion, optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, device, criterion)
        if args.scheduler == "cosine":
            scheduler.step()
        epoch_time = time.perf_counter() - t0

        if test_acc > best_acc:
            best_acc, best_epoch = test_acc, epoch
            torch.save({"model": model.state_dict(), "tag": tag,
                        "epoch": epoch, "test_acc": test_acc,
                        "config": args_dict}, args.out_dir / f"{tag}.pth")

        history.append({"epoch": epoch, "lr": lr, "train_loss": round(train_loss, 4),
                        "train_acc": round(train_acc, 4), "test_loss": round(test_loss, 4),
                        "test_acc": round(test_acc, 4), "sec": round(epoch_time, 1)})
        print(f"[{tag}] epoch {epoch:3d}/{args.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"test loss {test_loss:.4f} acc {test_acc:.4f} | "
              f"best {best_acc:.4f} @ {best_epoch} | {epoch_time:.1f}s")

        # 每个 epoch 都落盘，训练中断也能拿到已完成部分
        (args.out_dir / f"{tag}.json").write_text(json.dumps({
            "tag": tag, "model": args.model, "params": n_params,
            "config": args_dict, "history": history,
            "best_acc": best_acc, "best_epoch": best_epoch,
            "total_sec": round(time.perf_counter() - t_start, 1),
        }, ensure_ascii=False, indent=2))

    print(f"[{tag}] done. best test acc = {best_acc:.4f} (epoch {best_epoch}), "
          f"total {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
