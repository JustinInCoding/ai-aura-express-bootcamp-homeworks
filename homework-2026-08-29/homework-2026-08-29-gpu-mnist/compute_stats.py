"""统计 MNIST 的像素均值/标准差，验证 train.py 里硬编码的 0.1307 / 0.3081。

train.py:38 直接写死了这两个常数，这个脚本把它算出来对账一遍。

两个容易踩的点：

1. transform 只能是 ToTensor()，**不能带 Normalize**。ToTensor 已经把
   0-255 除成 0-1，我们要的就是这个尺度上的统计量；再套一层 Normalize
   就变成"对已经标准化过的数据再统计"，结果会是 mean≈0 std≈1。

2. 是把所有像素混在一起算一个数，不是每张图算一个再平均。因为
   Normalize 是逐像素操作的，它需要的就是全局标量。
   （两者在 MNIST 上恰好接近，因为每张图像素数相同；一旦图片尺寸不一
   就会有偏差。）

用 sum 与 sum-of-squares 累加、靠 E[X²]-E[X]² 反推方差，而不是把整个
数据集 load 进内存。MNIST 才 47MB 无所谓，但大数据集上这是唯一可行的写法。

用法：
    python compute_stats.py
    python compute_stats.py --num-workers 0     # 排查 worker 相关问题时
"""

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# train.py:38 里硬编码的值，用来对账
EXPECTED = {"mean": 0.1307, "std": 0.3081}


def compute(split, args):
    """遍历一个 split，返回 (mean, std, 像素总数)。"""
    dataset = datasets.MNIST(args.data, train=(split == "train"),
                             download=True, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    # 用 float64 累加：float32 累到 4700 万个像素时尾数精度会明显掉
    n = 0
    total = 0.0
    total_sq = 0.0
    for x, _ in loader:
        x = x.double()
        n += x.numel()
        total += x.sum().item()
        total_sq += (x ** 2).sum().item()

    mean = total / n
    var = total_sq / n - mean ** 2          # E[X²] - E[X]²
    return mean, var ** 0.5, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./data", help="数据集根目录")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    print(f"torch {torch.__version__}\n")

    stats = {}
    for split in ("train", "test"):
        mean, std, n = compute(split, args)
        stats[split] = (mean, std)
        print(f"{split:5s}  mean={mean:.6f}  std={std:.6f}   共 {n:,} 个像素")

    train_mean, train_std = stats["train"]
    test_mean, test_std = stats["test"]

    print(f"\n对账 train.py:38 的硬编码值（保留 4 位小数）：")
    for name, got in (("mean", train_mean), ("std", train_std)):
        want = EXPECTED[name]
        mark = "✅" if round(got, 4) == want else "❌"
        print(f"  {mark} {name}: 实测 {got:.4f}  硬编码 {want}")

    print(f"\n训练集与测试集的统计量并不相同：")
    print(f"  mean 差 {abs(train_mean - test_mean):.6f}   "
          f"std 差 {abs(train_std - test_std):.6f}")
    print(f"  归一化时两边都必须用训练集的那一组 —— 用测试集自己的是数据泄露。")


if __name__ == "__main__":
    main()
