"""环境基准测试：测算本机 CPU 训练 MNIST 的吞吐量，用于选择输入分辨率与网络规模。"""

import platform
import time

import torch
import torch.nn as nn


def timeit(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def bench_matmul(n=2048):
    a, b = torch.randn(n, n), torch.randn(n, n)
    sec = timeit(lambda: a @ b)
    return 2 * n**3 / sec / 1e9


def simple_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
    )


def alexnet_like(size):
    """AlexNet 风格骨干，用自适应池化兼容不同输入尺寸。"""
    return nn.Sequential(
        nn.Conv2d(1, 64, 11, stride=4, padding=5), nn.ReLU(), nn.MaxPool2d(3, 2),
        nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2),
        nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
        nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(2),
        nn.Flatten(), nn.Linear(256 * 4, 10),
    )


def bench_train_step(model, size, batch=64):
    """测一次 forward+backward+step 的耗时。"""
    x = torch.randn(batch, 1, size, size)
    y = torch.randint(0, 10, (batch,))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    lossf = nn.CrossEntropyLoss()

    def step():
        opt.zero_grad()
        lossf(model(x), y).backward()
        opt.step()

    return timeit(step, warmup=1, iters=3)


MNIST_TRAIN_N = 60_000

if __name__ == "__main__":
    print(f"platform : {platform.platform()} / {platform.machine()}")
    print(f"torch    : {torch.__version__} | threads={torch.get_num_threads()} "
          f"| mkldnn={torch.backends.mkldnn.is_available()}")
    print(f"\nmatmul 2048^3 : {bench_matmul():.1f} GFLOP/s\n")

    batch = 64
    cases = [
        ("SimpleCNN", simple_cnn(), 28),
        ("AlexNet-like", alexnet_like(28), 28),
        ("AlexNet-like", alexnet_like(64), 64),
        ("AlexNet-like", alexnet_like(224), 224),
    ]

    print(f"{'model':<14}{'input':>8}{'params':>10}{'s/batch':>10}{'img/s':>9}{'est.epoch':>12}")
    print("-" * 63)
    for name, model, size in cases:
        params = sum(p.numel() for p in model.parameters())
        sec = bench_train_step(model, size, batch)
        ips = batch / sec
        epoch = MNIST_TRAIN_N / ips
        epoch_str = f"{epoch:.0f}s" if epoch < 120 else f"{epoch / 60:.1f}min"
        print(f"{name:<14}{f'{size}x{size}':>8}{params / 1e6:>9.2f}M"
              f"{sec:>10.3f}{ips:>9.0f}{epoch_str:>12}")
