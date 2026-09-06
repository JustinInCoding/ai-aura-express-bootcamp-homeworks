"""探测四种经典网络适配到 28x28 后的形状与训练开销，用于确认输入尺寸方案。"""

import time

import torch
import torch.nn as nn


def conv_bn(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def alexnet28():
    """保留 5 层卷积堆叠 + 大 FC 头的 AlexNet 特征，下采样降为 4 倍。"""
    return nn.Sequential(
        nn.Conv2d(1, 64, 5, 1, 2), nn.ReLU(True), nn.MaxPool2d(2),      # 28->14
        nn.Conv2d(64, 192, 5, 1, 2), nn.ReLU(True), nn.MaxPool2d(2),    # 14->7
        nn.Conv2d(192, 384, 3, 1, 1), nn.ReLU(True),
        nn.Conv2d(384, 256, 3, 1, 1), nn.ReLU(True),
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
        nn.AdaptiveAvgPool2d(7),
        nn.Flatten(), nn.Dropout(0.5), nn.Linear(256 * 7 * 7, 1024), nn.ReLU(True),
        nn.Dropout(0.5), nn.Linear(1024, 10),
    )


def vgg28():
    """VGG 的核心是 3x3 小核堆叠成块，这里用 4 个块，下采样 3 次。"""
    layers, cin = [], 1
    for cout, n, pool in [(64, 1, True), (128, 1, True), (256, 2, True), (512, 2, False)]:
        for _ in range(n):
            layers += [conv_bn(cin, cout)]
            cin = cout
        if pool:
            layers += [nn.MaxPool2d(2)]                                  # 28->14->7->3
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 10)]
    return nn.Sequential(*layers)


class Inception(nn.Module):
    """GoogLeNet 的灵魂：四条不同感受野的分支并行后拼接。"""

    def __init__(self, cin, c1, c3r, c3, c5r, c5, cp):
        super().__init__()
        self.b1 = conv_bn(cin, c1, 1, 1, 0)
        self.b2 = nn.Sequential(conv_bn(cin, c3r, 1, 1, 0), conv_bn(c3r, c3))
        self.b3 = nn.Sequential(conv_bn(cin, c5r, 1, 1, 0), conv_bn(c5r, c5, 5, 1, 2))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), conv_bn(cin, cp, 1, 1, 0))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], 1)


def googlenet28():
    return nn.Sequential(
        conv_bn(1, 64), nn.MaxPool2d(2),                                 # 28->14
        Inception(64, 32, 48, 64, 8, 16, 16),                            # ->128
        Inception(128, 64, 64, 96, 16, 32, 32), nn.MaxPool2d(2),         # ->224, 14->7
        Inception(224, 96, 64, 128, 16, 32, 32),                         # ->288
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(288, 10),
    )


class BasicBlock(nn.Module):
    """ResNet 的灵魂：恒等捷径让梯度直通。"""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            conv_bn(cin, cout, 3, stride, 1),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout),
        )
        self.short = (nn.Sequential() if stride == 1 and cin == cout else
                      nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                    nn.BatchNorm2d(cout)))

    def forward(self, x):
        return torch.relu(self.body(x) + self.short(x))


def resnet28():
    """ResNet-18 结构，采用 CIFAR 惯例的精简 stem（去掉 7x7 s2 和 maxpool）。"""
    return nn.Sequential(
        conv_bn(1, 64),                                                  # 28
        BasicBlock(64, 64), BasicBlock(64, 64),
        BasicBlock(64, 128, 2), BasicBlock(128, 128),                    # 28->14
        BasicBlock(128, 256, 2), BasicBlock(256, 256),                   # 14->7
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 10),
    )


def bench(model, size, batch=64, iters=3):
    x = torch.randn(batch, 1, size, size)
    y = torch.randint(0, 10, (batch,))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    lossf = nn.CrossEntropyLoss()

    def step():
        opt.zero_grad()
        lossf(model(x), y).backward()
        opt.step()

    step()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    return (time.perf_counter() - t0) / iters


MNIST_N = 60_000

if __name__ == "__main__":
    builders = [("AlexNet", alexnet28), ("VGG", vgg28),
                ("GoogLeNet", googlenet28), ("ResNet", resnet28)]

    print(f"{'model':<12}{'params':>10}{'out':>8}{'s/batch':>10}{'epoch@28':>11}{'epoch@32':>11}")
    print("-" * 62)
    total28 = total32 = 0.0
    for name, fn in builders:
        m = fn()
        params = sum(p.numel() for p in m.parameters())
        out = tuple(m(torch.randn(2, 1, 28, 28)).shape)
        s28 = bench(m, 28)
        s32 = bench(fn(), 32)
        e28, e32 = MNIST_N / (64 / s28), MNIST_N / (64 / s32)
        total28 += e28
        total32 += e32
        print(f"{name:<12}{params / 1e6:>9.2f}M{str(out[1]):>8}{s28:>10.3f}"
              f"{e28 / 60:>10.1f}m{e32 / 60:>10.1f}m")
    print("-" * 62)
    print(f"{'4 nets x 1 epoch':<30}{total28 / 60:>28.1f}m{total32 / 60:>10.1f}m")
    print(f"{'4 nets x 5 epochs':<30}{total28 * 5 / 60:>28.1f}m{total32 * 5 / 60:>10.1f}m")
