"""四种经典 CNN 架构，适配 MNIST 的 28x28 单通道输入。

原版四个网络都是为 224x224 ImageNet 设计的，下采样倍数均为 32（224 -> 7）。
28x28 直接套用会让特征图塌缩为 0，因此必须削减下采样层。
这里的原则是：**减少下采样、收窄通道，但完整保留每个架构的标志性设计**，
这样对比出来的差异才真正反映架构思想的不同，而不是实现细节的差异。
"""

import torch
import torch.nn as nn


def conv_bn(cin, cout, k=3, s=1, p=1):
    """Conv-BN-ReLU 三件套。BN 让深层网络能用更大学习率且不易发散。"""
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, bias=False),   # 后接 BN，bias 会被其均值项吸收，故省去
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


# --------------------------------------------------------------------------
# AlexNet (2012) — 标志：5 层卷积堆叠 + 大卷积核 + 巨大 FC 头 + Dropout
# --------------------------------------------------------------------------
def alexnet():
    return nn.Sequential(
        nn.Conv2d(1, 64, 5, 1, 2), nn.ReLU(True), nn.MaxPool2d(2),     # 28 -> 14
        nn.Conv2d(64, 192, 5, 1, 2), nn.ReLU(True), nn.MaxPool2d(2),   # 14 -> 7
        # 连续三层卷积不夹池化，是 AlexNet 的原始特征
        nn.Conv2d(192, 384, 3, 1, 1), nn.ReLU(True),
        nn.Conv2d(384, 256, 3, 1, 1), nn.ReLU(True),
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2),  # 7 -> 3
        nn.Flatten(),
        # Dropout 是 AlexNet 论文的关键贡献之一，用来压制大 FC 层的过拟合
        nn.Dropout(0.5), nn.Linear(256 * 3 * 3, 1024), nn.ReLU(True),
        nn.Dropout(0.5), nn.Linear(1024, 10),
    )


# --------------------------------------------------------------------------
# VGG (2014) — 标志：全部用 3x3 小核，靠"块"堆叠加深
# 两个 3x3 叠加的感受野等于一个 5x5，但参数更少、非线性更多
# --------------------------------------------------------------------------
def vgg():
    layers, cin = [], 1
    # (输出通道, 块内卷积层数, 块后是否池化)
    for cout, n_conv, pool in [(32, 1, True), (64, 1, True), (128, 2, True), (256, 2, False)]:
        for _ in range(n_conv):
            layers.append(conv_bn(cin, cout))
            cin = cout
        if pool:
            layers.append(nn.MaxPool2d(2))                              # 28 -> 14 -> 7 -> 3
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 10)]
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------
# GoogLeNet (2014) — 标志：Inception 模块，多尺度分支并行
# --------------------------------------------------------------------------
class Inception(nn.Module):
    """四条不同感受野的分支并行计算，再沿通道维拼接。

    关键技巧是 3x3/5x5 分支前先用 1x1 卷积降通道（c3r/c5r），
    把昂贵的大核卷积放在更窄的特征图上算 —— 这就是 GoogLeNet
    能用极少参数（本实现仅 0.27M）达到高精度的原因。
    """

    def __init__(self, cin, c1, c3r, c3, c5r, c5, cp):
        super().__init__()
        self.b1 = conv_bn(cin, c1, 1, 1, 0)
        self.b2 = nn.Sequential(conv_bn(cin, c3r, 1, 1, 0), conv_bn(c3r, c3))
        self.b3 = nn.Sequential(conv_bn(cin, c5r, 1, 1, 0), conv_bn(c5r, c5, 5, 1, 2))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), conv_bn(cin, cp, 1, 1, 0))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


def googlenet():
    return nn.Sequential(
        conv_bn(1, 64), nn.MaxPool2d(2),                                # 28 -> 14
        Inception(64, 32, 48, 64, 8, 16, 16),                           # 通道 -> 128
        Inception(128, 64, 64, 96, 16, 32, 32), nn.MaxPool2d(2),        # -> 224, 14 -> 7
        Inception(224, 96, 64, 128, 16, 32, 32),                        # -> 288
        # 全局平均池化替代大 FC 层，也是 GoogLeNet 的贡献
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(288, 10),
    )


# --------------------------------------------------------------------------
# ResNet (2015) — 标志：残差捷径，让梯度绕过卷积层直接回传
# --------------------------------------------------------------------------
class BasicBlock(nn.Module):
    """输出 = ReLU(卷积主干(x) + 捷径(x))。

    捷径让梯度有一条不经过任何权重的通路，这是深网络能训起来的关键。
    当通道数或分辨率变化时，捷径需要 1x1 卷积做形状对齐。
    """

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            conv_bn(cin, cout, 3, stride, 1),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout),
        )
        self.short = nn.Sequential() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout),
        )

    def forward(self, x):
        return torch.relu(self.body(x) + self.short(x))


def resnet():
    # stem 先把 28->14。若照 CIFAR 惯例保持全分辨率，本机 CPU 上单 epoch 要 22 分钟
    # 而不是 5 分钟，对 MNIST 这种简单数据是纯浪费。
    return nn.Sequential(
        conv_bn(1, 32), nn.MaxPool2d(2),                                # 28 -> 14
        BasicBlock(32, 64), BasicBlock(64, 64),                         # 14
        BasicBlock(64, 128, 2), BasicBlock(128, 128),                   # 14 -> 7
        BasicBlock(128, 256, 2), BasicBlock(256, 256),                  # 7 -> 4
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 10),
    )

MODELS = {
    "alexnet": alexnet,
    "vgg": vgg,
    "googlenet": googlenet,
    "resnet": resnet,
}


def build(name):
    if name not in MODELS:
        raise ValueError(f"未知模型 {name!r}，可选：{list(MODELS)}")
    return MODELS[name]()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 自检：确认四个网络都能吃 28x28 并输出 10 类
    x = torch.randn(2, 1, 28, 28)
    print(f"{'model':<12}{'params':>12}{'output':>12}")
    print("-" * 36)
    for name in MODELS:
        m = build(name)
        m.eval()
        with torch.no_grad():
            out = m(x)
        assert out.shape == (2, 10), f"{name} 输出形状错误: {out.shape}"
        print(f"{name:<12}{count_params(m) / 1e6:>11.2f}M{str(tuple(out.shape)):>12}")
    print("\n四个网络形状自检通过")
