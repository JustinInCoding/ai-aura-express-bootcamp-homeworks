"""ResNet-18 (2015) 的 CIFAR-10 适配版。

原版 ResNet 首层是 7x7 stride 2 + maxpool，32x32 的图开局就掉到 8x8，
后面四个 stage 下采样完只剩 1x1，深层根本学不到东西。
这里按论文第 4.2 节的 CIFAR 实验来改：首层换成 3x3 stride 1，去掉 maxpool，
四个 stage 分别在 32 -> 32 -> 16 -> 8 -> 4 的分辨率上跑，残差连接原样保留。
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """ResNet 的基础残差块（两层 3x3 卷积 + 跳跃连接）。"""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        # 形状对不上时（通道数变化或空间下采样），shortcut 也做一次 1x1 卷积
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class ResNet18Cifar(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)  # 32 -> 32
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)  # 32 -> 16
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)  # 16 -> 8
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)  # 8 -> 4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        # 第一个块负责下采样（stride=2），后面的块保持分辨率
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = self.avgpool(out).flatten(1)
        return self.fc(out)
