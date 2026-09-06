"""GoogLeNet / Inception-v1 (2014) 的 CIFAR-10 适配版。

GoogLeNet 的核心是 Inception 模块：同一个输入并行跑 1x1、3x3、5x5 卷积和
maxpool，再把结果在通道维拼起来，让网络自己"选"感受野。
原版为 ImageNet 224x224 设计，前面是 7x7 stride 2 + maxpool，CIFAR 上改用
3x3 stride 1 的 stem，整个网络只在 3a/3b、4e 后下采样：32 -> 16 -> 8 -> 4。
4a 和 4d 后面各接一个辅助分类器（论文里的 deep supervision，
训练时 loss = 主分类器 + 0.3 * 辅1 + 0.3 * 辅2，推理时只用主干）。
"""

import torch
import torch.nn as nn


class Inception(nn.Module):
    """Inception 模块。通道数含义：c1 = 1x1 分支；c3_red/c3 = 3x3 分支的
    降通道/输出通道；c5_red/c5 = 5x5 分支；pool = pool 分支的 1x1 投影。"""

    def __init__(self, in_channels, c1, c3_red, c3, c5_red, c5, pool_proj):
        super().__init__()
        self.branch1 = self._conv_bn_relu(in_channels, c1, kernel_size=1)
        self.branch2 = nn.Sequential(
            self._conv_bn_relu(in_channels, c3_red, kernel_size=1),
            self._conv_bn_relu(c3_red, c3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            self._conv_bn_relu(in_channels, c5_red, kernel_size=1),
            self._conv_bn_relu(c5_red, c5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            self._conv_bn_relu(in_channels, pool_proj, kernel_size=1),
        )

    @staticmethod
    def _conv_bn_relu(in_channels, out_channels, **kw):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kw),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x),
                          self.branch3(x), self.branch4(x)], dim=1)


class AuxClassifier(nn.Module):
    """辅助分类头。原版在 14x14 的特征图上做 5x5 pool，这里特征图只有 8x8，
    简化成全局平均池化 + 1x1 卷积 + 一个小全连接层。"""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv(self.pool(x)).flatten(1)
        return self.fc(x)


class GoogLeNetCifar(nn.Module):
    def __init__(self, num_classes: int = 10, aux_logits: bool = True):
        super().__init__()
        self.aux_logits = aux_logits
        # 论文 Table 1 的通道配置，仅 stem 换成 CIFAR 版
        self.stem = self._conv_bn_relu(3, 192, kernel_size=3, padding=1)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)    # 32x32
        self.pool1 = nn.MaxPool2d(2, stride=2)                        # -> 16
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool2 = nn.MaxPool2d(2, stride=2)                        # -> 8
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.pool3 = nn.MaxPool2d(2, stride=2)                        # -> 4
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        if aux_logits:
            self.aux1 = AuxClassifier(512, num_classes)   # 挂在 4a 后
            self.aux2 = AuxClassifier(528, num_classes)   # 挂在 4d 后
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    @staticmethod
    def _conv_bn_relu(in_channels, out_channels, **kw):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kw),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.inception3a(x))
        x = self.pool2(self.inception3b(x))
        x = self.inception4a(x)
        aux1 = self.aux1(x) if self.aux_logits else None
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x) if self.aux_logits else None
        x = self.pool3(self.inception4e(x))
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.dropout(self.avgpool(x)).flatten(1)
        logits = self.fc(x)
        if self.training and self.aux_logits:
            # 训练时返回 (主输出, 辅1, 辅2)，由 train.py 加权求 loss
            return logits, aux1, aux2
        return logits
