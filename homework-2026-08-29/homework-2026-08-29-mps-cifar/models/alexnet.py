"""AlexNet (2012) 的 CIFAR-10 适配版。

原版第一层是 11x11 stride 4，32x32 的图一层就被压到 6x6，后面三次 pool 直接归零，
11x11 在 CIFAR 上没有生存空间。但 AlexNet 的结构签名正是"大核 + 浅"，全换成 3x3
就和 VGG 没了区别，所以首两层保留成 5x5 大核，后三层 3x3。

stem_kernel 参数切换首两层的核大小：
  - 5（默认）：保留"大核层级"的 AlexNet 正身；
  - 3：全 3x3 版本，即"VGG 用堆叠小核取代大核"这一演进的对照组——
    两者除首两层核大小外完全一致，可以直接衡量核大小这一个变量的影响。
保留原版的结构特征：5 个卷积层、中间两层不接 pool、末尾三层全连接 + Dropout。
"""

import torch.nn as nn


class AlexNetCifar(nn.Module):
    def __init__(self, num_classes: int = 10, stem_kernel: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            # 32 -> 16；padding = kernel//2，两种核的输出尺寸一致，
            # 保证 3x3 / 5x5 配置间唯一的变量就是核大小本身
            nn.Conv2d(3, 64, kernel_size=stem_kernel, stride=1,
                      padding=stem_kernel // 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 16 -> 8
            nn.Conv2d(64, 192, kernel_size=stem_kernel,
                      padding=stem_kernel // 2, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 连着三层卷积不做下采样，这是 AlexNet 的标志性结构
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 8 -> 4
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
