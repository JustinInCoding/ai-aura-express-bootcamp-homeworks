"""VGG-16 (2014) 的 CIFAR-10 适配版。

VGG 的核心思想是"只用 3x3 小卷积堆深度"，这一点在 32x32 上不用改。
唯一的改动在末尾：原版 5 次 maxpool 后是 7x7x512 接三个大全连接层，
32x32 经 5 次 pool 后是 1x1x512，直接接一个 Linear(512, 10) 即可。
"""

import torch.nn as nn

# 数字 = 输出通道数，"M" = maxpool。这是论文里的配置 D（16 层带权重层）
CFG_VGG16 = [
    64, 64, "M",
    128, 128, "M",
    256, 256, 256, "M",
    512, 512, 512, "M",
    512, 512, 512, "M",
]


def _make_layers(cfg, in_channels: int = 3):
    layers = []
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        layers += [
            nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(v),
            nn.ReLU(inplace=True),
        ]
        in_channels = v
    return nn.Sequential(*layers)


class VGG16Cifar(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = _make_layers(CFG_VGG16)  # 32 -> 16 -> 8 -> 4 -> 2 -> 1
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
