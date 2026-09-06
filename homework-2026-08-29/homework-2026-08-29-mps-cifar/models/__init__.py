"""CIFAR-10 版本的四个经典分类网络。

原版 AlexNet / VGG / GoogLeNet / ResNet 都是为 ImageNet 224x224 设计的，
下采样次数太多，32x32 的输入喂进去特征图会塌成 1x1 甚至更小。
这里按论文里 CIFAR 实验的通行做法做了适配：
  - 首层改小、去掉紧跟其后的 maxpool（AlexNet 首两层保留 5x5，
    维持它"大核 + 浅"的结构签名，详见 models/alexnet.py；
    其余网络首层换成 3x3 stride 1）
  - 相应减少下采样次数，保证进全连接层前还有 >= 1x1 的空间尺寸
  - 统一加 BatchNorm，让四个网络能在同一组超参下训练（便于公平对比）
"""

from functools import partial

from .alexnet import AlexNetCifar
from .vgg import VGG16Cifar
from .googlenet import GoogLeNetCifar
from .resnet import ResNet18Cifar

# 名字 -> 构造函数，train.py / compare.py 通过这个表来取模型。
# AlexNet 的两个名字只差首两层核大小：5x5 是保留大核的正身，3x3 是
# "VGG 用堆叠小核取代大核"的对照组，用来单独衡量核大小的影响
MODEL_REGISTRY = {
    "alexnet5x5": partial(AlexNetCifar, stem_kernel=5),
    "alexnet3x3": partial(AlexNetCifar, stem_kernel=3),
    "vgg16": VGG16Cifar,
    "googlenet": GoogLeNetCifar,
    "resnet18": ResNet18Cifar,
}

MODEL_NAMES = list(MODEL_REGISTRY)


def build_model(name: str, num_classes: int = 10):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"未知模型 {name!r}，可选：{MODEL_NAMES}")
    return MODEL_REGISTRY[name](num_classes=num_classes)


__all__ = ["build_model", "MODEL_REGISTRY", "MODEL_NAMES"]
