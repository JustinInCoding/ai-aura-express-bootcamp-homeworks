"""定位 GoogLeNet backward 的间发性段错误。

思路：崩溃大约每 400 次迭代出现一次，所以隔离测试每个候选算子必须跑到同一量级的
迭代数才有意义（之前只跑 3 次全部"通过"，是假阴性）。
子模块很小，几千次迭代只要几十秒，比压整个网络快得多。

每个 case 必须独立进程运行 —— 段错误直接杀进程。
"""
import os
import sys

# OMP_NUM_THREADS 等由外部 shell 设置，必须在 import torch（初始化 OpenMP 运行时）之前
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")

import faulthandler

faulthandler.enable()

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import Inception, conv_bn, googlenet

B = 64


def case_maxpool_s1():
    """Inception 第四分支：3x3 stride=1 的重叠窗口 MaxPool，非常规用法"""
    return nn.Sequential(nn.MaxPool2d(3, 1, 1), conv_bn(64, 16, 1, 1, 0)), (B, 64, 14, 14)


def case_maxpool_only():
    return nn.Sequential(nn.MaxPool2d(3, 1, 1)), (B, 64, 14, 14)


def case_conv5x5():
    return nn.Sequential(conv_bn(8, 16, 5, 1, 2)), (B, 8, 14, 14)


def case_conv1x1_narrow():
    """降到 8 通道这种极窄的 1x1，oneDNN 容易走到边缘 kernel"""
    return nn.Sequential(conv_bn(64, 8, 1, 1, 0)), (B, 64, 14, 14)


def case_cat4():
    class Cat4(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = nn.ModuleList([conv_bn(64, c, 1, 1, 0) for c in (32, 64, 16, 16)])

        def forward(self, x):
            return torch.cat([b(x) for b in self.b], dim=1)

    return Cat4(), (B, 64, 14, 14)


def case_inception1():
    return Inception(64, 32, 48, 64, 8, 16, 16), (B, 64, 14, 14)


def case_inception2():
    return Inception(128, 64, 64, 96, 16, 32, 32), (B, 128, 14, 14)


def case_inception3():
    return Inception(224, 96, 64, 128, 16, 32, 32), (B, 224, 7, 7)


def case_full():
    return googlenet(), (B, 1, 28, 28)


CASES = {k[5:]: v for k, v in sorted(globals().items()) if k.startswith("case_")}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CASES:
        print("可选 case:", " ".join(CASES))
        sys.exit(2)

    name = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    if os.environ.get("MKLDNN") == "off":
        torch.backends.mkldnn.enabled = False
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)

    model, shape = CASES[name]()
    # 让输入参与求导：有些 case（如纯 pooling）没有可训练参数，
    # 不这样做 backward 会直接报 "does not require grad"，测不到真正想测的算子
    x = torch.randn(*shape, requires_grad=True)
    print(f"{name} | iters={iters} | threads={torch.get_num_threads()} "
          f"| mkldnn={torch.backends.mkldnn.enabled}", flush=True)

    for i in range(iters):
        model.zero_grad()
        model(x).sum().backward()

    print("OK", flush=True)
