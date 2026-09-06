#!/usr/bin/env bash
# AutoDL 实例的一次性环境准备。
#
# 用「无卡模式开机」（¥0.1/小时）跑这个脚本，全部完成后再关机、换成带卡的
# 规格开机跑训练 —— 装依赖和下数据不需要 GPU，没必要按 GPU 价格计费。
#
# 用法：bash setup.sh

set -euo pipefail

echo "==== 1/5 检查 GPU 与驱动 ===="
# 无卡模式下没有 GPU，nvidia-smi 会失败，这是预期的，不中断
nvidia-smi || echo "[提示] 当前是无卡模式或无 GPU，属正常，训练前记得换带卡规格开机"

echo ""
echo "==== 2/5 检查 PyTorch ===="
# AutoDL 的 PyTorch 镜像自带 CUDA 版 torch。
# 千万不要 pip install torch —— 默认源装的是 CPU 版轮子，会把镜像里的覆盖掉。
python - <<'PY'
import torch, torchvision
print(f"torch       {torch.__version__}")
print(f"torchvision {torchvision.__version__}")
print(f"cuda 编译版本 {torch.version.cuda}")
print(f"cuda 可用     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU         {torch.cuda.get_device_name(0)}")
    print(f"显存        {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"cudnn       {torch.backends.cudnn.version()}")
PY

echo ""
echo "==== 3/5 安装画图依赖 ===="
pip install -q -r requirements.txt

echo ""
echo "==== 4/5 安装中文字体 ===="
# 不装的话 compare.py 出的图里所有中文都是方框。
# AutoDL 默认是 root，不需要 sudo；本地非 root 环境请自行加 sudo。
if fc-list :lang=zh 2>/dev/null | grep -q .; then
  echo "已有中文字体，跳过"
else
  apt-get update -qq && apt-get install -y -qq fonts-noto-cjk
  # matplotlib 会缓存字体列表，装完必须清掉，否则新字体扫不到
  rm -rf ~/.cache/matplotlib
  echo "已安装 fonts-noto-cjk 并清空 matplotlib 字体缓存"
fi

echo ""
echo "==== 5/5 预下载 MNIST ===="
# 训练脚本里 download=True 会自动下，但放在这里做有两个好处：
# 1. 无卡模式下完成，不占 GPU 机时
# 2. 下载失败能立刻发现，而不是等到训练启动才报错
python - <<'PY'
from torchvision import datasets
for train in (True, False):
    datasets.MNIST("./data", train=train, download=True)
print("MNIST 就绪")
PY

echo ""
echo "==== 环境准备完成 ===="
echo "下一步：关机 → 换带卡规格开机 → bash smoke.sh 冒烟测试"