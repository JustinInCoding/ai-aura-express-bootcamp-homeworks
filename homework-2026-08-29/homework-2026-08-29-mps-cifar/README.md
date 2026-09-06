# Day6 作业：CIFAR-10 上对比四种经典卷积网络

数据集：CIFAR-10（6 万张 32x32 彩色图，10 类，5 万训练 / 1 万测试）。
对比 AlexNet / VGG-16 / GoogLeNet / ResNet-18 四种网络在同一组超参下的表现，
再单独做超参敏感性实验。

## 环境

Mac Mini M1 (16GB)，PyTorch 走 MPS 后端：

```bash
conda create -n deep-learning-aura python=3.12
conda activate deep-learning-aura
pip install torch torchvision matplotlib
```

## 用法

```bash
# 单次训练（首次运行自动下载 CIFAR-10 到 data/）
python train.py --model resnet18 --tag baseline-resnet18

# 常用超参：--epochs --batch-size --lr --optimizer {sgd,adam} --scheduler --no-aug
# 冒烟测试（只跑几个 batch，验证环境通不通）：
python train.py --model googlenet --epochs 2 --limit-batches 20 --tag smoke

# 一键跑完整实验矩阵（基线 4 个 + 核大小消融 1 个 + 超参实验 4 个，串行）
bash run_experiments.sh

# 汇总结果：打印对比表、生成 markdown 表格片段和准确率 / loss 曲线图
python compare.py                                          # 全部 run
python compare.py baseline-resnet18 baseline-alexnet ...   # 指定 tag
python compare.py resnet18-lr001 resnet18-adam resnet18-noaug --title "ResNet18 调参" --out ablation
```

## 代码结构

```
models/
  __init__.py    # MODEL_REGISTRY：train.py 通过名字取模型
  alexnet.py     # AlexNetCifar
  vgg.py         # VGG16Cifar
  googlenet.py   # GoogLeNetCifar（带辅助分类器）
  resnet.py      # ResNet18Cifar
train.py         # 训练：逐 epoch 记录指标，结果写 runs/<tag>.json + 最优权重 .pth
compare.py       # 汇总：对比表 + 曲线图
run_experiments.sh
REPORT.md        # 实验结果与分析
```

四个网络都为 32x32 输入做了适配（原版为 ImageNet 224x224 设计，下采样次数
太多会把特征图压没）：首层卷积改小（AlexNet 首两层保留 5x5 大核层级，全 3x3
版注册为 alexnet3x3 作对照组）、减掉多余 pool、统一加 BatchNorm，
保证四个模型能在同一组超参下公平对比。细节见各文件开头的注释。
