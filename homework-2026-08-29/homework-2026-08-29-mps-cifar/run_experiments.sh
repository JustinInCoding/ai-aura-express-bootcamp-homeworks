#!/bin/bash
# 按 homework 要求跑完整实验矩阵，全部串行（MPS 上多进程会互相抢卡）。
#   第 1 组：四个模型的基线对比（作业第 2 题）
#   第 2 组：AlexNet 首两层核大小消融（5x5 vs 3x3，检验 VGG 的设计规则）
#   第 3 组：单变量超参实验（作业第 3 题），在 ResNet18 / VGG16 上各改一个因素
set -e
cd "$(dirname "$0")"
PY="conda run --no-capture-output -n deep-learning-aura python"

echo "=== 基线：10 epochs, SGD lr=0.1, cosine, 数据增强 ==="
$PY train.py --model resnet18   --tag baseline-resnet18 --workers 0
$PY train.py --model alexnet5x5 --tag baseline-alexnet  --workers 0
$PY train.py --model vgg16      --tag baseline-vgg16    --workers 0
$PY train.py --model googlenet  --tag baseline-googlenet --workers 0

echo "=== AlexNet 核大小消融：全 3x3 对照组（同骨架同配方，唯一变量是首两层核）==="
$PY train.py --model alexnet3x3 --tag alexnet3x3 --workers 0

echo "=== 超参实验（基线之外每次改一个变量）==="
$PY train.py --model resnet18 --lr 0.01      --tag resnet18-lr001 --workers 0
$PY train.py --model resnet18 --optimizer adam --tag resnet18-adam --workers 0
$PY train.py --model resnet18 --no-aug       --tag resnet18-noaug --workers 0
$PY train.py --model vgg16    --lr 0.01      --tag vgg16-lr001 --workers 0

echo "=== 全部实验完成 ==="
