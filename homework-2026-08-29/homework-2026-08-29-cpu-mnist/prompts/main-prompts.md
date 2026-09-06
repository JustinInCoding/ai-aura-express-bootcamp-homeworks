### 计划任务/判断环境

请帮我检查一下当前电脑环境是否满足进行图像分类神经网络深度学习的训练
- 训练集采用 mnist 手写体
- 当前操作系统是 MacOS，Intel 芯片，16 G 内存
- 框架采用 pytorch，在本机已创建了 conda 环境 （deep-learning-aura）
告诉我结果是否可行

```plain

结论：可行，但需要先装 PyTorch

  硬件实测
  - macOS 11.7.11 (Big Sur)，Intel i7-4770HQ 4核8线程，16 GB 内存
  - 磁盘只剩 11 GB（已用 95%） ← 唯一风险点

  环境现状
  - deep-learning-aura：Python 3.10.18，完全空的（只有 pip/setuptools/wheel），torch 未安装

  关键限制
  - 无 CUDA、无 MPS（MPS 只支持 Apple Silicon）→ 只能纯 CPU 训练
  - torch 版本被锁在 2.2.2（配 torchvision 0.17.2）——PyTorch 2.3+ 不再发布 macOS x86_64 wheel

  MNIST 可行性
  数据集仅 ~55 MB，28×28 灰度图。简单 CNN（2 层卷积）在 4 核 CPU 上约 40–70 秒/epoch，10 epoch 十分钟内跑完，内存占用 < 2 GB。完全够用。

  要我现在装 torch 2.2.2 + torchvision 0.17.2 并跑一个基准测试验证吗？（安装约占 1.5 GB 磁盘）

```

### 任务计划

我们先来配置环境，在 deep-learning-aura 中需要安装 torch 2.2.2 + torchvision 0.17.2 并跑一个基准测试验证。
然后在 README.md 中进行环境配置说明
接下来，我们的任务是针对 minist 数据集进行图像分类神经网络的训练，首先告诉我数据集的下载地址
然后我们需要构建 AlexNet, VGG, GoogLeNet, ResNet 四种神经网络，并进行训练，并对比四种神经网络的识别效果
最后，我们需要适当调整网络的超参数，看看网络的准确性发生了什么变化 (这一步先帮我做一个计划，我需要确认一下)
每完成一个操作步骤都告诉我一下，我确认后再继续（当中可能会有调整）

安装 torch 告诉我命令行，我来安装