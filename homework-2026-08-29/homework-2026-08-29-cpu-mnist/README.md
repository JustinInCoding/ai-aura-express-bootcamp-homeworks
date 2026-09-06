# 深度学习 Day6 作业：MNIST 图像分类神经网络对比

用 AlexNet / VGG / GoogLeNet / ResNet 四种经典 CNN 架构对 MNIST 手写数字做分类，并对比识别效果与超参数影响。

**结论速览**（详见 [6. 实验结果](#6-实验结果)）：

- 四种架构准确率只差 **0.11pp**，但参数量差 18 倍、耗时差 3.6 倍 —— MNIST 上架构差异体现不出来，能比出来的只有成本。
- 最佳配置是 **VGG + 余弦退火 + 数据增强，20 epochs，99.67%**，比最好的 ResNet（99.37%）高 0.30pp，参数量却只有它的 41%。**调超参数比换架构划算。**
- 最有价值的一条发现：**cosine 和 augment 单独用在 20 轮下都没有收益（−0.04 / +0.04），组合起来却有 +0.14 —— 提升完全来自交互项。** 严格的单变量对照能保证归因清晰，代价是看不见这种交互。

---

## 1. 硬件环境

| 项 | 值 |
|---|---|
| CPU | Intel Core i7-4770HQ @ 2.20GHz（Haswell，4 核 8 线程） |
| 内存 | 16 GB |
| 系统 | macOS 11.7.11 (Big Sur)，x86_64 |
| GPU 加速 | **无**（无 NVIDIA 显卡 → 无 CUDA；MPS 仅支持 Apple Silicon → 不可用） |

训练全程为 **纯 CPU**。好消息是 oneDNN (MKL-DNN) 后端可用，卷积有 SIMD 加速；实测 `matmul 2048³` 达 **276 GFLOP/s**。

---

## 2. 环境配置

### 2.1 创建 conda 环境

```bash
conda create -n deep-learning-aura python=3.10 -y
conda activate deep-learning-aura
```

### 2.2 安装依赖

```bash
pip install torch==2.2.2 torchvision==0.17.2
pip install "numpy<2"
pip install matplotlib pandas tqdm
```

> 国内网络慢可加镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`

### 2.3 两个必须锁版本的坑

**坑 1：torch 必须锁 2.2.2**

PyTorch 从 **2.3 起不再发布 macOS x86_64 wheel**（只提供 arm64）。因此 Intel Mac 上可安装的最高版本就是 `2.2.2`，配套 torchvision 为 `0.17.2`。不写 `==` 会直接解析失败。

```bash
# 验证本平台可用版本上限
pip index versions torch        # → 2.2.2 是最高版
pip index versions torchvision  # → 0.17.2 是最高版
```

也**不要**加 `--index-url .../cu121` 之类的 CUDA 源，本机无 NVIDIA 显卡，默认 wheel 即 CPU 版。

**坑 2：numpy 必须降到 1.x**

torch 2.2.2 是用 NumPy 1.x 编译的。若环境里是 NumPy 2.x，会报：

```
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
```

此时 torch 张量运算仍能跑，但 **torch ↔ numpy 的桥梁是断的**：`tensor.numpy()`、`torch.from_numpy()`、以及 torchvision 的 `ToTensor()` 全部会崩，数据加载第一步就挂。必须 `pip install "numpy<2"`（会装成 1.26.4）。pandas / matplotlib 均兼容 1.26.4，无需改动。

### 2.4 最终版本

| 包 | 版本 |
|---|---|
| python | 3.10.18 |
| torch | 2.2.2 |
| torchvision | 0.17.2 |
| numpy | **1.26.4**（必须 <2） |
| pandas | 2.3.3 |
| matplotlib | 3.10.9 |
| tqdm | 4.70.0 |

磁盘占用约 1.5 GB。

### 2.5 验证安装

```bash
python -c "
import numpy as np, torch, torchvision
from torchvision import transforms
from PIL import Image
print('torch', torch.__version__, '| tv', torchvision.__version__, '| np', np.__version__)
print('mkldnn:', torch.backends.mkldnn.is_available(), '| threads:', torch.get_num_threads())
print('bridge:', torch.randn(2,3).numpy().shape,
      transforms.ToTensor()(Image.fromarray(np.zeros((28,28),'uint8'))).shape)
"
```

预期输出关键项：`mkldnn: True`，且 `bridge` 一行不报 NumPy 警告。

---

## 3. 数据集：MNIST

### 3.1 基本信息

| 项 | 值 |
|---|---|
| 训练集 / 测试集 | 60,000 / 10,000 |
| 尺寸 | **28 × 28**，单通道灰度 |
| 类别 | 10（数字 0–9） |
| 磁盘占用 | 63 MB（含 .gz 与解压后的 idx 文件） |
| 归一化统计量 | `mean=0.1307`，`std=0.3081`（本机实测，与公认值一致） |
| 类别分布 | 5421 ~ 6742，基本均衡 → 可直接用 accuracy 作为指标 |

### 3.2 下载方式（推荐：torchvision 自动下载）

```python
from torchvision import datasets, transforms

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train = datasets.MNIST('./data', train=True,  download=True, transform=tf)
test  = datasets.MNIST('./data', train=False, download=True, transform=tf)
```

首次运行自动下载并校验 MD5，实测耗时 **24.8 秒**。数据落在 `./data/MNIST/raw/`。

### 3.3 下载地址与镜像

torchvision 0.17.2 内置两个镜像，按顺序尝试：

| 优先级 | 地址 | 实测状态 |
|---|---|---|
| 1 | `http://yann.lecun.com/exdb/mnist/` | ❌ **404**（LeCun 官网已下线原始文件） |
| 2 | `https://ossci-datasets.s3.amazonaws.com/mnist/` | ✅ 200（PyTorch 官方 AWS 镜像） |

**第一个地址失败是正常现象**，torchvision 会自动 fallback 到第二个，日志里会看到 `Failed to download (trying next): HTTP Error 404`，不影响结果。

四个文件及其 MD5：

| 文件 | MD5 |
|---|---|
| `train-images-idx3-ubyte.gz` | `f68b3c2dcbeaaa9fbdd348bbdeb94873` |
| `train-labels-idx1-ubyte.gz` | `d53e105ee54ea40749a09fcbcd1e9432` |
| `t10k-images-idx3-ubyte.gz` | `9fb629c4189551a2d022fa330f9573f3` |
| `t10k-labels-idx1-ubyte.gz` | `ec29112dd5afa0611ce80d1b7f02629c` |

**手动下载**（自动下载失败时备用）：

```bash
mkdir -p data/MNIST/raw && cd data/MNIST/raw
BASE=https://ossci-datasets.s3.amazonaws.com/mnist
for f in train-images-idx3 train-labels-idx1 t10k-images-idx3 t10k-labels-idx1; do
  curl -LO "$BASE/${f}-ubyte.gz"
done
gunzip -k *.gz
```

放好后调用时用 `download=False` 即可。

**其他备选源**：Kaggle (`datasets/hojjatk/mnist-dataset`)、HuggingFace (`datasets/ylecun/mnist`)。

> `data/` 已被仓库根 `.gitignore` 忽略（规则 `machine-learning-and-deep-learning/homework-*/data`），不会误提交。

### 3.4 查看原始数据：ubyte → 图片

`data/MNIST/raw/` 下的 `*-ubyte` 是 **IDX 格式**，一种极简二进制容器。`export_images.py` 自己解析它并导出成可直接打开的 png —— 不走 torchvision，因为 `datasets.MNIST` 会把这一层完全藏起来。

IDX 的结构就是「短头部 + 裸字节」：

| 文件 | 头部 | 之后 |
|---|---|---|
| `*-images-idx3-ubyte` | 16 字节：magic `2051` / 张数 / 行数 `28` / 列数 `28` | 每字节一个像素（0=黑 255=白），行优先铺开 |
| `*-labels-idx1-ubyte` | 8 字节：magic `2049` / 标签个数 | 每字节一个标签（0–9） |

没有分隔符、没有压缩、没有元数据。两个细节：magic number 的**最后一字节就是维度数**（3=图片三维，1=标签一维）；所有整数是**大端序**，所以解析要用 `>I` 而不是本机默认的小端。

```bash
python export_images.py                             # 测试集前 100 张 → 一张总览图
python export_images.py --split train -n 64         # 换训练集
python export_images.py --digit 7 -n 36 --cols 6    # 只看某个数字
python export_images.py --mode files -n 50          # 每张单独存 png
python export_images.py --mode both -n 100          # 两者都要
open results/mnist_png                              # macOS 下直接看
```

两种模式：

- `grid`（默认）拼成一张总览图，另存一份**排列完全一致**的标签 txt。标签不写在图上是为了避免引入中文/字体依赖。
- `files` 每张单独存成 `00015_label7.png`，文件名带原始下标和标签，方便回头对照数据集。

放大用 `Image.NEAREST` 而不是默认的双线性 —— 28×28 要看清像素本身，不能被插值抹平。

---

## 4. 性能基准与输入尺寸决策

### 4.1 为什么不用 224×224

AlexNet/VGG/GoogLeNet/ResNet 原版都是为 **224×224 ImageNet** 设计的，下采样倍数均为 32（224 → 7）。实测在本机 CPU 上放大到 224 的代价：

| 模型 | 输入 | s/batch | 单 epoch |
|---|---|---|---|
| AlexNet 风格 | 28×28 | 0.068 | 64 秒 |
| AlexNet 风格 | 64×64 | 0.151 | 2.4 分 |
| AlexNet 风格 | 224×224 | 2.002 | **31.3 分** |

VGG 计算量是 AlexNet 的 5～10 倍 → 单 epoch 需 3～5 小时。四网络跑完 10+ 小时，本机不现实（那种情况需挂 AutoDL 远程 GPU）。

### 4.2 选定：28×28 原生尺寸

对比 28 与 32 的实测（同一批网络）：28×28 比 32×32 **省 31% 算力**，且 MNIST 原生即 28×28，不做 resize/padding 就没有插值失真。故采用 **28×28**。

代价：无法直接加载 ImageNet 预训练权重。但 MNIST 从头训练即可达 99%+，无影响。

### 4.3 架构适配原则

四个网络的下采样层必须改造（原版 32 倍会让 28×28 塌缩为 0），但**架构核心思想全部保留**：

| 网络 | 保留的架构特征 |
|---|---|
| AlexNet | 5 层卷积堆叠、大卷积核（5×5）、大 FC 头 + Dropout |
| VGG | 全部 3×3 小核，按块堆叠加深 |
| GoogLeNet | Inception 模块：1×1 / 3×3 / 5×5 / pool 四分支并行后拼接，1×1 降维 |
| ResNet | BasicBlock 残差捷径，让梯度直通 |

### 4.4 最终基准（batch=64，含 forward + backward + step）

| 模型 | 参数量 | s/batch | 单 epoch |
|---|---|---|---|
| AlexNet | 4.82M | 0.378 | 5.9 分 |
| VGG | 1.13M | 0.095 | 1.5 分 |
| GoogLeNet | 0.27M | 0.279 | 4.4 分 |
| ResNet | 2.76M | 0.330 | 5.1 分 |
| **四网络合计** | | | **16.9 分/epoch** |

→ 每个网络 5 epochs，全部跑完约 **1.4 小时**。

两个调优记录：
- **ResNet 21.9 分 → 5.1 分（4.3×）**：原先照搬 CIFAR 惯例保持全分辨率 stem，前两个残差块在 28×28 全尺寸、64 通道下硬算，对 MNIST 是纯浪费。改为 stem 直接 28→14 下采样。
- **AlexNet 15.30M → 4.82M 参数**：原先 83% 的参数堆在 `256×7×7 → 1024` 全连接层，多加一次池化即消掉 10M 冗余。

---

## 5. 一个必须处理的崩溃：OpenMP 嵌套 fork 导致段错误

单进程连续训练四个模型（`train.py --model all`）时进程直接挂掉：

```
[1]  9878 segmentation fault  python train.py --model all --epochs 5 --tag baseline
```

而且 `| tee logs/baseline.log` 存下来的日志是**空文件**，一行都没有。

### 5.1 为什么日志是空的

stdout 接到管道时 Python 用的是**块缓冲**（攒满 4 KB 才写一次），只有直连 tty 才是行缓冲。段错误是 `SIGSEGV`，进程被内核立即杀死，缓冲区里的内容还没 flush 就随进程一起消失了。

修复：`python -u` 关闭缓冲。**排查崩溃时这一步必须先做**，否则等于闭着眼睛调试。

### 5.2 崩溃根因

macOS 会把崩溃报告写到 `~/Library/Logs/DiagnosticReports/`。读崩溃线程的调用栈：

```
Exception Type:  EXC_BAD_ACCESS (SIGSEGV) / EXC_I386_GPFLT

Thread 0 Crashed:
0    libtorch_cpu.dylib   .omp_outlined..79
1    libiomp5.dylib       __kmp_invoke_microtask
2    libiomp5.dylib       __kmp_invoke_task_func
3    libiomp5.dylib       __kmp_fork_call
4    libiomp5.dylib       __kmp_fork_call
...                       ← 共 500+ 层同样的帧
```

`__kmp_fork_call` 递归 500 多层 —— **OpenMP 并行区嵌套 fork 失控**。

成因是两级并行叠乘。torch 在 CPU 上有两套线程池：

| 类型 | 作用 | 本机默认值 |
|---|---|---|
| intra-op（OMP） | 单个算子内部的数据并行，如一个 conv 拆给多线程算 | 4（= 物理核数） |
| inter-op | 算子之间的任务并行，如多分支模型的并行分支 | **8**（= 逻辑核数） |

inter-op 的 8 个线程各自派发算子，每个算子内部再开一个 4 线程的 OMP 并行区。而 `max_active_levels` 默认不加限制，libiomp5 就允许并行区里再开并行区、再开并行区……一层层 fork 下去。主线程栈只有 8 MB（`ulimit -s` → 8192），500 层撑爆栈空间 → `EXC_BAD_ACCESS`。

只训 VGG 一个模型时没触发，连着构建并训练 4 个模型才踩到 —— 这类竞态型崩溃有偶发性，不是每次都必崩。

### 5.3 三层修复

**第一层：限制 OpenMP 活动层级**（`train.py` 顶部，必须在 `import torch` 之前）

```python
import os
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
os.environ.setdefault("KMP_MAX_ACTIVE_LEVELS", "1")
```

`import torch` 会加载 libiomp5 并初始化 OpenMP 运行时，之后再改这些变量就无效了，所以顺序不能颠倒。设为 1 表示只允许一层并行区，嵌套请求退化为串行执行。

> 注意别用 `OMP_NESTED=FALSE`，该变量已废弃，设置它只会打印一行 `OMP: Info #269` 噪音。

**第二层：inter-op 线程降到 1**

```python
torch.set_num_interop_threads(1)
```

从源头掐掉嵌套：没有 inter-op 并行，就不会出现"并行区里再开并行区"。本次对比的四个网络都是单分支串行结构（GoogLeNet 的 Inception 四分支在 PyTorch 里也是顺序执行再 `cat`），inter-op 本来就无事可做，置 1 不损失性能。

**第三层：分进程训练**（`run_all.sh`）

每个模型开一个独立进程，OMP 运行时每次从干净状态起步，且单个模型崩溃不会连带丢掉其他三个已完成的结果。这一层不只是防崩 —— 1.4 小时的任务把鸡蛋放在一个进程里本身就不划算。

**附带**：`train.py` 里 `faulthandler.enable()`，下次再崩会打出 Python 层调用栈，而不是只留一句 `segmentation fault`。

### 5.4 修复后 GoogLeNet 仍然崩：一个真正的间发性竞态

上面三层改完，AlexNet（99.26%）和 VGG（99.29%）顺利跑完 5 epochs，**GoogLeNet 仍然崩**。但这次 `faulthandler` 起作用了，直接指到了出事的那一行：

```
Fatal Python error: Segmentation fault

Current thread 0x000000011bd58e00 (most recent call first):
  File ".../torch/autograd/__init__.py", line 266 in backward
  File ".../torch/_tensor.py", line 522 in backward
  File ".../train.py", line 135 in train_one      ← loss.backward()
```

#### 逐个算子压测：排除"某个算子有 bug"

GoogLeNet 相比另外三个网络的独特之处是 Inception 的四分支结构，带来一批不常见的算子和通道数。把它们拆开单独压测（`probe_crash.py`）：

| 测试项 | 迭代数 | 结果 | 耗时 |
|---|---|---|---|
| `MaxPool2d(3, 1, 1)` 单独 | 2000 | ✅ 通过 | 28s |
| `MaxPool2d(3,1,1)` + 1×1 conv（第四分支） | 2000 | ✅ 通过 | 38s |
| `conv_bn(64, 8, 1, 1, 0)`（极窄 1×1 降维） | 2000 | ✅ 通过 | 10s |
| `conv_bn(8, 16, 5, 1, 2)`（5×5 分支） | 2000 | ✅ 通过 | 10s |
| 四分支 `torch.cat` | 2000 | ✅ 通过 | 63s |
| `Inception(64, ...)` 完整模块 | 2000 | ✅ 通过 | 186s |
| `Inception(128, ...)` 完整模块 | 2000 | ✅ 通过 | 289s |
| `Inception(224, ...)` 完整模块 | 2000 | ✅ 通过 | 106s |
| **完整 GoogLeNet** | 400 | ❌ **崩** | — |

**结论：不是某个算子的 bug。** 单个模块跑 2000 次迭代都没事，只有完整网络会崩 —— 需要足够多的 OMP 并行区反复创建/销毁才能撞上竞态。这也解释了为什么冒烟测试（20 batch）能过、跑到 400 batch 才崩。

> 有个坑：最初压测时每个 case 只跑 3 次迭代，全部"通过"，是**假阴性** —— 崩溃间隔在 400 次迭代量级，3 次根本测不出来。隔离测试的迭代数必须和故障间隔同一量级。
>
> 另一个坑：`MaxPool2d` 单独测时模型没有可训练参数，`backward()` 直接抛 `RuntimeError: element 0 of tensors does not require grad`，被脚本误判成"崩溃"。要让输入 `requires_grad=True` 才测到真正想测的东西。

#### 崩溃频率决定了修复策略

实测崩溃概率约 **每 400 次迭代 50%**。而一个 epoch 是 938 个 batch —— **故障间隔短于一个 epoch**。这一点很关键：如果只按 epoch 存断点，那么每次都在 epoch 中途崩掉，断点永远存不下第一个，重试等于从零开始，永远推进不了。

所以两件事都得做：

**① 每个 epoch 存断点 + 自动续训**（`train.py`）

把损失从"丢掉整个模型的训练成果"限制为"最多丢一个 epoch"：

```python
torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
            "sched": ..., "rng": torch.get_rng_state(),
            "epoch": ep, "history": history, ...}, ckpt_path)
```

其中 `torch.get_rng_state()` 是必须存的：不恢复 RNG 状态，续训后的数据 shuffle 顺序就和不中断跑完的一次不一样，结果不可复现。实测恢复后续训的 loss 与不中断跑完**逐位相同**，说明这一步做对了。

`run_all.sh` 配合做自动重试，日志用 `tee -a` 追加以免重试覆盖历史记录。

**② 只给 GoogLeNet 降 OMP 线程数**

崩溃发生在 OMP 并行区内，线程少了竞态窗口就小甚至消失。`--threads 2` 只作用于 GoogLeNet，另外三个网络保持 4 线程全速：

```bash
THREADS_GOOGLENET=2 bash run_all.sh 5 baseline   # 仍崩就换 1
```

代价是 GoogLeNet 变慢，但这是本机唯一能稳定跑完的路子 —— 根因在 torch 2.2.2 的 CPU 后端，应用层改不了。

> 换个环境就没这些事：这批坑全部源自 **Intel Mac 只能用 torch 2.2.2（最后一个 x86_64 macOS wheel）**。挂 AutoDL 之类的远程 GPU 用新版 torch + CUDA，四个网络单 epoch 都在秒级，也不会碰到 libiomp5 的这个竞态。

---

## 6. 实验结果

完整的配置、命令、事先假设与逐条分析都在 `experiments.md`，这里只放结论。

### 6.1 实验一：四网络架构对比

同一数据集、同一超参数（5 epochs / batch 64 / lr 0.01 / SGD+momentum 0.9 / 无调度 / 无增强 / seed 42）：

| 模型 | 参数量 | 最佳测试准确率 | 末轮准确率 | 单 epoch 耗时 | 总耗时 |
|---|---|---|---|---|---|
| AlexNet | 4.82M | 99.26% | 99.22% | 421s | 35.1 分 |
| VGG | 1.13M | 99.29% | 99.19% | **117s** | **9.7 分** |
| GoogLeNet | **0.27M** | 99.36% | 99.36% | 355s | 29.6 分 |
| ResNet | 2.76M | **99.37%** | 99.24% | 315s | 26.3 分 |

**四者准确率只差 0.11 个百分点，但参数量差 18 倍、耗时差 3.6 倍。** MNIST 太简单，架构差异体现不出来 —— 换句话说，用它来比较四种架构的「能力上限」是无效的，能比出来的只有成本。

### 6.2 实验二：超参数单变量对照（VGG，5 epochs）

固定用 VGG 因为它最快（117s/epoch），6 组约 1 小时；换 AlexNet 要 4 小时以上。每组只改一个变量，否则准确率变化无法归因。

| 配置 | 改动 | 最佳准确率 | 相对基准 | gap（train−test） |
|---|---|---|---|---|
| `baseline` | 基准 | 99.29% | — | +0.54 |
| `lr001` | lr=0.001 | 99.32% | +0.03 | +0.29 |
| `lr1` | lr=0.1 | 99.32% | +0.03 | +0.33 |
| `adam` | Adam, lr=1e-3 | 99.09% | **−0.20** | +0.32 |
| `bs256` | batch=256 | 99.22% | −0.07 | **+0.65** |
| `cosine` | 余弦退火 | **99.49%** | **+0.20** | +0.42 |
| `augment` | 旋转10°+平移10% | 99.47% | +0.18 | **−0.42** |

**前提：多数差异在噪声范围内。** 测试集 10000 张，0.20pp 就是 20 张图，且只跑了一个 seed。`lr001`/`lr1` 的 +0.03（3 张图）不能算「更好」。

四条能站住的结论：

1. **有 BN 时学习率不敏感。** lr 在 0.001–0.1 这 100 倍区间内几乎不影响最终结果，预期的「大 lr 发散」没有出现 —— BN 把每层输入重新归一化，压掉了梯度的尺度差异。不值得在 lr 上做网格搜索。
2. **CPU 上调大 batch 没有收益。** `bs256` 只快 4%（112s vs 117s），因为 batch=64 已经足够喂饱 4 个物理核；同时过拟合最严重（gap +0.65）—— 大 batch 梯度噪声更小，而 SGD 的梯度噪声本身就是一种正则化。
3. **Adam 不是「换上就更好」的直接替代品。** 唯一显著变差的一组。这里比的是「未调参的 Adam」和「已调参的 SGD」，公平地说 Adam 需要重调 lr。
4. **数据增强的效果和预期相反。** 我预判「5 epochs 太短，增强来不及体现收益」，实际 +0.18 排第二。错在以为增强的收益需要长时间累积，忽略了它是**即时**的 —— 每个 batch 都是新的扰动样本，第一个 epoch 就在起作用。最有说服力的是 `augment` 是唯一负 gap 的配置（train 99.05% 反而低于 test 99.47%），这正是增强的特征信号。

### 6.3 实验三：20 epochs 下的 cosine × augment 因子实验

实验二在 5 epochs 下留了两个待验证方向：轮数够不够、两者能否叠加。

**为什么不能只跑一条 `--epochs 20 --scheduler cosine --augment`**：那样同时改了轮数、调度、增强三样，涨了多少都无法归因。所以做成 2×2 因子设计，其中 `e20_base` 是 20 epochs 下**自己的**基准 —— 没有它就分不清 `e20_cos` 的变化是 cosine 的功劳还是多跑 15 轮的功劳。

| 配置 | scheduler | augment | 最佳准确率 | 相对 `e20_base` | 最佳轮 | 末轮 | gap | 末轮训练损失 |
|---|---|---|---|---|---|---|---|---|
| `e20_base` | none | off | 99.53% | — | e15 | 99.51% | +0.49 | 0.0002 |
| `e20_cos` | cosine | off | 99.49% | −0.04 | e9 | 99.46% | +0.54 | 0.0003 |
| `e20_aug` | none | on | 99.57% | +0.04 | e13 | 99.47% | +0.08 | 0.0134 |
| `e20_cosaug` | cosine | on | **99.67%** | **+0.14** | e16 | **99.65%** | **+0.03** | 0.0098 |

#### ❌ 方向一判断错了：轮数拉长后，两个单项的优势全部消失

| 相对基准的优势 | 5 epochs | 20 epochs |
|---|---|---|
| cosine | **+0.20** | **−0.04** |
| augment | **+0.18** | **+0.04** |

我预期「延长到 20 epochs 会拉开更大差距」，**实际是差距被抹平了**。

错在把「末轮即最佳轮」读成了「这个配置还有上升空间」，但真正在上升期的**是基准自己** —— 基准从 99.29% 涨到 99.53%（+0.24），涨幅超过 cosine 和 augment 在 5 轮时的全部优势。也就是说，5 轮实验里那 +0.2pp **不是「更高的天花板」，而只是「更快到达同一个天花板」**。

`e20_cos` 最能说明问题：最佳轮出现在 **e9**，之后 11 轮里 lr 一路降到 6e-5，准确率再没超过 e9。原因在训练损失那一列 —— e11 时 train_acc 已经 **100.00%**、损失 0.0005。**训练集被完全记住之后，调度器就无事可做了**：学习率再怎么精细退火，也只是在一个已经归零的损失面上挪动。

> 修正后的结论：**学习率调度的价值是「加速收敛」，不是「提高上限」。** 预算紧张时收益明显（5 轮 +0.20），预算充足时归零（20 轮 −0.04）。

#### ✅ 方向二判断对了，而且是超线性叠加

交互项 = `cosaug` − `cos` − `aug` + `base` = 99.67 − 99.49 − 99.57 + 99.53 = **+0.14**

- 单独 cosine：−0.04
- 单独 augment：+0.04
- 两者相加的预期：**0.00**（即「跟基准没区别」）
- 实际组合：**+0.14**

`e20_cosaug` 的提升**完全来自交互，而不是来自任何一项本身**。两个单独都没用的改动，凑在一起有用了。

机制上讲得通，关键就在上一条发现的那个「调度器无事可做」：

1. `augment` 每个 batch 都在生成新的扰动样本，训练集事实上被无限放大，模型**永远记不住** —— `e20_aug` 跑满 20 轮 train_acc 只有 99.56%，训练损失 0.0134 是 `e20_base`（0.0002）的 **82 倍**。
2. 损失面没有塌成零，**调度器才重新有了发挥空间** —— 后期小 lr 是在仍然崎岖的损失面上精修，而不是在平地上空转。
3. 反过来也成立：`augment` 单独用时曲线一直在抖（末轮比最佳轮低 0.10），因为增强带来的梯度噪声到最后也没衰减。**cosine 正好负责把这个噪声按下去** —— `e20_cosaug` 从 e11 起稳定在 99.6% 以上，末轮 99.65% 与最佳轮 99.67% 只差 0.02。

> **这是本次全部实验里最有价值的一条：两个超参数的效果不能各自测完再相加。**
> 只做单变量对照（实验二的做法），20 轮下会得到「cosine 没用、augment 没用」的结论，从而永远不会去试这个组合 —— 而它恰恰是全场最好的配置。
> 单变量对照能保证归因清晰，**代价是看不见交互**。

#### gap 那一列给出的独立佐证

注意 `e20_cos` 的 gap（+0.54）比基准（+0.49）还**略大**。这再次说明**学习率调度不是正则化手段** —— 它只影响优化路径怎么走，不影响模型能记住多少。真正压过拟合的是 augment（改变数据分布），gap 从 +0.49 直接砍到 +0.08。

5 轮时我看到 cosine gap +0.42、augment gap −0.42，据此判断「过拟合方向互补，理论上可叠加」。**结论蒙对了，但推理过程是错的** —— cosine 的 +0.42 不是「它偏向过拟合」，只是它收敛更快所以 train_acc 更早冲高；20 轮下基准也一样冲到 100%，两者 gap 就趋同了（+0.49 vs +0.54）。真正让叠加成立的不是 gap 互补，而是上面讲的「augment 保住了损失面的坡度」。

### 6.4 最终排名与实践结论

| 配置 | 轮数 | 最佳准确率 | 耗时 | 备注 |
|---|---|---|---|---|
| **`e20_cosaug`** | 20 | **99.67%** | 44 分 | **全场最佳**，末轮≈最佳轮，无需挑 checkpoint |
| `e20_aug` | 20 | 99.57% | 44 分 | |
| `e20_base` | 20 | 99.53% | 38 分 | 光靠多跑 15 轮就 +0.24 |
| `cosine`（5 轮） | 5 | 99.49% | **9.7 分** | **性价比最高** |
| ResNet（5 轮基准） | 5 | 99.37% | 26 分 | 参数量是 VGG 的 2.4 倍 |
| `baseline`（5 轮） | 5 | 99.29% | 9.7 分 | |

- **预算紧**（10 分钟以内）：用 `cosine`，5 轮 99.49%。调度器把收敛压缩到最少轮数。
- **要最好**（45 分钟）：用 `cosine + augment`，20 轮 99.67%。**必须两个一起开，单开任何一个都等于白跑。**

最终 99.67% 比四网络对比中最好的 ResNet（99.37%）高 **0.30pp**，而 VGG 参数量只有 ResNet 的 41%。**在这个任务上调超参数比换架构划算**，且这个结论在 20 轮下比 5 轮时更成立。

### 6.5 这批结果的可信度边界

必须说明：实验三的四组落在 99.49%–99.67%，极差 **0.18pp = 18 张图**，单 seed，没有误差棒。

交互项 +0.14（14 张图）之所以值得当真，不是因为幅度，而是因为**曲线形态支持它** —— `e20_cosaug` 从 e9 起就持续高于其余三组（99.60 / 99.64 / 99.64 / 99.67），是 12 个连续 epoch 的稳定领先，不是某一轮的运气。相比之下 `e20_aug` 的 +0.04 就只能算噪声。

要把 +0.14 坐实，需要跑 3–5 个 seed 看均值±标准差。按 44 分钟/组算，4 组 × 3 seed ≈ 8.8 小时，本机 CPU 上暂未做 —— **这是目前最大的短板。**

---

## 7. 文件说明

| 文件 | 用途 |
|---|---|
| `models.py` | 四种架构定义（适配 28×28），`python models.py` 可做形状自检 |
| `train.py` | 训练主脚本，所有超参数走命令行，便于调参对比 |
| `run_all.sh` | 四网络对比，每个模型一个独立进程（见 5.3） |
| `compare.py` | 汇总各模型 JSON → 对比表格 + markdown 表 + 三张图 |
| `sweep.sh` | 超参数单变量对照实验（固定 VGG，6 组配置） |
| `long.sh` | 20 epochs 下的 cosine × augment 2×2 因子实验（4 组，约 2.7 小时） |
| `sweep_compare.py` | 汇总超参数对照结果 → 对比表 + 两张图，`--base` / `--prefix` 可换基准与输出名 |
| `export_images.py` | 解析 IDX(ubyte) 原始文件 → 可查看的 png（见 3.4） |
| `benchmark.py` | 环境算力基准：matmul GFLOPS + 不同输入尺寸下的训练吞吐 |
| `probe_archs.py` | 四种架构适配到 28×28 后的形状与耗时探测 |
| `probe_crash.py` | GoogLeNet 段错误的逐算子隔离压测（见 5.4） |
| `experiments.md` | **所有实验的配置、命令、假设与逐条分析** |
| `homework.md` | 作业原始要求 |
| `prompts/` | AI Coding 过程的提示词记录 |
| `results/` | 训练结果 JSON、`summary.csv`、`figs/` 图表、`mnist_png/` 数据集图片 |
| `logs/` | 各次训练的完整终端日志 |

### 7.1 复现步骤

```bash
conda activate deep-learning-aura

# 环境基准（可选）
python benchmark.py
python probe_archs.py

# 看一眼原始数据（可选，几秒钟）
python export_images.py

# 实验一：四网络对比，约 1.4 小时
THREADS_GOOGLENET=2 bash run_all.sh 5 baseline
python compare.py --tag baseline          # 汇总出图

# 实验二：超参数单变量对照，约 1 小时（基准复用 baseline_vgg，不重跑）
bash sweep.sh 5

# 实验三：20 epochs 的 2×2 因子实验，约 2.7 小时
bash long.sh 20

# 单独重跑某一组
python -u train.py --model resnet --epochs 5 --tag baseline
```

三个脚本都会跳过已有结果的组，崩溃自动从断点续训（最多 3 次），跑完自动汇总出表和图。改完代码先冒烟测试再开长跑：

```bash
python -u train.py --model vgg --epochs 1 --limit-batches 3 --tag smoke --out /tmp/smoke
```

各组配置、命令与事先假设详见 `experiments.md`。

