# MNIST CNN 对比实验 · 复现手册

一份**从零到出结果**的线性操作手册。每一步都给出命令、预期输出、以及「怎么确认这步没错」。

三份文档的分工：

| 文档 | 回答什么问题 | 什么时候看 |
|---|---|---|
| **本文档** | *怎么一步步跑出来* | 想自己复现、或想加新实验时 |
| `README.md` | *结果是什么、为什么* | 想看结论和分析时 |
| `experiments.md` | *每组配置的事先假设 vs 实际* | 想知道「当初预期错在哪」时 |

---

## 0. 先看时间预算

纯 CPU（Intel i7-4770HQ，4 核）实测耗时。**先决定要跑到哪一步，再开始动手** —— 全跑完是 5 小时以上。

| 步骤 | 内容 | 耗时 | 必要性 |
|---|---|---|---|
| Step 1 | 环境搭建 | 10 分钟 | 必须 |
| Step 2 | 数据准备 | 1 分钟 | 必须 |
| Step 3 | 冒烟测试 | 30 秒 | **必须**（别跳过） |
| Step 4 | 实验一：四网络对比 | **1.7 小时** | 作业要求 1、2 |
| Step 5 | 实验二：超参数单变量对照 | **1 小时** | 作业要求 3 |
| Step 6 | 实验三：20 轮 2×2 因子实验 | **2.7 小时** | 加分项，结论最有价值 |
| | **合计** | **≈ 5.5 小时** | |

**只有 1 小时的裁剪方案**见 [§8](#8-时间不够时的裁剪方案)。

所有长跑步骤都支持**中断后从断点续训**，不必一次性跑完，见 [§6](#6-中断与恢复)。

---

## Step 1 · 环境搭建

### 1.1 建环境装包

```bash
conda create -n deep-learning-aura python=3.10 -y
conda activate deep-learning-aura

pip install torch==2.2.2 torchvision==0.17.2
pip install "numpy<2"
pip install matplotlib pandas tqdm
```

> 国内网络慢加镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`

**两个版本号必须写死，原因见 README 2.3。** 简单说：

- `torch==2.2.2` —— PyTorch 从 2.3 起不再发布 macOS x86_64 wheel，Intel Mac 上这是最高版本。
- `numpy<2` —— torch 2.2.2 是用 NumPy 1.x 编译的，装了 2.x 会让 `torch ↔ numpy` 的桥梁断掉，`ToTensor()` 直接崩，数据加载第一步就挂。

非 Intel Mac 的环境（Apple Silicon / Linux + CUDA）可以用新版 torch，不受这两条限制。

### 1.2 验证点 ✅

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

必须满足**三件事**，否则不要往下走：

- `mkldnn: True` —— 没有它卷积会慢好几倍
- `bridge` 那一行正常打印出两个 shape
- **全程没有 `Failed to initialize NumPy: _ARRAY_API not found` 警告** —— 出现这行说明 numpy 还是 2.x

### 1.3 可选：跑一下算力基准

```bash
python benchmark.py       # matmul GFLOPS + 不同输入尺寸的训练吞吐
python probe_archs.py     # 四种架构适配 28×28 后的形状与耗时
python models.py          # 四个网络的形状自检
```

`python models.py` 的预期输出，可用来确认架构定义没被改坏：

```
model             params      output
------------------------------------
alexnet            4.82M     (2, 10)
vgg                1.13M     (2, 10)
googlenet          0.27M     (2, 10)
resnet             2.76M     (2, 10)

四个网络形状自检通过
```

---

## Step 2 · 数据准备

### 2.1 自动下载

数据由 `train.py` 首次运行时自动下载，**不需要单独操作**。想提前下好：

```bash
python -c "
from torchvision import datasets
datasets.MNIST('./data', train=True,  download=True)
datasets.MNIST('./data', train=False, download=True)
print('done')
"
```

实测 24.8 秒，落在 `./data/MNIST/raw/`。

> **日志里出现 `HTTP Error 404` 是正常的**，不是故障。torchvision 内置两个镜像，第一个 `yann.lecun.com` 已下线，会自动 fallback 到第二个 AWS 镜像。手动下载方式见 README 3.3。

### 2.2 验证点 ✅

```bash
ls -la data/MNIST/raw/*-ubyte
```

应该有 4 个解压后的文件，大小分别约 47MB / 60KB / 7.8MB / 10KB。

### 2.3 可选：看一眼原始数据

不看数据就直接训练，是很容易踩坑的习惯 —— 至少确认一下标签和图像是对应的。

```bash
python export_images.py              # 测试集前 100 张 → 一张 10×10 总览图
open results/mnist_png               # macOS 直接看
```

预期输出（同时能验证 IDX 解析正确）：

```
解析 IDX 头部：
  t10k-images-idx3-ubyte: magic=2051 张数=10000 尺寸=28x28
  t10k-labels-idx1-ubyte: magic=2049 个数=10000
导出 100 张（test 集，原始下标 0..99）

总览图已保存 results/mnist_png/grid_test_100.png  （10 行 x 10 列，894x894px）
...
标签速览：
  7 2 1 0 4 1 4 9 5 9
  ...
```

**验证点** ✅：打开图，第一行确实是 `7 2 1 0 4 1 4 9 5 9`。magic number 对上（2051/2049）说明 IDX 头部解析正确。

其他用法：

```bash
python export_images.py --split train -n 64         # 换训练集
python export_images.py --digit 7 -n 36 --cols 6    # 只看某个数字
python export_images.py --mode files -n 50          # 每张单独存 png
```

---

## Step 3 · 冒烟测试（不要跳过）

**每 epoch 至少 2 分钟，四网络跑完 1.7 小时。** 在这之前必须用 3 个 batch 确认流程通不通 —— 30 秒的成本换掉「跑了 40 分钟才发现参数拼错」的风险。

```bash
python -u train.py --model vgg --epochs 1 --limit-batches 3 --tag smoke --out /tmp/smoke
```

预期输出：

```
torch 2.2.2 | intra-op=4 inter-op=1 | mkldnn=True | device=cpu
配置：{'model': 'vgg', 'epochs': 1, ...}

========================================================================
训练 vgg  |  参数量 1.13M  |  1 epochs x 3 batches
========================================================================
epoch    train_loss  train_acc  test_acc        lr     time
------------------------------------------------------------------------
1            2.2779     17.71%      9.80%   0.01000     7.7s
------------------------------------------------------------------------
vgg 完成：最佳测试准确率 9.80%  |  总耗时 0.1 分钟
```

**验证点** ✅：

- `inter-op=1` —— 防段错误的关键设置生效了（默认是 8，见 README 5.3）
- `mkldnn=True`
- 准确率 9.80% ≈ 随机猜（1/10）—— **这是对的**，只喂了 3 个 batch = 192 张图，模型什么都没学到
- 输出写到 `/tmp/smoke`，**不会污染 `results/`**

改完任何代码都先跑这条。换个模型验证：`--model googlenet`。

---

## Step 4 · 实验一：四网络架构对比

对应作业要求 1、2：搭建四种架构 + 对比识别效果。

### 4.1 跑

```bash
THREADS_GOOGLENET=2 bash run_all.sh 5 baseline
```

依次训练 alexnet → vgg → googlenet → resnet，每个 5 epochs，跑完自动汇总出图。**约 1.7 小时。**

三个设计要点：

| 做法 | 为什么 |
|---|---|
| 每个模型一个独立进程 | 单进程连训 4 个会踩 libiomp5 的 OpenMP 段错误，一崩全丢 |
| `THREADS_GOOGLENET=2` | GoogLeNet 的 backward 会间发性段错误，降线程能缩小 OMP 竞态窗口 |
| 判成功看结果文件而非退出码 | 训练完成落盘后，解释器退出阶段的 GC 仍可能崩（退出码 139）。只看退出码会把跑完的模型判成失败并重训 |

完整的崩溃排查过程见 README 第 5 节 —— 那是本次作业里最值得读的部分。

### 4.2 中途看进度

长跑期间另开一个终端：

```bash
tail -f logs/baseline_googlenet.log        # 跟某个模型的实时日志
grep -E '^[0-9]+ ' logs/baseline_*.log     # 只看已完成的 epoch 行
ls results/baseline_*.json                 # 已完成几个模型
```

### 4.3 验证点 ✅

跑完应该有 4 个 JSON，且准确率都在 99.2%–99.4%：

```bash
python compare.py --tag baseline    # 单独重新汇总（run_all.sh 已自动调过）
column -s, -t results/summary.csv
```

预期结果（差个 ±0.1pp 属正常波动）：

| 模型 | 参数量 | 最佳准确率 | 单 epoch |
|---|---|---|---|
| alexnet | 4.82M | 99.26% | 421s |
| vgg | 1.13M | 99.29% | 117s |
| googlenet | 0.27M | 99.36% | 355s |
| resnet | 2.76M | 99.37% | 315s |

产出物：`results/baseline_{model}.json`、`results/baseline_table.md`、`results/figs/baseline_{curves,per_class,tradeoff}.png`

### 4.4 单独重跑某一个

```bash
python -u train.py --model resnet --epochs 5 --tag baseline
```

已有结果的模型 `run_all.sh` 会自动跳过；想强制重跑加 `FORCE=1`。

---

## Step 5 · 实验二：超参数单变量对照

对应作业要求 3：调整超参数并观察准确率变化。

### 5.1 跑

```bash
bash sweep.sh 5
```

6 组配置，**约 1 小时**。基准直接复用 Step 4 的 `results/baseline_vgg.json`，不重跑。

两个设计决策，值得先理解再跑：

**为什么固定用 VGG？** 它 117s/epoch，是四个网络里最快的，6 组约 1 小时；同样的实验换 AlexNet 要 4 小时以上。超参数的作用规律在各架构间是通用的，不必每个网络都重跑。

**为什么每组只改一个变量？** 同时改多个的话，准确率的变化无法归因到具体哪一项。**但要注意这个做法的代价 —— 它看不见交互效应，Step 6 就是被这一点绊倒的。**

6 组配置：

| 标签 | 改动 | 想验证什么 |
|---|---|---|
| `lr001` | lr 0.01 → 0.001 | 学习率过小是否欠拟合 |
| `lr1` | lr 0.01 → 0.1 | 学习率过大是否震荡/发散 |
| `adam` | SGD → Adam(lr=1e-3) | 自适应优化器 |
| `bs256` | batch 64 → 256 | 大 batch 的速度与泛化 |
| `cosine` | 加余弦退火 | 学习率调度 |
| `augment` | 随机旋转 10° + 平移 10% | 数据增强的正则化 |

### 5.2 验证点 ✅

```bash
python -u sweep_compare.py --model vgg
```

预期：7 组全部落在 **99.09%–99.49%**，其中 `cosine` 最高（99.49%）、`adam` 最低（99.09%）。

**注意一个自检信号**：`cosine` 的 e1 应该与 `baseline` 的 e1 **完全一致**（都是 `98.76 / 0.1037`）—— 因为余弦退火第一个 epoch 的 lr 就是 0.01，且 seed 相同。如果这两个数不一样，说明随机性没控制住，各组之间的差异掺了数据顺序的噪声，实验作废。

产出物：`results/sweep_vgg_table.md`、`results/figs/sweep_vgg_{curves,delta}.png`

### 5.3 逐组单独跑

```bash
python -u train.py --model vgg --epochs 5 --lr 0.001                   --tag lr001
python -u train.py --model vgg --epochs 5 --lr 0.1                     --tag lr1
python -u train.py --model vgg --epochs 5 --optimizer adam --lr 0.001  --tag adam
python -u train.py --model vgg --epochs 5 --batch-size 256             --tag bs256
python -u train.py --model vgg --epochs 5 --scheduler cosine           --tag cosine
python -u train.py --model vgg --epochs 5 --augment                    --tag augment
```

---

## Step 6 · 实验三：20 轮 2×2 因子实验

Step 5 在 5 epochs 下留了两个待验证方向：**轮数够不够**、**cosine + augment 能否叠加**。

### 6.1 为什么不能只跑一条命令

直觉上会想这么干：

```bash
# ❌ 不要这样
python -u train.py --model vgg --epochs 20 --scheduler cosine --augment --tag best20
```

这条命令同时改了**轮数、调度、增强**三样。假设它跑出 99.67%，比 5 轮基准的 99.29% 高 0.38pp —— 但这 0.38 里有多少来自多跑的 15 轮？多少来自 cosine？多少来自 augment？**分不开。**

所以做成 2×2 因子设计，关键是 `e20_base`：**20 epochs 下自己的基准**。有了它才能把「轮数变多」和「加了 cosine/augment」分离。四组齐了还有额外收益 —— 能算交互项，直接回答「能否叠加」。

| tag | scheduler | augment | 作用 |
|---|---|---|---|
| `e20_base` | none | off | 2×2 的对照原点 |
| `e20_cos` | cosine | off | 只开调度 |
| `e20_aug` | none | on | 只开增强 |
| `e20_cosaug` | cosine | on | 两个都开 |

### 6.2 跑

```bash
bash long.sh 20 2>&1 | tee logs/long_run.log
```

**约 2.7 小时。** 怕终端断掉就挂后台：

```bash
nohup bash long.sh 20 > logs/long_run.log 2>&1 &
tail -f logs/long_run.log
```

中途看进度：

```bash
grep -E '^[0-9]+ ' logs/e20_*_vgg.log | tail -20
```

脚本里配置的顺序是 `e20_base` → `e20_cos` → `e20_aug` → `e20_cosaug`，**先跑基准侧** —— 万一时间不够中断了，前两组也已经能回答方向一。

### 6.3 验证点 ✅

```bash
python -u sweep_compare.py --model vgg --prefix long --base e20_base \
  --tags e20_base e20_cos e20_aug e20_cosaug
```

（`sweep_compare.py` 默认拿 `baseline` 当基准，这里必须用 `--base e20_base` 换掉，否则会把 20 轮的组跟 5 轮的基准比，白费功夫。`--prefix long` 是为了不覆盖 Step 5 的输出文件。）

预期：

| 配置 | 最佳准确率 | 相对 `e20_base` | gap |
|---|---|---|---|
| `e20_base` | 99.53% | — | +0.49 |
| `e20_cos` | 99.49% | −0.04 | +0.54 |
| `e20_aug` | 99.57% | +0.04 | +0.08 |
| `e20_cosaug` | **99.67%** | **+0.14** | **+0.03** |

### 6.4 算交互项

这是这组实验的核心产出，`sweep_compare.py` 不会自动算，手工算一下：

```bash
python - <<'EOF'
import json
g = lambda t: json.load(open(f"results/{t}_vgg.json"))["best_test_acc"]
b, c, a, ca = g("e20_base"), g("e20_cos"), g("e20_aug"), g("e20_cosaug")
print(f"单独 cosine   : {c-b:+.2f}")
print(f"单独 augment  : {a-b:+.2f}")
print(f"两者相加的预期 : {(c-b)+(a-b):+.2f}")
print(f"实际组合      : {ca-b:+.2f}")
print(f"交互项        : {ca-c-a+b:+.2f}   (正数 = 超线性叠加)")
EOF
```

预期输出：

```
单独 cosine   : -0.04
单独 augment  : +0.04
两者相加的预期 : +0.00
实际组合      : +0.14
交互项        : +0.14   (正数 = 超线性叠加)
```

**两个单独都没用的改动，凑在一起有用了。** 完整机制分析见 README 6.3。

---

## 6. 中断与恢复

长跑不必一次跑完。三个脚本都做了两层保护：

**① 每个 epoch 存断点**（`train.py`）

断点在 `results/ckpt_{tag}_{model}.pt`，包含 model / optimizer / scheduler / **RNG 状态** / history。崩溃或手动 Ctrl-C 后，重跑同一条命令会自动续训：

```
[续训] vgg 已完成 12/20 epochs，当前最佳 99.44%，从第 13 epoch 继续
```

存 RNG 状态是必须的 —— 不恢复它，续训后的数据 shuffle 顺序就和不中断跑完的一次不一样，结果不可复现。实测恢复后的 loss 与不中断跑完**逐位相同**。

正常跑完会自动删掉断点文件，避免下次同 tag 实验误用旧断点。

**② 已有结果自动跳过**

三个脚本都会检查 `results/{tag}_{model}.json`，存在就跳过。所以中断后直接重跑整条脚本命令即可，已完成的组不会白跑：

```bash
bash long.sh 20        # 会跳过已完成的组，从断的地方继续
FORCE=1 bash long.sh 20  # 强制全部重跑
```

**手动放弃某一组重来**：

```bash
rm results/e20_cos_vgg.json results/ckpt_e20_cos_vgg.pt
```

---

## 7. 故障排查速查表

| 现象 | 原因 | 处理 |
|---|---|---|
| `Failed to initialize NumPy: _ARRAY_API not found` | numpy 是 2.x | `pip install "numpy<2"` |
| `pip install torch` 解析失败 / 版本冲突 | Intel Mac 上 torch>2.2.2 无 wheel | 版本号写死 `torch==2.2.2 torchvision==0.17.2` |
| 下载数据报 `HTTP Error 404` | LeCun 官网已下线 | **正常现象**，会自动 fallback 到 AWS 镜像 |
| `segmentation fault`，**日志是空文件** | stdout 块缓冲，崩溃时缓冲区随进程丢失 | 命令加 `python -u`。排查崩溃时这一步必须先做 |
| GoogLeNet 反复段错误 | torch 2.2.2 CPU 后端的 OMP 竞态 | `THREADS_GOOGLENET=1 bash run_all.sh 5 baseline` |
| 退出码 139 但结果 JSON 已生成 | 解释器退出阶段 GC 崩溃 | **可以忽略**，脚本已按「结果文件是否存在」判成功 |
| `缺少 baseline 结果，无法做对照` | `sweep_compare.py` 找不到基准 | 先跑完 Step 4，或用 `--base` 指定别的基准 tag |
| 图里中文显示成方框 | 没找到中文字体 | 不影响数据，`compare.py:setup_cjk_font()` 里加本机字体路径 |
| 训练准确率一直在 10% 左右 | 学得动但没学到 / 数据或标签错位 | 先跑 Step 3 冒烟测试；`export_images.py` 确认图和标签对应 |

**排查崩溃的通用顺序**：`python -u` 拿到日志 → 看 `faulthandler` 打出的 Python 栈 → macOS 的 `~/Library/Logs/DiagnosticReports/` 看原生栈 → 逐算子隔离压测（`probe_crash.py` 是个例子）。

> 隔离压测有个陷阱：迭代数必须和故障间隔同一量级。本次崩溃间隔在 400 次迭代量级，最初每个 case 只跑 3 次，全部「通过」—— 是假阴性。

---

## 8. 时间不够时的裁剪方案

| 预算 | 跑什么 | 能回答什么 | 放弃什么 |
|---|---|---|---|
| **10 分钟** | Step 1–3 + `python -u train.py --model vgg --epochs 5 --scheduler cosine --tag cosine` | 单个模型能到 99.49% | 所有对比 |
| **30 分钟** | 上面 + `--tag augment`、`--tag baseline` | 两个超参数各自的效果 | 架构对比、20 轮验证 |
| **1 小时** | Step 1–3 + `bash sweep.sh 5`（需先有 baseline_vgg） | **作业要求 3 完整** | 架构对比 |
| **2 小时** | Step 1–4 | **作业要求 1、2 完整** | 超参数实验 |
| **3 小时** | Step 1–5 | 作业要求 1、2、3 全部 | 交互效应这个最有价值的发现 |
| **5.5 小时** | 全部 | 全部 | — |

**另一条省时思路：把 epochs 减半。** 结论的方向不会变，但注意 Step 6 的结论**依赖于 20 轮这个量级** —— 因为它的核心机制是「基准在 20 轮时 train_acc 冲到 100%，调度器无事可做」。10 轮下基准还没完全记住训练集，交互项会小很多。

---

## 9. 加一组自己的实验

### 9.1 单独跑一个新配置

所有超参数都走命令行，改配置不需要动代码：

```bash
python -u train.py --model vgg --epochs 10 --weight-decay 1e-4 --tag wd1e4
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | all | `alexnet` / `vgg` / `googlenet` / `resnet` / `all` |
| `--epochs` | 5 | |
| `--batch-size` | 64 | |
| `--lr` | 0.01 | |
| `--optimizer` | sgd | `sgd` / `adam` |
| `--momentum` | 0.9 | 仅 SGD |
| `--weight-decay` | 0.0 | L2 正则 |
| `--scheduler` | none | `none` / `cosine` / `step` |
| `--augment` | 关 | 随机旋转 10° + 平移 10% |
| `--seed` | 42 | 各模型用同一 seed 起步，保证对比公平 |
| `--tag` | baseline | 实验名，用于结果文件命名，**必须唯一** |
| `--threads` | 0 | intra-op(OMP) 线程数，0=默认 4。调小可规避段错误 |
| `--resume` | 开 | 有断点则续训，`--no-resume` 强制从头 |
| `--limit-batches` | 0 | 每 epoch 只跑前 N 个 batch，冒烟测试用 |

### 9.2 加进批量脚本

以 `sweep.sh` 为例，在 `CONFIGS` 数组里加一行，格式是 `"标签|额外参数"`：

```bash
CONFIGS=(
  "lr001|--lr 0.001"
  ...
  "wd1e4|--weight-decay 1e-4"      # ← 新增
)
```

标签会同时用作结果文件名，**必须唯一**，否则会覆盖别人的结果。然后把它加进汇总的 tags 列表：

```bash
python -u sweep_compare.py --model vgg --tags baseline lr001 lr1 adam bs256 cosine augment wd1e4
```

「改动」那一列是把每组的 hparams 与基准逐项 diff 出来的，不是写死的文字 —— 所以加了新配置不用改 `sweep_compare.py`，也不会出现描述与实际不符。

### 9.3 设计新实验时的三条经验

这三条都是本次实验踩出来的，值得先看再动手：

**① 一次只改一个变量，但要意识到它看不见交互。**
Step 5 是严格的单变量对照，结论清晰。但如果只做到这一步，20 轮下会得到「cosine 没用、augment 没用」，从而永远不会去试那个恰恰是全场最好的组合。**做完单变量对照后，把「机制上不冲突」的几项配对再测一次。**

**② 改了轮数就必须重跑基准。**
拿 20 轮的实验组去比 5 轮的基准，涨的部分无法归因。这是 Step 6 里 `e20_base` 存在的全部理由。同理，改了 batch size 也要想清楚基准是什么。

**③ 判断「有没有效果」要看幅度 + 曲线形态，不能只看一个数。**
测试集 10000 张，0.20pp 就是 20 张图，单 seed 下这个量级基本是噪声。本次的判据是：**幅度 ≥ 0.2pp，且有机制上的解释、曲线形态能对上**。Step 6 的交互项 +0.14 之所以值得当真，不是因为幅度，而是因为 `e20_cosaug` 是 12 个连续 epoch 的稳定领先，不是某一轮的运气。

---

## 10. 复现性检查清单

想让自己的结果可比、可复现，这几条要都做到：

- [ ] **`--seed` 保持一致**（默认 42）。`train_one()` 里每个模型训练前都重设 seed，保证各组从同一起点出发。
- [ ] **续训后 RNG 状态要恢复**（`train.py` 已做）。否则 shuffle 顺序变了，结果和不中断跑完不一样。
- [ ] **归一化只用训练集统计量**（`mean=0.1307, std=0.3081`）。用测试集的统计量归一化是数据泄露 —— 那属于「未来信息」。
- [ ] **对比组之间只差要研究的那一项**。用 `sweep_compare.py` 输出的「改动」列自动 diff 核对，别靠记忆。
- [ ] **检查那个自检信号**：`cosine` 的 e1 应与 `baseline` 的 e1 逐位相同（见 5.2）。不同就说明随机性没控住。
- [ ] **报告结论时说明 seed 数量**。本次全部实验都是**单 seed**，这是最大的短板 —— 0.1–0.2pp 级别的差异没有误差棒支撑。要坐实需要 3–5 个 seed 看均值±标准差（4 组 × 3 seed ≈ 8.8 小时）。

---

## 11. 产出物清单

跑完全部步骤后，`results/` 下应该有：

| 路径 | 内容 |
|---|---|
| `{tag}_{model}.json` | 每组实验的完整记录：逐 epoch history、每类准确率、参数量、全部 hparams |
| `summary.csv` | 所有实验的一行摘要，追加写入，可直接用 pandas 读 |
| `baseline_table.md` | 四网络对比表（可直接贴进 README） |
| `sweep_vgg_table.md` | 超参数对照表 |
| `long_vgg_table.md` | 20 轮因子实验表 |
| `figs/baseline_curves.png` | 四网络收敛曲线 |
| `figs/baseline_per_class.png` | 各类别准确率 |
| `figs/baseline_tradeoff.png` | 参数量 / 耗时 / 准确率的权衡 |
| `figs/sweep_vgg_curves.png` | 各超参数配置的曲线（基准为粗黑线） |
| `figs/sweep_vgg_delta.png` | 相对基准的准确率变化柱状图 |
| `figs/long_vgg_{curves,delta}.png` | 20 轮因子实验的对应两张图 |
| `mnist_png/` | 数据集导出的图片 |

`logs/{tag}_{model}.log` 是各次训练的完整终端日志，用 `tee -a` 追加写入，重试不会覆盖历史记录。
