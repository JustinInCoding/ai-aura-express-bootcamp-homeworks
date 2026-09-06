# GPU 移植计划

**目标**：把 CPU 版的三组 MNIST 实验原样搬到云 GPU 重跑，网络结构与实验设计一字不改，只换执行环境。

**状态**：代码移植已完成，待上机验证。执行步骤见第 5 节。

---

## 1. 为什么要做这件事

CPU 版跑完三组实验共 ~2.5 小时（i7-4770HQ，纯 CPU）。这带来两个问题：

1. **迭代慢**。改一个超参数要等两小时才知道结果，实际上限制了能做的对照组数量。
2. **有些结论被硬件污染了**。CPU 版里 GoogLeNet 的 s/MFLOP 是 VGG 的 3 倍，
   我把它归因为"Inception 的多分支小卷积在 CPU 上并行度低"。这是个关于**硬件**的结论，
   却很容易被读成关于**架构**的结论。换到 GPU 上跑一遍，才能把这两者分开。

GPU 上预计三组实验共 ~18 分钟，按量计费成本 ¥1 以内。

---

## 2. 机器选型

| 项 | 建议 | 理由 |
|---|---|---|
| 平台 | AutoDL | 按量计费、有无卡模式、镜像自带 CUDA 版 torch |
| 卡型 | RTX 3060 / 3080 / 3090 任一 | MNIST 28×28 + batch 64，显存占用 1GB 出头，**瓶颈在数据加载不在算力**，贵卡不会更快，按最便宜的选 |
| 镜像 | PyTorch 2.x + CUDA 11.8 或 12.1 官方镜像 | 自己装容易装成 CPU 版 |
| 磁盘 | 默认系统盘即可 | MNIST 约 50MB，四个 ckpt 共约 100MB |

**成本估算**：无卡模式准备 ~15 分钟 × ¥0.1/h ≈ ¥0.03；带卡训练 ~30 分钟（含冒烟测试与出图）× ¥1.5~2/h ≈ ¥1。
总计 **¥1 左右**。真正的开销风险是**跑完忘记关机**。

---

## 3. 移植改动清单

### 3.1 一行没改的文件

| 文件 | 说明 |
|---|---|
| `models.py` | 逐字节复制，MD5 已核对。改了就与 CPU 版不可比，`smoke.sh` 第 1 项会用参数量兜底检查 |
| `compute_stats.py` | 纯静态统计，与设备无关 |

### 3.2 `train.py`：删掉的 CPU 规避代码

CPU 版为了绕开 macOS + OpenMP 的段错误加了一堆补丁，GPU 上全部不需要，留着反而会误导：

| 删除 | 原因 |
|---|---|
| `OMP_MAX_ACTIVE_LEVELS` / `KMP_MAX_ACTIVE_LEVELS` 环境变量 | 针对 GoogLeNet 多分支触发的 OpenMP 嵌套并行崩溃，GPU 上计算不走 OpenMP |
| `torch.set_num_interop_threads(1)` | 同上 |
| `--threads` 参数 | GPU 上没有 CPU 线程数这个调节量 |
| `faulthandler.enable()` | 装来抓段错误栈的，问题不存在了 |
| 结尾的 `os._exit(0)` | CPU 版在解释器退出阶段会段错误，用它跳过析构。GPU 上正常退出，且 `os._exit` 会绕过 CUDA 上下文清理，留着有害 |

### 3.3 `train.py`：新增的 GPU 逻辑

这五处是移植的全部技术含量，每一处写错都**不会报错，只会悄悄给出错误的数字**。

**① 计时必须同步**（`sync()`，train.py:90，调用点 200 / 233）

CUDA kernel 是异步下发的，`loss.backward()` 返回时 GPU 还没算完。
不加 `torch.cuda.synchronize()` 的话，`time.perf_counter()` 测到的是**kernel 下发耗时**而不是执行耗时，
s/epoch 会小一个数量级 —— 而且看起来完全合理，不会有任何报错。
epoch 起点也要同步，否则会把上一轮拖尾算进这一轮。

同步点的位置与 CPU 版对齐：**s/epoch 含评估时间**，这样两边口径一致。

**② cudnn.benchmark + 预热**（`warmup()`，train.py:118）

`cudnn.benchmark=True` 会在每种新的（层配置, 输入形状）组合上试跑多种卷积算法挑最快的。
本任务输入固定 28×28，一次试跑之后全程受益，是净赚。
代价是**第 1 个 epoch 包含了试跑开销**，会让 epoch 1 的耗时虚高。

解决办法是训练开始前用随机数据跑 3 轮前反向把算法表填好。
关键细节：`warmup()` 必须在 `train_one()` 里 `manual_seed()` **之前**调用，否则它消耗的随机数会污染训练的随机流。

**③ CUDA 随机数种子**（train.py:155）

AlexNet 和 GoogLeNet 有 Dropout。Dropout 掩码在 GPU 上由 **CUDA RNG** 生成，
只设 `torch.manual_seed()` 管不到它 —— 必须加 `torch.cuda.manual_seed_all()`。
漏了的话这两个模型不可复现，而 VGG/ResNet（无 Dropout）看起来一切正常，很难发现。

同理，断点续训的 ckpt 里也存了 `cuda_rng`（train.py:250 存，179 取），否则续训后随机流对不上。

**④ 数据搬运与 DataLoader**（train.py:56-62, 108, 207）

| 改动 | 作用 |
|---|---|
| `num_workers` 默认 0 → 4 | Linux 上多进程加载是净收益（macOS 上 fork 开销大所以 CPU 版用 0） |
| `persistent_workers=True` | 一个 epoch 只要几秒，每轮重启 worker 的开销占比会很可观 |
| `pin_memory=True` | 锁页内存，H2D 拷贝才能走 DMA 并与计算重叠 |
| `.to(device, non_blocking=True)` | 配合 pin_memory 做异步拷贝 |
| `zero_grad(set_to_none=True)` | 置 None 比写零省一次显存遍历 |
| `evaluate()` 的每类计数器放在 GPU 上 | 避免每个 batch 都 D2H 同步一次，那会把异步流水线打断 |

**⑤ 环境信息落盘**（`env_info()`，train.py:278）

结果 json 里记 torch 版本、GPU 型号、CUDA/cudnn 版本、`cuda_available`。
用途有二：事后对比 CPU/GPU 结果时知道数据来自哪里；`smoke.sh` 第 3 项靠 `cuda_available` 断言没有白跑在 CPU 上。

### 3.4 新增参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--device` | `auto` | `auto` / `cuda` / `cuda:0` / `cpu`。auto 时有卡用卡 |
| `--num-workers` | `4` | 0 表示主进程加载 |
| `--cudnn-benchmark` / `--no-cudnn-benchmark` | 开 | 关掉可用于排查算法选择导致的数值差异 |

其余参数（`--model --epochs --batch-size --lr --optimizer --momentum --weight-decay --scheduler --augment --seed --resume --limit-batches --tag --out --data`）与 CPU 版完全一致。

### 3.5 出图与对比脚本

| 文件 | 改动 |
|---|---|
| `compare.py` | 字体候选表加入 Noto Sans CJK SC / WenQuanYi 等 Linux 常见中文字体。Ubuntu 默认无中文字体，不加图里全是方框 |
| `sweep_compare.py` | 参数差异对比的忽略字段：去掉 `threads`，加入 `device` 和 `cudnn_benchmark`。否则每组都会被报告为"与基准不同" |

### 3.6 Shell 脚本

三个实验脚本的骨架（跳过已有结果、失败重试、日志 tee）全部保留，只做了：

- 去掉 `--threads` 及 GoogLeNet 单独降线程的特判；
- 重试次数 3 → 2。CPU 上重试是为了扛 OpenMP 段错误，GPU 上崩溃概率低得多，
  但**保留重试仍有必要** —— 防的是 SSH 断连和按量实例被回收，不是防崩溃。

新增 `setup.sh`（环境准备）和 `smoke.sh`（移植验证），见第 5 节。

---

## 4. 已知不可比之处

上机前就要写下来，免得看到数字对不上时误判成 bug：

**① 准确率会有 ±0.1pp 级别的漂移，这是正常的。**
cudnn 选的卷积算法与 CPU 的 oneDNN 不同 → 浮点累加顺序不同 → 逐位结果不同 → 训练轨迹从第一个 batch 就开始分叉。
同 seed 也不会复现 CPU 的数字。注意这不只发生在跨硬件之间：`cudnn.benchmark` 按实测耗时挑算法，
同一张卡上两次进程也可能选中不同算法，GPU 上连「同机两次运行逐位一致」都不成立
（冒烟测试第 4 项专门量这个本底）。

**② `num_workers` 0 → 4 让数据加载的随机流也变了**，这是第二个逐位不可复现的来源。

**③ 耗时完全不可比**，这本来就是要换掉的量。有意义的是**同一硬件内部的相对关系**。

因此**能拿来对比的是结论的方向，不是数字本身**：

- cosine 在 5 轮下有正收益（CPU: +0.20pp）→ GPU 上是否仍为正？
- adam 明显更差（CPU: −0.20pp）→ 是否仍然最差？
- 20 轮下 cosine 与 augment 单独都无收益、组合 +0.14pp 的**交互效应** → 是否仍然存在？
- **GoogLeNet 的 s/MFLOP 劣势**（CPU 上是 VGG 的 3 倍）→ **这一条预期会消失或大幅缩小**，
  这正是这次重跑最想验证的东西。

同时注意 GPU 上的新混淆项：MNIST 太小，GPU 很可能**吃不满**，
四个网络的耗时差异会被数据加载时间掩盖。如果四个模型 s/epoch 都趋同到接近同一个值，
那说明测的是 DataLoader 而不是模型，需要在结论里明确指出。

---

## 5. 执行步骤

> 以下命令**全部在 AutoDL 实例上执行**，除了标注「本地 Mac」的两步。
> 本地 Mac 上不要跑任何训练或测试脚本。

### 步骤 0（本地 Mac）上传代码

```bash
cd .../machine-learning-and-deep-learning
scp -P <端口> -r homework-2026-08-29-gpu root@<实例地址>:~/autodl-tmp/
```

端口和地址在 AutoDL 控制台的 SSH 登录指令里。也可以在实例上直接 `git clone`。

### 步骤 1 无卡模式开机，准备环境

控制台选「**无卡模式开机**」（¥0.1/小时），SSH 进去：

```bash
cd ~/autodl-tmp/homework-2026-08-29-gpu
bash setup.sh
```

**预期输出**：torch 版本打印出来；`cuda 编译版本` 非 None；装完 matplotlib 和中文字体；`MNIST 就绪`。
无卡模式下 `nvidia-smi` 失败和 `cuda 可用 False` 都是**正常的**，脚本已容错。

**若 `cuda 编译版本` 是 None**：说明镜像里是 CPU 版 torch，换镜像重开，不要试图 pip 修。

完成后**关机**。

### 步骤 2 换带卡规格开机，冒烟测试

```bash
cd ~/autodl-tmp/homework-2026-08-29-gpu
bash smoke.sh
```

耗时几十秒。四项检查：

| # | 检查 | 预期 | 失败含义 |
|---|---|---|---|
| 1 | 参数量 | 4.82 / 1.13 / 0.27 / 2.76 M | `models.py` 被改过 |
| 2 | VGG 跑 3 个 batch | 不报错 | 移植有语法/形状错误 |
| 3 | `cuda_available` | true + 打印 GPU 型号 | 白跑在 CPU 上 |
| 4 | base vs cosine 的 e1 差异 | 落在同配置噪声本底内（0.1pp 量级） | 种子没控住 |

**四项全过才往下走。** 第 4 项尤其重要：三组实验里大量结论建立在 0.1~0.2pp 的组间差异上，
如果随机性没控住，这些差异根本无法区分是效果还是噪声。

第 4 项的原理：余弦退火第 1 个 epoch 的 lr 就等于初始 lr 0.01，base 和 cosine 两组
在第 1 轮理论上是同一个计算过程。CPU 版这里要求逐位一致；GPU 版**不能这么要求**——
`cudnn.benchmark` 按实测耗时给每种卷积形状挑算法，掐表有噪声，两次进程可能选中不同实现，
浮点累加顺序随之不同，结果在 1e-4 量级漂移（4090 实测：Δloss 2~4e-4、Δacc 0.1~0.4pp）。
所以 GPU 版的口径是：先同配置复跑一遍（dup）量出本机噪声本底，再要求 base vs cosine
的差异不超过「2 倍本次实测噪声」（下限 Δloss 0.005 / Δacc 0.5pp 兜底，防止本底恰好
测得很小而误报）；seed 真没控住时差异会大一到两个数量级，照样能抓到。
2026-09-06 首跑时旧检查报「seed 没控住」，其差异量级是数值噪声的典型特征、不像 seed
失控（那会差出 pp 级），故改为此口径；新检查自带的 dup 对比即是对此的常备验证。

### 步骤 3 三组实验

建议先挂 tmux：`tmux new -s mnist`（Ctrl-b d 脱离，`tmux a -t mnist` 重连）。

```bash
bash run_all.sh 5 baseline     # 实验一：四网络对比，~6 分钟
bash sweep.sh 5                # 实验二：超参数单变量，~4 分钟
bash long.sh 20                # 实验三：20 轮 2×2 因子，~8 分钟
```

三个脚本都会**跳过已有结果的组**，中断后重跑同一条命令即可续上；`FORCE=1` 强制重跑。

每组结束会自动出汇总表（`results/*_table.md`）和图（`results/figs/`）。

**跑完第一组时先做一次抽查**：看 `results/baseline_table.md` 里四个模型的准确率是否都在 99.2~99.4% 区间。
偏离太多（比如 <98%）说明有问题，不要接着跑后两组。

### 步骤 4（本地 Mac）取回结果

```bash
scp -P <端口> -r root@<实例地址>:~/autodl-tmp/homework-2026-08-29-gpu/results ./results-gpu
```

### 步骤 5 关机

**按量计费不关机会一直扣钱。** 跑完立刻在控制台关机。

---

## 6. 验收标准

移植算成功需要同时满足：

1. `smoke.sh` 四项全过；
2. 三组实验全部产出 json + 汇总表 + 图，无缺失组；
3. 四网络准确率都落在 **99.2~99.5%**，与 CPU 版（99.26 / 99.29 / 99.36 / 99.37）同区间；
4. 实验二里 cosine 与 augment 相对基准**仍为正收益**，adam **仍为负**；
5. 图里中文正常显示，不是方框；
6. 总 GPU 机时 < 1 小时。

第 3、4 条是核心：数字可以有 ±0.1pp 漂移，但**结论的方向必须一致**。
方向反了说明移植引入了实质性错误（最可能是随机性或数据管线），要回头查而不是直接采信新数字。

---

## 7. CPU 版基线数据（对比用）

### 实验一：四网络（5 epochs）

| 模型 | 参数量 | 最佳准确率 | 单 epoch 耗时 |
|---|---|---|---|
| alexnet | 4.82M | 99.26% | 421s |
| vgg | 1.13M | 99.29% | 117s |
| googlenet | 0.27M | 99.36% | 355s |
| resnet | 2.76M | 99.37% | 315s |

### 实验二：VGG 超参数（5 epochs）

| 配置 | 改动 | 最佳准确率 | 相对基准 |
|---|---|---|---|
| baseline | — | 99.29% | — |
| lr001 | lr=0.001 | 99.32% | +0.03 |
| lr1 | lr=0.1 | 99.32% | +0.03 |
| adam | adam, lr=0.001 | 99.09% | **−0.20** |
| bs256 | batch=256 | 99.22% | −0.07 |
| cosine | 余弦退火 | 99.49% | **+0.20** |
| augment | 旋转+平移 | 99.47% | **+0.18** |

### 实验三：VGG 20 轮 2×2 因子

| 配置 | 最佳准确率 | 相对基准 |
|---|---|---|
| e20_base | 99.53% | — |
| e20_cos | 99.49% | −0.04 |
| e20_aug | 99.57% | +0.04 |
| e20_cosaug | 99.67% | **+0.14** |

完整分析见 CPU 版 [experiments.md](../homework-2026-08-29/experiments.md) 与 [结论.md](../homework-2026-08-29/结论.md)。
