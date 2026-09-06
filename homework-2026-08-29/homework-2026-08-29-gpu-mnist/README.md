# MNIST 四网络对比 —— GPU 版（AutoDL / Ubuntu）

把 [CPU 版作业](../homework-2026-08-29/) 原样搬到云 GPU 上重跑。
**四个网络结构不动、三组实验内容不动、超参数不动**，只换执行环境。

目的有两个：

1. CPU 版跑完全部实验要 ~2.5 小时，GPU 上大约 10 分钟，把迭代成本压下来；
2. 拿到一组同代码、同种子、只有硬件不同的数据，用来看"哪些结论是模型的性质、哪些只是 CPU 的性质"。
   最典型的就是 GoogLeNet —— CPU 上它每 MFLOP 要花 3 倍于 VGG 的时间，这条结论在 GPU 上大概率不成立。

> 这里只放代码和执行说明。**实验设计、超参数含义、结果分析都在 CPU 版**：
> [experiments.md](../homework-2026-08-29/experiments.md) / [结论.md](../homework-2026-08-29/结论.md)

---

## 1. 目录内容

| 文件 | 作用 | 与 CPU 版的关系 |
|---|---|---|
| `models.py` | 四个网络定义 | **逐字节复制，不允许改** |
| `compute_stats.py` | 参数量 / FLOPs 统计 | 原样复制 |
| `train.py` | 训练主程序 | 已移植，见 [GPU计划.md](GPU计划.md) 第 3 节 |
| `compare.py` | 四网络对比出图出表 | 仅补了 Linux 中文字体名 |
| `sweep_compare.py` | 超参数组对比出表 | 仅改了参数差异对比的忽略字段 |
| `setup.sh` | 一次性环境准备 | 新增 |
| `smoke.sh` | 移植正确性冒烟测试 | 新增 |
| `run_all.sh` | 实验一：四网络对比 | 去掉 CPU 线程规避 |
| `sweep.sh` | 实验二：超参数单变量对照 | 去掉 `--threads` |
| `long.sh` | 实验三：20 轮 2×2 因子实验 | 去掉 `--threads` |

跑完之后产物落在 `results/`（json + md 表 + figs 图）和 `logs/`，两者都不入库。

---

## 2. 三组实验

与 CPU 版一一对应，细节见 CPU 版 `experiments.md`。

| # | 脚本 | 内容 | 预计 GPU 耗时 |
|---|---|---|---|
| 一 | `run_all.sh 5 baseline` | AlexNet / VGG / GoogLeNet / ResNet，各 5 epochs | ~6 分钟 |
| 二 | `sweep.sh 5` | 固定 VGG，单变量改 lr / optimizer / batch / scheduler / augment，6 组 | ~4 分钟 |
| 三 | `long.sh 20` | 固定 VGG，cosine × augment 的 2×2 因子，各 20 epochs | ~8 分钟 |

耗时按 RTX 3090 估，含数据加载。慢卡（如 3060）翻倍也仍在 30 分钟内。

---

## 3. 在 AutoDL 上怎么跑

### 3.1 选机器

- 镜像：**PyTorch 2.x + CUDA 11.8/12.1 官方镜像**，别自己从 miniconda 装。
- 卡型：**RTX 3060 / 3080 / 3090 任意一档都够**。MNIST 28×28、batch 64，显存占用 1GB 出头，
  瓶颈在数据加载而不在算力，买贵卡不会更快。按最便宜的选。

### 3.2 无卡模式做准备（¥0.1/小时）

装依赖和下数据不需要 GPU，没必要按 GPU 价格计费。用 AutoDL 的「**无卡模式开机**」：

```bash
cd ~/autodl-tmp/homework-2026-08-29-gpu
bash setup.sh
```

`setup.sh` 会检查 GPU/驱动、打印 torch 版本、装 matplotlib、装中文字体、预下载 MNIST。
无卡模式下 `nvidia-smi` 会失败，脚本里已经容错，属正常。

> ⚠️ **不要 `pip install torch`**。默认源装的是 CPU 版轮子，会把镜像自带的 CUDA 版覆盖掉，
> 之后 `torch.cuda.is_available()` 变 False，而训练照跑不误，只是慢 30 倍且你不会发现。

### 3.3 关机 → 换带卡规格开机 → 冒烟测试

```bash
bash smoke.sh
```

这一步**不是可选的**，四项检查任何一项挂了都不要往下跑，详见下面第 4 节。

### 3.4 正式跑

```bash
bash run_all.sh 5 baseline     # 实验一
bash sweep.sh 5                # 实验二
bash long.sh 20                # 实验三
```

三个脚本都做了「结果文件已存在就跳过」，中断后直接重跑同一条命令即可续上。
想强制重跑某组，加 `FORCE=1`。

跑长实验建议挂 tmux，SSH 断了进程不会死：

```bash
tmux new -s mnist
bash long.sh 20
# Ctrl-b d 脱离，重连后 tmux a -t mnist
```

### 3.5 取回结果

在**本地 Mac** 上执行（不是在服务器上）：

```bash
scp -P <端口> -r root@<实例地址>:~/autodl-tmp/homework-2026-08-29-gpu/results ./results-gpu
```

端口和地址在 AutoDL 控制台的 SSH 登录指令里。

### 3.6 关机

**按量计费不关机会一直扣钱**，跑完立刻在控制台关机。

---

## 4. 冒烟测试查什么

`smoke.sh` 的四项，以及各自失败意味着什么：

| # | 检查 | 挂了说明 |
|---|---|---|
| 1 | 四个网络参数量 = 4.82 / 1.13 / 0.27 / 2.76 M | `models.py` 被改过，与 CPU 版不可比 |
| 2 | VGG 跑通 3 个 batch | 移植有语法/形状错误 |
| 3 | 结果 json 里 `cuda_available=true` | 白跑在 CPU 上（多半是装了 CPU 版 torch） |
| 4 | cosine 第 1 轮 vs baseline 第 1 轮，差异落在同配置噪声本底内 | 种子没控住，后续 0.1pp 级别的组间差异全部无法解释 |

第 4 项的道理：余弦退火第一个 epoch 的 lr 就等于初始 lr 0.01，两组在第 1 轮应当是同一个计算过程。
CPU 版要求两组逐位相同；GPU 版做不到——`cudnn.benchmark` 按实测耗时给卷积挑算法，掐表有噪声，
两次进程可能选中不同实现，浮点累加顺序不同就带来 ~0.1pp 的漂移。所以 GPU 版的口径是：
先同配置复跑一遍量出噪声本底，再要求 base/cosine 的差异不超过它的 2 倍
（下限 Δloss 0.005 / Δacc 0.5pp 兜底）。seed 真没控住时（最常见是 DataLoader worker 的随机流出了问题）
差异会大一到两个数量级，照样能抓到；否则"cosine 比 baseline 高 0.20pp"这种结论就无从判断是真效果还是噪声。

---

## 5. 与 CPU 版对比时的注意事项

拿到 GPU 结果后**不要直接把两边的准确率并排比**，有三处天然不可比：

1. **准确率会有 ±0.1pp 级别的漂移**。cudnn 的卷积算法与 CPU 的 oneDNN 不同，
   浮点累加顺序不同 → 逐位结果不同 → 训练轨迹分叉。这是正常的，不是 bug。
2. **`num_workers` 从 0 改成了 4**，数据加载的随机流因此不同，同 seed 也不会复现 CPU 的逐位结果。
3. **耗时完全不可比**，这是本来就要换掉的量。有意义的是**同一硬件内部的相对关系**，
   比如"GoogLeNet 相对 VGG 的耗时倍数"在 CPU 上是 3.0×，GPU 上是多少。

可以直接对比的是**结论的方向**：cosine 有没有收益、adam 是不是更差、
20 轮下 cosine×augment 的交互项是不是仍然为正。

改动清单和每一处的理由见 [GPU计划.md](GPU计划.md)。
