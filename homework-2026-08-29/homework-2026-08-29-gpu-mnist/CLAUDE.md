# CLAUDE.md

本目录是 MNIST 四网络对比作业的 **GPU 版**（AutoDL / Ubuntu）。
CPU 版在 `../homework-2026-08-29/`，是本目录的来源与对照基准。

## 协作方式

- **用户在 Mac 上写代码，在远程 AutoDL 实例上跑训练。**
  本地 Mac 没有 NVIDIA 显卡，**不要在本地执行任何训练或测试脚本**。
- 需要验证的地方，把命令和预期输出写进文档（`GPU计划.md` 第 5 节 / `README.md`），由用户自己在服务器上跑，
  再把输出贴回来一起排查。
- 用户是深度学习新手。解释要讲清"为什么"，不要只给结论；术语第一次出现时说明含义。

## 硬约束

**① `models.py` 一个字节都不能改。**
四个网络的定义必须与 CPU 版完全一致，否则两边的结果不可比，这次重跑就失去意义。
`smoke.sh` 第 1 项用参数量（4.82 / 1.13 / 0.27 / 2.76 M）做兜底检查。
如果确实需要改网络，那是另一个实验，应该新开一组 tag，而不是就地修改。

**② 实验设计不能改。**
三组实验（四网络对比 / VGG 超参数单变量 / 20 轮 2×2 因子）的分组、超参数取值、epoch 数
都与 CPU 版对齐。这次换的只有硬件。

**③ seed 固定 42，且 GPU 上必须同时设 CPU 和 CUDA 两套 RNG。**
AlexNet 和 GoogLeNet 有 Dropout，掩码在 GPU 上由 CUDA RNG 生成，
只设 `torch.manual_seed()` 管不到，必须加 `torch.cuda.manual_seed_all()`。
断点续训的 ckpt 里也要存 `cuda_rng`。

**④ 计时必须带 `torch.cuda.synchronize()`。**
CUDA kernel 异步下发，不同步的话测到的是 kernel 下发耗时而非执行耗时，
数字会小一个数量级且不会报错。同步点见 `train.py` 的 `sync()`，epoch 起点和终点各一次。
口径与 CPU 版对齐：**s/epoch 包含评估时间**。

**⑤ 不要 `pip install torch`。**
AutoDL 镜像自带 CUDA 版 torch，默认 pip 源装的是 CPU 版轮子，会覆盖掉它。
覆盖后训练照跑不误，只是慢 30 倍，且 `cuda_available` 变 False 而没人注意。
`requirements.txt` 里只有 matplotlib，这是故意的。

## 分析纪律

**判断过拟合不能只看 train/test gap。** 按可靠性排序看四个信号：
(1) 测试准确率过峰后下降 → (2) 最佳轮次远早于末轮 → (3) 训练 loss 趋近 0 → (4) gap 变大。
只有 (4) 成立时不能下过拟合结论。详见 CPU 版 `experiments.md` §5.9 ——
这条规则是从一次误判里总结出来的（曾据 gap 单项就断言 bs256 加重过拟合，实际测试曲线是单调上升的）。

**不要把 GPU 和 CPU 的数字直接并排比。** 三处天然不可比：
cudnn 与 oneDNN 的浮点累加顺序不同（±0.1pp 漂移是正常的）、
`num_workers` 从 0 改成了 4（随机流不同）、耗时口径完全不同。
**能比的是结论的方向，不是数字本身。**

**区分"架构的性质"和"硬件的性质"。**
CPU 版里 GoogLeNet 的 s/MFLOP 是 VGG 的 3 倍，那是关于 CPU 并行度的结论，不是关于 Inception 的结论。
这次重跑最想验证的就是这一条会不会消失。

**注意 GPU 上的新混淆项**：MNIST 太小，GPU 可能吃不满，
四个模型的耗时差异会被 DataLoader 时间掩盖。如果 s/epoch 都趋同，说明测的是数据管线不是模型，
要在结论里写明，不要假装测到了架构差异。

## 代码结构

| 文件 | 说明 |
|---|---|
| `models.py` | 四个网络定义。**只读** |
| `train.py` | 训练主程序，单模型或 `--model all` |
| `compute_stats.py` | 参数量 / FLOPs 静态统计 |
| `compare.py` | 实验一出图出表，按 `--tag` 聚合 |
| `sweep_compare.py` | 实验二/三出表，按 `--tags` 列表对比，自动 diff 各组参数差异 |
| `setup.sh` | 一次性环境准备，在**无卡模式**下跑 |
| `smoke.sh` | 移植正确性冒烟测试，四项检查，**正式实验前必跑** |
| `run_all.sh` / `sweep.sh` / `long.sh` | 三组实验 |

三个实验脚本共用同一套骨架：**结果文件已存在就跳过**（`FORCE=1` 覆盖）、
失败重试（GPU 上防的是 SSH 断连和实例回收，不是防崩溃）、日志 tee 到 `logs/`。
判成功的依据是**结果 json 是否落盘**，而不是退出码。

产物：`results/`（json + `*_table.md` + `figs/`）和 `logs/`，均不入库。

## 相对 CPU 版删掉的东西

CPU 版为绕开 macOS + OpenMP 段错误加的补丁，GPU 上全部删除，**不要因为"以防万一"加回来**：
`OMP_MAX_ACTIVE_LEVELS` / `KMP_MAX_ACTIVE_LEVELS` 环境变量、`set_num_interop_threads(1)`、
`--threads` 参数、`faulthandler.enable()`、结尾的 `os._exit(0)`。

其中 `os._exit(0)` 尤其不能留：它会绕过 CUDA 上下文清理。

## 出图

Ubuntu 默认没有中文字体，不装的话图里中文全是方框。
`setup.sh` 会装 `fonts-noto-cjk` 并清空 matplotlib 字体缓存（缓存不清，新装的字体扫不到）。
`compare.py` 的字体候选表已包含 Linux 常见中文字体名。
