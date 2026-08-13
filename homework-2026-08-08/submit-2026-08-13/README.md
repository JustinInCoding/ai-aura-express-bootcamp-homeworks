# PUBG 玩家排名预测 — 机器学习分析

> 基于 PUBG 比赛数据（4,446,966 条样本），构建**随机森林**与**决策树**回归模型，预测玩家最终排名百分位 `winPlacePerc`（0~1，越大排名越靠前）。
> 涵盖完整 ML 流水线：探索分析 → 数据清洗 → 特征工程 → 模型训练 → 超参数调优 → 评估与特征重要性分析。

| 目标指标 | 目标值 | 最终结果（随机森林） | 状态 |
|----------|--------|----------------------|------|
| R² Score | > 0.925 | **0.9489** | ✅ 达标 |
| MAE | < 0.059 | **0.0504** | ✅ 达标 |

---

## 目录

- [项目简介](#项目简介)
- [最终结果](#最终结果)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [分析流程](#分析流程)
- [关键发现](#关键发现)
- [相关文档](#相关文档)

---

## 项目简介

**任务**：根据每局比赛的玩家行为数据（击杀、伤害、移动距离、团队表现等），预测玩家在该局中的最终排名百分位。

**技术要点**：

- 数据量：444 万行 × 29 列原始特征，处理后 54 个特征
- 两个模型对比：**随机森林**（主力模型）vs **决策树**（对比/基线模型）
- 数据预处理 4 项：特殊缺失值清洗、无关特征剔除、matchType 独热编码、长尾特征 Winsorize 截尾
- 特征工程：队伍/比赛聚合特征（13 列）+ 4 个自定义特征 + 缺失指示特征
- 超参数优化：`RandomizedSearchCV`（3 折交叉验证，子样本调参）

---

## 最终结果

| 阶段 | 模型 | 训练集 R² | 验证集 R² | 验证集 MAE | 训练时间 |
|------|------|-----------|-----------|------------|----------|
| 基线（默认参数） | 随机森林 | 0.9931 | 0.9509 | 0.0490 | 402s |
| 基线（默认参数） | 决策树 | 1.0000 | 0.8973 | 0.0690 | 23s |
| **超参数调优** | **随机森林** | **0.9854** | **0.9489** | **0.0504** | **54s** |
| 超参数调优 | 决策树 | 0.9249 | 0.9229 | 0.0619 | 23s |

**主要结论**：

1. **集成学习显著胜出**：随机森林验证集 R² 0.9489 vs 决策树 0.9229，MAE 0.0504 vs 0.0619
2. **调优性价比极高**：随机森林验证 R² 仅损失 0.002，但训练时间从 402s 降至 54s，且过拟合差距从 0.042 收窄到 0.036
3. **决策树是容量不足而非过拟合**：训练-验证差距仅 0.002，单棵树在该数据的拟合上限约 0.92

---

## 项目结构

```
homework-2026-08-08/
├── pubg-analysis.ipynb     # 主分析 Notebook（51 个 cell，完整流水线）
├── plan.md                 # 分析计划（算法选型 / 预处理 / 特征工程方案）
├── conclusion.md           # 学习总结（逐阶段知识点 + 实验对比 + 踩坑记录）
├── check_longtail.py       # 长尾特征检查工具（脚本）
├── longtail_check.png      # 长尾特征直方图（脚本生成的示例输出）
├── note.md                 # 学习笔记（概念速记）
├── homework.md             # 作业要求
├── submit.md               # 作业提交链接
├── prompt/
│   ├── prompt.md           # 原始任务提示词
│   └── meaning-of-features.md  # 数据集字段含义对照表
└── data/                   # 数据集（需手动导入，见下文）
    ├── train_V2.csv        # 训练数据（约 629MB，4,446,966 行）
    └── test_V2.csv         # 测试数据（约 261MB，无标签）
```

---

## 环境要求

- Python 3.x（推荐使用 Conda 环境）
- 依赖库：

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

建议创建独立 Conda 环境（本项目使用 `machine-learning-aura`）：

```bash
conda create -n machine-learning-aura python=3.11
conda activate machine-learning-aura
pip install numpy pandas matplotlib seaborn scikit-learn
```

> 推荐在 Jupyter 中运行 Notebook：`conda install jupyter` 或 `pip install notebook`。

---

## 数据准备

数据文件较大（训练集约 629MB），需**手动导入**到 `data/` 目录（课程百度网盘）。

**数据来源**：[Kaggle — PUBG Finish Placement Prediction](https://www.kaggle.com/datasets/razamh/pubg-dataset)

文件要求：

| 文件 | 说明 |
|------|------|
| `data/train_V2.csv` | 训练数据，含目标列 `winPlacePerc` |
| `data/test_V2.csv` | 测试数据，无标签（本次作业未使用） |

---

## 快速开始

### 1. 运行主分析 Notebook

```bash
conda activate machine-learning-aura
jupyter notebook pubg-analysis.ipynb
```

按顺序执行全部 cell（约 10~15 分钟，取决于机器配置）。Notebook 中包含详细中文注释，每个阶段均输出中间结果，可逐步追踪。

### 2. 使用长尾特征检查工具（可选）

`check_longtail.py` 用于验证哪些数值特征是长尾分布、是否需要 Winsorize 截尾（Notebook 2.4 节选用了 5 个截尾列）：

```bash
# 基础：只看判定表格
python check_longtail.py

# 指定其他 CSV
python check_longtail.py data/train_V2.csv

# 同时绘制直方图（左=全量，右=去掉 top 1% 看清主体）
python check_longtail.py --plot
```

---

## 分析流程

```
加载数据 → 探索分析 → 数据清洗 → 特征工程 → 划分数据集
        → 基线模型 → 超参数调优 → 最终模型 → 评估 → 特征重要性分析
```

| 阶段 | 主要操作 | 目的 |
|------|----------|------|
| 探索分析 | `info()` / `describe()` / 直方图 / 箱线图 | 了解数据规模、缺失值、分布形状 |
| 特殊缺失值 | `rankPoints == -1` 等转为 NaN | 处理数据说明中的"隐式缺失值" |
| 类别编码 | matchType 合并为 7 类 + One-Hot（`drop_first=True`） | 把文字类别转成模型可用的 0/1 列 |
| 异常值处理 | 对 5 个长尾连续列 Winsorize 截尾 | 抑制极端值干扰（树模型对离散计数列不截断） |
| 特征工程 | 队伍/比赛聚合（13 列）+ 4 个新特征 + 缺失指示列 | 把原始信息变成更有预测力的特征 |
| 数据划分 | 80/20 划分，训练集子采样至 50 万，验证集保留完整 88.9 万 | 兼顾训练速度与评估可靠性 |
| 模型训练 | 随机森林 / 决策树（基线 + 调优） | 对比"单棵树 vs 集成" |
| 超参数调优 | `RandomizedSearchCV`（3 折，5 万子样本） | 自动寻找更优参数组合 |
| 特征重要性 | `feature_importances_` 分类汇总 | 回答"什么因素决定胜负" |

---

## 关键发现

基于随机森林特征重要性分类汇总：

1. **移动活动是最重要的获胜因素**（约 29%）——`walkDistance`、`totalDistance`、`rideDistance` 等。跑图积极、持续移动、搜索资源的玩家排名普遍更高。
2. **团队/队伍聚合特征次之**（约 25%）——`team_*` 聚合特征（队伍总伤害/总击杀/队内人数）是个人排名的强信号。与移动活动合计解释约 54% 的重要性。
3. **战斗能力约 15%**——`killPlace`、`kills`、`longestKill` 等。
4. **支援能力不可忽视**——`boosts`、`heals`、`assists`、`revives` 体现团队协作与生存续航。

**结论**：想赢 PUBG，跑图积极 + 持续移动 + 团队协作比单纯刷击杀更重要。

---

## 相关文档

| 文档 | 内容 |
|------|------|
| [plan.md](./plan.md) | 完整分析计划：算法选型理由、预处理与特征工程方案、调参策略 |
| [conclusion.md](./conclusion.md) | 学习总结：每个阶段做了什么、为什么、学到了什么，含踩坑记录 |
| [note.md](./note.md) | 概念速记（缺失值填充、Winsorize、drop_first 等） |
| [prompt/meaning-of-features.md](./prompt/meaning-of-features.md) | 数据集 29 个字段的中文含义与易错点 |

---

## 改进方向

- 尝试 **XGBoost / LightGBM** 梯度提升模型（大规模表格数据上通常更强）
- 按 `matchType` 分组建模（solo / duo / squad 的游戏逻辑不同）
- 将训练集从 50 万扩至 100 万，观察分数回升情况
- 对 `team_*` 特征做更多组合交互
