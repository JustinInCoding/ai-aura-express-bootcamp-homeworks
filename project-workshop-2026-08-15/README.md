# 电商客户分群分析（Online Retail · K-means 与 DBSCAN 对比）

基于 **Online Retail 在线零售交易数据**（541,909 行 × 8 列）的机器学习聚类项目：数据清洗 → RFM 特征工程 → K-means / DBSCAN 双算法建模 → 多维度对比评估 → 客户价值分层与营销策略建议。

---

## 一、快速上手（30 秒看懂怎么跑）

```bash
# 1. 激活环境（每次运行脚本前必做）
conda activate machine-learning-aura

# 2. 安装依赖（仅首次）
python -m pip install -r requirements.txt

# 3. 一键复现全流程（src/ 下每个脚本可独立运行）
python src/data_preprocessing.py   # 清洗：541,909 → 392,692 行 / 4,338 客户
python src/feature_engineering.py  # RFM 特征工程 → (4338, 3) 标准化矩阵
python src/kmeans_clustering.py    # K-means 选 K + 稳定性测试（K=4，轮廓 0.3368）
python src/dbscan_clustering.py    # DBSCAN 调参（证实不适用）
python src/evaluation.py           # 评估指标 + 簇画像 + 特征区分度
```

> ⚠️ **环境坑（本项目最大的坑）**：`conda run -n machine-learning-aura python ...` 里的 pip/python 可能解析到 pyenv 的 Python，把包装错地方。**务必先 `conda activate machine-learning-aura` 再跑**，或直接用绝对路径 `/opt/anaconda3/envs/machine-learning-aura/bin/python`。

### 交互式探索（推荐先看 notebook）

```bash
conda activate machine-learning-aura
jupyter lab notebooks/analysis.ipynb      # 全流程：EDA→清洗→RFM→K-means/DBSCAN→评估→结论
jupyter lab notebooks/optimization.ipynb  # 优化实验：特征扩展/Winsorize/K 对比/RFM 校验/PCA
```

---

## 二、项目结构

```
.
├── README.md                  # 本文件
├── 项目计划书.md                # 立项文档（背景/方案/步骤/产出清单）
├── 数据集推荐.md                # 候选数据集调研与推荐
├── requirements.txt           # Python 依赖（已固定版本）
│
├── data/                      # 数据（gitignore 忽略）
│   ├── OnlineRetail.csv       # 原始数据（latin-1 编码，含 £ 符号）
│   └── customer_rfm_clustered.csv  # 聚类后客户 RFM + 簇标签
│
├── notebooks/                 # 分析主流程 + 优化实验（逐 cell 可复现）
│   ├── analysis.ipynb         # EDA→清洗→RFM→K-means/DBSCAN→对比评估→结论
│   └── optimization.ipynb     # 5 项优化实验（特征扩展/Winsorize/K/RFM 校验/PCA）
│
├── src/                       # 可复用脚本（从 analysis.ipynb 抽取，与 notebook 数字一致）
│   ├── data_preprocessing.py  # 数据加载 + 清洗（5 条规则）
│   ├── feature_engineering.py # RFM 构建 + log1p + 标准化
│   ├── kmeans_clustering.py   # 选 K / 拟合 / 稳定性测试
│   ├── dbscan_clustering.py   # KNN 距离 / eps 扫描 / 拟合
│   ├── evaluation.py          # 轮廓/DBI/CH/ARI/NMI + 簇画像 + η²
│   └── visualization.py       # 全部绘图函数（16+ 个，中文字体已配）
│
├── results/                   # 结果产物（gitignore 忽略，可由 src/ 重新生成）
│   ├── 评估指标对比表.csv       # K-means vs DBSCAN 指标表
│   ├── 聚类散点图.png          # PCA/t-SNE × 两算法 2×2 核心成果图
│   ├── 簇画像表.csv            # 4 簇 RFM 画像 + 业务标签 + 占比
│   ├── figures/               # 各阶段过程图表 20 张（EDA/特征/选K/画像/稳定性/DBSCAN）
│   └── code_screenshots/      # 代码截图（PPT 用）
│
├── docs/                      # 文档
│   ├── 数据字典.md              # 字段说明 + 清洗规则
│   └── EDA报告.md              # 数据探索报告（含图表）
│
├── notes/                     # 笔记
│   ├── notes.md               # 核心概念（RFM/PCA/t-SNE/DBI/CH/ARI/NMI）
│   └── 调整与踩坑记录.md         # 实操坑点与调整（最详实）
│
├── prompt/                    # 需求与提示词
│   ├── prompt.md              # 原始需求
│   ├── features.md            # 数据集特征说明
│   └── PPT交接plan.md          # 给 workbuddy 做 PPT 的交接提纲与素材
│
├── report/                    # 交付报告
│   ├── 项目报告.md              # 完整项目报告（含图表、运营策略、复现命令）
│   └── 演示PPT.pptx            # 答辩 PPT（workbuddy 生成）
│
├── workbuddy-tooling/         # workbuddy 辅助脚本（gitignore 忽略）
│   ├── build_ppt.py           # 生成 PPT
│   └── gen_comparison_charts.py # 生成 PPT 用对比图
│
└── template/                  # PPT 模板（gitignore 忽略）
```

---

## 三、环境与依赖

| 项 | 值 |
|----|----|
| 环境管理器 | Anaconda (macOS) |
| 环境名 | `machine-learning-aura` |
| Python | 3.10+（建议） |

核心依赖（`requirements.txt` 已固定版本）：

| 包 | 用途 |
|----|------|
| `pandas` | 数据处理与 RFM 聚合 |
| `numpy` | 数值计算 |
| `scikit-learn` | K-means / DBSCAN / 评估 / PCA / t-SNE |
| `matplotlib`、`seaborn` | 可视化 |
| `scipy` | 距离计算与统计 |
| `jupyterlab`、`notebook` | Notebook 交互分析 |

### 中文字体配置

项目图全部用 **Arial Unicode MS** 渲染中文。若换机后图表中文变方块，在绘图脚本头部设置（**必须放在 `sns.set_theme()` 之后**）：

```python
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang HK", "Hiragino Sans GB", "Heiti TC", "Songti SC"]
plt.rcParams["axes.unicode_minus"] = False
```

---

## 四、数据说明

- **来源**：UCI / Kaggle `Online Retail`
- **规模**：541,909 行 × 8 列，时间跨度 2010-12-01 至 2011-12-09
- **编码**：**latin-1**（含 `£` 等非 UTF-8 字符），读取需 `encoding='latin-1'`
- **字段**：InvoiceNo / StockCode / Description / Quantity / InvoiceDate / UnitPrice / CustomerID / Country
- 特征细节见 `prompt/features.md`，口径见 `docs/数据字典.md`

### 数据口径（三种粒度，别混淆）

| 粒度 | 规模 | 用途 |
|------|------|------|
| 行级 | 541,909 | 每行 = 订单中某商品，仅用于字段清洗 |
| 订单级 | 25,900 | 按 InvoiceNo 聚合 |
| 客户级 | **4,372 → 4,338** | 按 CustomerID 聚合，**RFM 与聚类粒度** |

### 清洗 5 条规则

| 规则 | 剔除行数 |
|------|---------|
| CustomerID 缺失 | 135,080（24.9%） |
| Quantity ≤ 0（退货/异常） | 10,624 |
| UnitPrice ≤ 0 | 2,517 |
| C 开头取消单 | 9,288 |
| 完全重复行 | 5,192 |

清洗结果：541,909 → **392,692 行 / 18,532 订单 / 4,338 客户**

---

## 五、核心结论（快速回顾）

### 模型对比

| 算法 | 簇数 | 轮廓系数 | DBI | CH |
|------|------|---------|------|------|
| **K-means (K=4)** | 4 | **0.3368** | 1.0130 | 3,299.86 |
| DBSCAN | 1（+43 噪声） | 0.5522 | 1.1033 | 220.61 |

- **DBSCAN 不适用**：RFM 数据密度连续，eps 扫描无稳定平台，只分出 1 簇——是数据结构问题，非调参问题。
- **K-means 4 簇**是"方便运营的分层"而非天然分离。

### 4 簇客户画像

| 簇 | 业务标签 | Recency | Frequency | Monetary | 占比 |
|----|---------|---------|-----------|----------|------|
| 0 | 活跃高价值 | 8 天 | 10 单 | 3,750£ | 16.1% |
| 1 | 沉睡低价值 | 173 天 | 1 单 | 297£ | 37.7% |
| 2 | 近期低频·新客 | 16 天 | 2 单 | 480£ | 19.0% |
| 3 | 中等价值 | 54 天 | 4 单 | 1,352£ | 27.2% |

### 优化实验 5 项（`notebooks/optimization.ipynb`）

1. **特征扩展（7 维）**：最优 K 移到 3（轮廓 0.3796↑），K=4 原地下降（高维惩罚）
2. **Monetary Winsorize**：ARI=0.993，log1p 已让模型对极端值稳健
3. **K 取舍**：K 本质是业务决策（K=3 指标最优 / K=5 识别顶级客户 / K=4 居中）
4. **RFM 5 档交叉验证（125 组合）**：序关系完全自洽，互为校验
5. **PCA**：3 维基线无需 PCA，价值在高维场景与可视化

---

## 六、复现命令

### 全流程脚本（src/，每个独立可跑，数字与 notebook 一致）

```bash
conda activate machine-learning-aura
python src/data_preprocessing.py    # 清洗 → 打印各步行数
python src/feature_engineering.py   # RFM + 变换 → (4338, 3)
python src/kmeans_clustering.py     # 选 K + 稳定性 → K=4 轮廓 0.3368
python src/dbscan_clustering.py     # KNN + eps 扫描 → 证实不适用
python src/evaluation.py            # 指标 + 簇画像 + η²
```

### 重新生成 results/ 产物

`results/` 已被 gitignore，可随时用 `src/` 重算。核心成果图（`聚类散点图.png`）按 notebook 9.2 原配色生成：

```python
import matplotlib; matplotlib.use("Agg")
from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import build_rfm, transform_features
from src.kmeans_clustering import fit_kmeans
from src.dbscan_clustering import fit_dbscan
from src.visualization import setup_style, plot_clusters_2d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

df = clean_data(load_data("data/OnlineRetail.csv"))
rfm = build_rfm(df[0])
X, _, _ = transform_features(rfm)
rfm["Cluster_KM"] = fit_kmeans(X, 4)[1]
rfm["Cluster_DB"] = fit_dbscan(X, 0.6, 5)[1]
X_pca = PCA(2, random_state=42).fit_transform(X)
X_tsne = TSNE(2, random_state=42).fit_transform(X)
setup_style()
plot_clusters_2d([("PCA", X_pca), ("t-SNE", X_tsne)], rfm, "Cluster_KM", "Cluster_DB")
```

### 答辩 PPT

- 成品：`report/演示PPT.pptx`
- 交接提纲：`prompt/PPT交接plan.md`（素材路径、关键数字速查、注意事项）
- workbuddy 辅助脚本：`workbuddy-tooling/`（`build_ppt.py` 生成 PPT、`gen_comparison_charts.py` 生成对比图）

---

## 七、调整与踩坑记录（摘要）

详细版见 `notes/调整与踩坑记录.md`。

1. **依赖装错环境（最大坑）**：`conda run` 解析到 pyenv Python → 用 `/opt/anaconda3/envs/machine-learning-aura/bin/python -m pip install -r requirements.txt`，装完 **Restart Kernel**。
2. **matplotlib 中文方块字**：字体配置必须放在 `sns.set_theme()` 之后（否则被 seaborn 重置）。
3. **清洗数字对不上**：`print` 写在 `drop_duplicates()` 之前 → 分两步打印。
4. **Recency 出现 -1**：`InvoiceDate` 含时间戳，`REF_DATE` 是零点 → 按自然日 `(REF_DATE - x.max().normalize()).days`，修复后 min=0。
5. **3 种粒度混淆**：RFM 与聚类用**客户级**，别用行级/订单级。

---

## 八、当前进度

对照项目计划书「四、项目实施步骤」：

- ✅ 数据获取 → 探索（EDA）→ 清洗 → RFM 特征工程 → 聚类建模 → 对比评估 → 结论
- ✅ 优化实验 5 项（`optimization.ipynb`）
- ✅ `src/` 6 个可复用脚本
- ✅ `results/` 产物 + `docs/` 文档 + `report/项目报告.md`
- ✅ `report/演示PPT.pptx`（workbuddy 生成）+ `prompt/PPT交接plan.md`
