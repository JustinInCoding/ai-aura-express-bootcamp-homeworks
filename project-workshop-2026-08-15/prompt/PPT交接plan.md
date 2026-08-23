# PPT 制作交接 Plan（给 workbuddy）

> 目标：把 Online Retail 电商客户聚类项目做成**答辩演示 PPT**。
> 项目根目录：`/Users/justinwang/Developer/ai-aura-express-bootcamp/machine-learning-and-deep-learning/project-workshop-2026-08-15`
> 所有文字要点、数字、图表路径均为**本项目实测产出**，可直接使用，无需重新计算。

---

## 一、项目一句话简介（封面/开场用）

基于 UCI Online Retail 电商交易数据（英国在线零售商，13 个月），用 **RFM 特征 + K-means / DBSCAN 无监督聚类**对 4,338 个客户做价值分群，产出 4 类客户画像与差异化运营策略，并完成 5 项优化实验验证模型稳健性。

---

## 二、PPT 页级提纲（共 13 页，约 8-10 分钟）

### P1 封面
- 主标题：电商客户价值聚类分析 —— 基于 RFM 与 K-means / DBSCAN
- 副标题：Online Retail 数据集 · 无监督学习 · 客户分群与运营策略
- 可放图：`results/聚类散点图.png` 做背景（弱化/半透明）或不用图

### P2 自我介绍（**留白页**）
- **此页只放标题「自我介绍」+ 占位提示框，正文由用户本人填写，不要生成虚构内容**
- workbuddy 只需做版式：姓名 / 学校专业 / 技能 / 联系方式 的占位区块，供用户填空

### P3 目录
- 项目背景 → 数据来源与处理 → 模型选择与构建 → 评估与优化 → 成果展示 → 项目心得

### P4 项目背景（1 页）
要点：
- 问题：在线零售商积累大量交易数据，但**不知道客户是谁、值多少、该怎么差异化运营**
- 三个目标：
  1. 用数据驱动识别客户价值分层
  2. 对比 K-means 与 DBSCAN 的适用性
  3. 产出每类客户画像与运营策略建议
- 放法建议：左侧问题陈述 + 右侧目标列表

### P5 数据来源（1 页）
- 数据集：UCI Online Retail Dataset（英国在线零售商，2010-12 ~ 2011-12）
- 原始规模：**541,909 行 × 8 列**，373 天，38 个国家
- 字段：InvoiceNo / StockCode / Description / Quantity / InvoiceDate / UnitPrice / CustomerID / Country
- 配图：无（或 `results/figures/eda_01_缺失值热力图.png` 引出下页清洗）

### P6 数据处理（1 页）
- 清洗 5 条规则（剔除 27.5% 脏数据）：
  | 规则 | 剔除行数 |
  |------|---------|
  | CustomerID 缺失 | 135,080 |
  | Quantity ≤ 0（退货/异常） | 10,624 |
  | UnitPrice ≤ 0 | 2,517 |
  | C 开头取消单 | 9,288 |
  | 完全重复行 | 5,192 |
- 清洗结果：541,909 → **392,692 行 / 18,532 订单 / 4,338 客户**
- 配图：`results/figures/eda_01_缺失值热力图.png`

### P7 数据探索 EDA（1 页，可选并入 P6）
- 英国一家独大：行口径 91.4%、客户口径 **90.4%**
- 典型「二八结构」：**一半客户只下过 ≤2 单**，少数大客户贡献大量订单
- 月度交易整体上升，11-12 月购物季冲高
- 配图：`results/figures/eda_02_月度交易走势.png`、`results/figures/eda_06_国家分布Top15.png`、`results/figures/eda_07_客户订单数分布.png`

### P8 模型选择与构建（1-2 页）
- RFM 特征工程：
  | 特征 | 定义 | 实测 |
  |------|------|------|
  | Recency | 距 2011-12-09 最近购买天数 | 中位约 50 天 |
  | Frequency | 不同订单数 | 中位 2，最大 209 |
  | Monetary | 累计消费额 | 中位约 673£ |
- 预处理关键：三特征**严重右偏**（Quantity 偏度 460）→ **log1p 压缩长尾**（偏度降到 1.03）→ StandardScaler 标准化 → `(4338, 3)` 特征矩阵
- 算法选择理由：K-means 适合连续特征、分区聚类；DBSCAN 密度聚类、自动识别噪声，二者对比
- 配图：`results/figures/fe_01_RFM原始分布.png`、`results/figures/fe_02_RFM_log1p分布.png`（或 eda_03/04）

### P9 选 K 与模型对比（1 页）
- 选 K：肘部法（inertia）拐点 + 轮廓系数，K=2~6 实测：
  | K | inertia | 轮廓系数 |
  |---|---------|---------|
  | 2 | 6,488 | 0.4340 |
  | 3 | 4,879 | 0.3381 |
  | **4** | **3,963** | **0.3368** |
  | 5 | 3,332 | 0.3155 |
  | 6 | 2,875 | 0.3143 |
- 结论：K=4 是性价比拐点，且业务上正好对应四档价值层级
- 模型对比：
  | 算法 | 簇数 | 噪声 | 轮廓系数 | DBI | CH |
  |------|------|------|---------|------|------|
  | **K-means** | 4 | 0 | 0.3368 | 1.0130 | 3,299.86 |
  | DBSCAN | 1 | 43 | 0.5522 | 1.1033 | 220.61 |
- 关键解读：DBSCAN 只分出 1 簇 + 43 噪声，eps 扫描无稳定平台——**RFM 数据密度连续，DBSCAN 结构性不适用**（不是参数问题）；K-means 4 簇是"方便运营的分层"而非天然分离
- 配图：`results/figures/selectk_肘部法轮廓系数.png`、`results/figures/dbscan_eps扫描.png`、`results/figures/dbscan_KNN距离图.png`

### P10 评估（1 页）
- 稳定性：抽样 500→4,338 重跑 20 次，**ARI 0.756 → 0.982**，分群收敛可复现
- 特征区分度 η²（聚类版"特征重要性"）：**Frequency 0.735 > Recency 0.686 > Monetary 0.665**，三特征都有贡献、无废特征
- 配图：`results/figures/km_稳定性vs样本量.png`、`results/figures/km_特征区分度eta2.png`

### P11 评估与优化（1-2 页）
5 项优化实验结论（来自 `notebooks/optimization.ipynb`）：
1. **特征扩展（7 维）**：K=4 轮廓 0.3368→0.2759 下降（高维惩罚 + 冗余特征），但**最优 K 移到 3**（轮廓 0.3796↑）——扩展价值在改变/明确最优结构
2. **Monetary Winsorize**：前后 ARI=0.993，分群几乎不变——**log1p 已让模型对极端值稳健**
3. **K 取舍**：K=3 指标最优（0.3796）、K=5 可识别顶级客户、K=4 居中——**K 本质是业务决策**
4. **RFM 5 档交叉验证（125 组合）**：序关系完全自洽（高价值簇=R4/F4/M4），互为校验；ARI 不能跨粒度直接比较
5. **PCA**：5 主成分保留 95.7% 方差，PCA 后 ARI=0.990 不改变分群——**3 维基线无需 PCA**，价值在高维场景与可视化
- 配图：无专属图（本小节是文字结论为主，如需可用 `results/figures/km_业务散点_RecencyMonetary.png` 过渡）

### P12 成果展示（1-2 页）★ 重点页
- 4 簇画像（数据来自 `results/簇画像表.csv`）：
  | 簇 | 业务标签 | Recency | Frequency | Monetary | 占比 |
  |----|---------|---------|-----------|----------|------|
  | 0 | **活跃高价值** | 8 天 | 10 单 | 3,750£ | 16.1% |
  | 1 | **沉睡低价值** | 173 天 | 1 单 | 297£ | 37.7% |
  | 2 | **近期低频·新客** | 16 天 | 2 单 | 480£ | 19.0% |
  | 3 | **中等价值** | 54 天 | 4 单 | 1,352£ | 27.2% |
- 运营策略（按簇落地）：
  - 活跃高价值 → VIP 专属权益、新品优先、大额订单提醒（维护+深耕）
  - 沉睡低价值 → 定向优惠券/邮件召回，连续 3 个月无响应降级（控制成本）
  - 近期低频·新客 → 首购 30 天内二次触达、捆绑推荐（培育复购）
  - 中等价值 → 交叉销售、满减激励、会员积分（提升客单）
- 配图：`results/聚类散点图.png`（主图）、`results/figures/km_簇中心热力图.png`、`results/figures/km_簇间箱线图.png`

### P13 项目心得 + 结束页（1-2 页）
心得 4 条（素材来源 `notes/调整与踩坑记录.md`、`notes/notes.md`）：
1. **环境问题占坑最多**：conda 与 pyenv 路径冲突导致包装错环境——工具链要先理顺再动手
2. **数据质量决定结论**：清洗口径（去重前后）、时间戳导致 Recency=-1、三种粒度混淆——每个数字都要标注口径
3. **先怀疑选型，再怀疑调参**：DBSCAN 因 RFM 密度连续而结构性不适用，换 eps 徒劳——算法与数据分布匹配比调参更重要
4. **实验思维**：特征扩展 ≠ 指标必然提升（先看相关矩阵）；稳健性检验看 ARI 而非轮廓微变；K 的选择是业务取舍不是纯指标问题
- 结束页：谢谢 + 一句收尾（如"用数据看清客户，用分群指导运营"）

---

## 三、可用素材速查（绝对路径，workbuddy 直接取用）

### 3.1 图表（results/figures/ 17 张 + results/ 1 张）
| 文件 | 用途 |
|------|------|
| `results/聚类散点图.png` | **核心成果图**（PCA/t-SNE × K-means/DBSCAN 2×2），P1/P12 |
| `results/figures/selectk_肘部法轮廓系数.png` | 选 K，P9 |
| `results/figures/km_簇中心热力图.png` | 簇画像，P12 |
| `results/figures/km_簇间箱线图.png` | 簇画像，P12 |
| `results/figures/km_业务散点_RecencyMonetary.png` | 业务视角散点，P11/P12 |
| `results/figures/km_特征区分度eta2.png` | η² 特征区分度，P10 |
| `results/figures/km_稳定性vs样本量.png` | 稳定性/ARI，P10 |
| `results/figures/dbscan_KNN距离图.png` | DBSCAN 调参诊断，P9 |
| `results/figures/dbscan_eps扫描.png` | DBSCAN 不适用证据，P9 |
| `results/figures/eda_01_缺失值热力图.png` | 缺失值，P5/P6 |
| `results/figures/eda_02_月度交易走势.png` | 月度走势，P7 |
| `results/figures/eda_03_数值分布_原始.png` | 原始分布（Quantity/UnitPrice），P8 备选 |
| `results/figures/eda_04_数值分布_log1p.png` | log1p 后分布，P8 备选 |
| `results/figures/eda_05_粒度对比_行级订单级客户级.png` | 粒度说明，可选 |
| `results/figures/eda_06_国家分布Top15.png` | 国家分布，P7 |
| `results/figures/eda_07_客户订单数分布.png` | 二八结构，P7 |
| `results/figures/fe_01_RFM原始分布.png` | RFM 右偏分布，P8 |
| `results/figures/fe_02_RFM_log1p分布.png` | RFM 变换后，P8 |

> 以上绝对路径均以项目根目录为前缀，如：`/Users/justinwang/Developer/ai-aura-express-bootcamp/machine-learning-and-deep-learning/project-workshop-2026-08-15/results/figures/eda_01_缺失值热力图.png`

### 3.2 数据文件
- `results/评估指标对比表.csv`：模型指标对比（K-means vs DBSCAN）
- `results/簇画像表.csv`：4 簇 RFM 画像 + 业务标签

### 3.3 文字素材源（需要更多细节时查阅）
- `report/项目报告.md`：**最全，PPT 所有文字数字都源自这里**（背景/清洗/方法论/实验/结论/建议）
- `docs/数据字典.md`：字段与清洗口径
- `docs/EDA报告.md`：EDA 细节
- `notes/调整与踩坑记录.md`：项目心得素材（第 1/4/5/8/9 节最有用）
- `notes/notes.md`：概念学习（RFM/PCA/t-SNE/DBI/CH/ARI/NMI）
- `项目计划书.md`：立项原始文档

---

## 四、给 workbuddy 的注意事项

1. **P2 自我介绍必须留白**：只做版式占位（姓名/学校/技能/联系方式），**不要生成任何虚构的个人信息**，由用户本人填写。
2. **中文渲染**：PPT 文字全部中文；图表已是 Arial Unicode MS 渲染好的 PNG，直接嵌入即可。
3. **风格建议**：答辩风 = 简洁 + 数据说话。建议主色中性（深蓝/灰），每页一个核心结论放标题区，图表大图 + 少量要点文字，避免长段落。
4. **图片引用**：所有图已确认存在、可直接使用；若 workbuddy 无法读取绝对路径，用相对项目根路径 `results/figures/xxx.png`。
5. **数字准确性**：PPT 内所有数字与本仓库 `report/项目报告.md` 一致，请勿改写或四舍五入出新的"约数"（如需约数，保持原样）。
6. **产出位置**：建议输出到 `report/演示PPT.pptx`（对齐项目计划书附件清单）。
