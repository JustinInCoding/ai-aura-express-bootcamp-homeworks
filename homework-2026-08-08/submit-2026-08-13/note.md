# 笔记 · 机器学习速记

> 面向 PUBG 排名预测项目（随机森林 vs 决策树）的概念速查，一句话记住一个点。

---

## 一、缺失值处理

- **两条路**：`dropna()` 删掉缺失行，或**填充**（均值/中位数/众数）——填充能保留样本量
- **填充器**：`SimpleImputer(strategy='median')`，先 `fit` 学中位数再 `transform`
- **特殊缺失值**：PUBG 里 `rankPoints == -1` 代表"未定级"，先转成 `NaN` 再统一处理
- **`_missing` 缺失指示特征**：`df[col].isna().astype(int)` 生成 0/1 列，**在填充前**把"缺失"这个事实记下来——缺失本身可能是信号（未定级玩家 = 菜鸟群体），填充会抹掉它
- **泄漏注意**：均值/中位数填充最好**先切分再 fit**（只从训练集学参数），防止验证集信息泄漏

---

## 二、类别特征编码

- **One-Hot**：`pd.get_dummies(columns=['matchType'])`，把文字类别变成 0/1 列
- **`drop_first=True`**：k 个类别只需 k−1 列（第 k 列可由其他列推出 = 冗余），去掉一列避免线性模型的**多重共线性**；树模型不依赖线性组合，drop 不 drop 都行，但更整洁
- **低频类别合并**：10+ 种 matchType 合并成 6 个常见 + `other`，减少稀疏列数
- **列数心算**：`get_dummies` 是"换列"不是"加列"——净增 = 新生成列 − 原列 1 个

---

## 三、异常值处理（Winsorize 截尾）

- **Winsorize（温莎化）**：把极端值**截断替换为边界值**，而不是删除（保留样本量）
- **IQR 法则**：`lower = Q1 − 1.5×IQR`，`upper = Q3 + 1.5×IQR`，越界值夹回边界
- **长尾判断三步**：① `describe()` 看 max 与 75% 分位差一个数量级+；② `skew()` 偏度 > 2 明显右偏；③ 画直方图看尾巴。脚本：`python check_longtail.py --plot`
- **离散计数不截**：`kills`/`heals`/`revives` 即使 skew 高也不截——计数列取值有限，小值（kills≥3）是有效信号，截断反而丢信息
- **树模型对离群值不敏感**：按阈值切分，个别极端值只影响局部分裂点，Winsorize 是锦上添花

---

## 四、特征工程

- **`groupby` + `transform`**：`count` = 数有几行（数人头），`sum` = 值求和（加数值）；`transform` 把结果**广播回每一行**，保持原形状才能直接赋值
- **队伍聚合**：`groupby(['matchId','groupId']).transform('sum')` 造 13 个 `team_*` 列——团队表现是个人排名的强信号（贡献约 25% 重要性）
- **比赛级特征**：`playersInMatch`（本场人数，`count`）直接决定排名百分位分母；`matchTotalKills`（全场总击杀，`sum`）
- **新特征**：
  - `totalDistance` = 走 + 载具 + 游泳（移动积极性）
  - `killEfficiency` = kills / (damageDealt + 1)（补刀效率，+1 防除零）
  - `teamContribution` = assists + revives（团队贡献）
  - `survivalScore` = (heals + boosts) / (kills + DBNOs + 1)（苟分 vs 刚枪风格）

---

## 五、数据划分

- `train_test_split(test_size=0.2, random_state=42)`：固定种子保证**可复现**
- **子采样加速**：训练集抽 50 万，**验证集保留完整 88.9 万**——验证集越大评估越可靠；"训练集 > 验证集"只是小数据经验法则
- `y = df['label'].copy()`：**copy 隔离**，防止后续改数据时误伤标签，也防止标签混进特征（泄漏）

---

## 六、模型与评估

- **随机森林 vs 决策树**：集成（bagging + 随机子空间）碾压单棵（Val R² 0.9489 vs 0.9229）；决策树差距小是**容量不足**不是过拟合
- **`model.score(X, y)` 就是 `r2_score(y, model.predict(X))`**：sklearn 回归器 score() 默认实现就是 R²
- **R² vs MAE**：R² 看"解释了多少方差"（1 最好，0 ≈ 均值瞎猜）；MAE 看"平均差多少分"（越小越好），两个视角互补
- **过拟合诊断**：训练 R² 高、验证 R² 低 → 背答案了；两者都低 → 容量不够
- **交叉验证**：`cross_val_score` 跑多指标要跑两遍；`cross_validate` 一次跑完双指标（R² + MAE），快一半

---

## 七、超参数调优

- **超参数 ≠ 参数**：超参数是训练前人为设定的（树数、深度、叶子大小），不是模型自己学的
- **RandomizedSearchCV vs GridSearchCV**：随机抽组合（大范围快锁定）vs 穷举（小范围精调）
- **子样本调参**：在 5 万样本上搜（快 10 倍），找到 `best_params_` 再用全量重训
- **约束搜索空间**：排除 `max_depth=None` + 500 树 + 全特征这种巨型树组合，否则单次拟合 10-30 分钟（实测 88 分钟翻车）
- **`ccp_alpha` 剪枝**：成本复杂度剪枝，决策树"留足容量 + 剪枝"组合可兼顾拟合与泛化；alpha 下限取 25 分位而非最小值，防巨型树

---

## 八、工程技巧

- `del df` 手动释放大变量内存
- matplotlib 中文：`plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'SimHei', ...]` + `axes.unicode_minus = False`；单独重跑 cell 要确保字体 cell 也跑过
- 全量训练前 `warnings.filterwarnings('ignore')` 保持输出整洁

---

## 九、结果速查

| 模型 | Val R² | Val MAE | 训练时间 |
|------|--------|---------|----------|
| 随机森林（调优）✅ | **0.9489** | **0.0504** | 54s |
| 决策树（调优） | 0.9229 | 0.0619 | 23s |

- **达标线**：R² > 0.925，MAE < 0.059
- **特征重要性**：移动活动 29% > 团队聚合 25% > 战斗能力 15% → 跑图积极 + 团队协作 > 单纯刷击杀
- **下一步**：XGBoost / LightGBM、按 matchType 分组建模
