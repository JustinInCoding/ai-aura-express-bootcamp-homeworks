"""
长尾特征检查工具
================
用途：查看数据里哪些数值特征是"长尾分布"（少数极端值把尾巴拖得很长），
      并判断它们是否值得做 Winsorize 截尾。

用法：
    python check_longtail.py                      # 分析默认路径 data/train_V2.csv
    python check_longtail.py 某文件.csv            # 分析指定 CSV
    python check_longtail.py --plot               # 额外画出长尾候选列的直方图
    python check_longtail.py 某文件.csv --plot

判断规则（启发式阈值，可自行调整）：
    1. 离散判定：列值全部为整数 且 唯一值数量 <= 500  -> 计数特征，谨慎截尾
    2. skew（偏度）> 2        -> 明显右偏
    3. max / p75 > 10         -> 极端值比大多数人大一个数量级以上
    - 同时满足 2 和 3 且非离散  -> 长尾明显，建议截尾
    - 只满足其一且非离散        -> 偏长尾，可考虑截尾
"""

import argparse

import numpy as np
import pandas as pd

# 默认数据路径（可被命令行参数覆盖）
DEFAULT_CSV = 'data/train_V2.csv'

# 判定阈值（改这里即可调整灵敏度）
SKEW_THRESHOLD = 2.0      # 偏度超过该值视为明显右偏
RATIO_THRESHOLD = 10.0    # max 超过 p75 该倍数视为数量级差异
MAX_UNIQUE_DISCRETE = 500 # 唯一值数不超过该值且为整数 -> 视为离散计数


def main():
    parser = argparse.ArgumentParser(description='长尾特征检查工具')
    parser.add_argument('csv', nargs='?', default=DEFAULT_CSV, help='CSV 文件路径')
    parser.add_argument('--plot', action='store_true', help='画出长尾候选列的直方图')
    args = parser.parse_args()

    # ---- 1. 先读 100 行样本，推断出所有数值列（避免读入 Id 等文本大列）----
    # 注意：不能 nrows=0，那样 pandas 推断不出 dtype，所有列都会被当成 object
    sample = pd.read_csv(args.csv, nrows=100)
    num_cols = sample.select_dtypes(include='number').columns.tolist()
    print(f'数值列共 {len(num_cols)} 个：{num_cols}')

    # ---- 2. 只读数值列，减少内存和 IO ----
    df = pd.read_csv(args.csv, usecols=num_cols)
    print(f'读取完成，shape={df.shape}')

    # ---- 3. 逐列计算指标 ----
    results = []
    for col in num_cols:
        s = df[col].dropna()  # 该列非空的值
        if len(s) == 0:
            continue

        skew = s.skew()                 # 偏度：>0 右偏，>2 明显右偏
        p75 = s.quantile(0.75)          # 75% 分位（大多数人的水平）
        p90 = s.quantile(0.90)          # 90% 分位（p75=0 时兜底用）
        mx = s.max()                    # 最大值（极端值）
        n_unique = s.nunique()          # 唯一值数量（判断离散/连续）

        # 计算 max 相对"正常值"的倍数；p75/p90 都为 0 时用最小正数兜底，全 0 则记 NaN
        base = p75 if p75 > 0 else p90
        if base <= 0:
            pos = s[s > 0]
            base = pos.min() if len(pos) else np.nan
        ratio = mx / base if base == base and base > 0 else np.nan  # 防 NaN 传播

        results.append({'特征': col, 'skew': round(skew, 2), 'p75': p75,
                        'max': mx, 'max/p75': ratio, '唯一值': n_unique})

    res = pd.DataFrame(results).sort_values('skew', ascending=False).reset_index(drop=True)

    # ---- 4. 逐行判定 ----
    def verdict(row):
        col = row['特征']
        s = df[col].dropna()
        is_integer = bool((s == s.round()).all())          # 列值全为整数？
        discrete = is_integer and row['唯一值'] <= MAX_UNIQUE_DISCRETE  # 计数特征？
        skewed = row['skew'] > SKEW_THRESHOLD
        extreme = (row['max/p75'] is not None and pd.notna(row['max/p75'])
                   and row['max/p75'] > RATIO_THRESHOLD)

        if discrete:
            return '离散计数（整数、取值少）：即使 skew 高也不建议截尾，小值才是有效信号'
        if skewed and extreme:
            return '长尾明显：建议 Winsorize 截尾'
        if skewed or extreme:
            return '偏长尾：可考虑截尾'
        return '轻偏/正常：无需处理'

    res['判断'] = res.apply(verdict, axis=1)

    # ---- 5. 格式化输出 ----
    pd.set_option('display.width', 180)
    pd.set_option('display.max_rows', None)
    fmt = res.copy()
    fmt['max/p75'] = fmt['max/p75'].apply(
        lambda v: 'inf' if (v is not None and pd.notna(v) and v == float('inf'))
        else (f'{v:.0f}倍' if pd.notna(v) else 'NaN')
    )
    print('\n===== 长尾检查结果（按 skew 降序）=====')
    print(fmt.to_string(index=False))

    # ---- 6. 汇总建议 ----
    strong = res[res['判断'].str.startswith('长尾明显')]['特征'].tolist()
    weak = res[res['判断'].str.startswith('偏长尾')]['特征'].tolist()
    print('\n===== 汇总建议 =====')
    print(f'强烈建议截尾：{strong}')
    print(f'可考虑截尾  ：{weak}')
    print(f'不建议截尾  ：其余列（离散计数或分布正常）')

    # ---- 7. 可选：画直方图对比"全量 vs 去掉 top 1%" ----
    if args.plot:
        candidates = strong + weak
        if not candidates:
            print('\n没有需要画图的长尾候选列。')
            return
        import matplotlib
        matplotlib.use('Agg')  # 无界面环境下保存图片
        import matplotlib.pyplot as plt
        # 中文字体配置（与 notebook 保持一致）
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC',
                                           'Hiragino Sans GB', 'SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        n = len(candidates)
        fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))
        if n == 1:
            axes = axes.reshape(1, -1)
        for i, col in enumerate(candidates):
            s = df[col].dropna()
            # 左图：全量，能看到被极端值拉长的尾巴
            axes[i, 0].hist(s, bins=50, color='steelblue', edgecolor='white')
            axes[i, 0].set_title(f'{col}：全量（max={s.max():.0f}）')
            # 右图：只画 99% 分位以内的主体，看清大多数人的分布
            th = s.quantile(0.99)
            axes[i, 1].hist(s[s <= th], bins=50, color='coral', edgecolor='white')
            axes[i, 1].set_title(f'{col}：去掉 top 1%（p99={th:.0f}）')
        plt.tight_layout()
        out = 'longtail_check.png'
        plt.savefig(out, dpi=120)
        print(f'\n直方图已保存：{out}（每行左=全量，右=去掉 top 1%）')


if __name__ == '__main__':
    main()
