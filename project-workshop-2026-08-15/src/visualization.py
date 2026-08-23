# -*- coding: utf-8 -*-
"""
可视化脚本（visualization.py）
================================
对应 analysis.ipynb 各节的图表：EDA 分布、选 K、KNN 距离、聚类散点、
簇中心热力图、特征区分度、箱线图、稳定性、eps 扫描、业务散点。

所有绘图函数只负责「画图并展示」，不修改数据。使用前请先调用 setup_style()
设置全局风格与中文字体（macOS）。

用法：
    from visualization import setup_style, plot_elbow_silhouette, ...
    setup_style()
    plot_elbow_silhouette(list(k_range), inertias, sils)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_style():
    """全局绘图风格 + macOS 中文字体（放在所有绘图之前调用一次）。"""
    sns.set_theme(style="whitegrid", palette="muted")
    # 解决 macOS 中文字体乱码：matplotlib 默认 Arial 不含中文字符
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS", "PingFang HK", "Hiragino Sans GB",
        "Heiti TC", "Songti SC",
    ]
    plt.rcParams["axes.unicode_minus"] = False     # 负号正常显示为 "-"
    plt.rcParams["figure.figsize"] = (10, 5)       # 默认画布大小
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")


# ---------------- EDA ----------------

def plot_missing_heatmap(df):
    """缺失值热力图：黄=缺失、紫=有值。"""
    plt.figure(figsize=(9, 4))
    sns.heatmap(df.isna(), cbar=False, cmap=["#5a189a", "#f4c542"], yticklabels=False)
    plt.title("缺失值分布（黄=缺失，紫=有值）")
    plt.tight_layout()
    plt.show()


def plot_monthly_trend(df):
    """月度交易行数折线图。"""
    monthly = df.groupby(df["InvoiceDate"].dt.to_period("M")).size()
    plt.figure(figsize=(12, 4.5))
    monthly.plot(marker="o")
    plt.title("月度交易行数")
    plt.xlabel("月份")
    plt.ylabel("行数")
    plt.tight_layout()
    plt.show()
    return monthly


def plot_numeric_distribution(df, cols=("Quantity", "UnitPrice"), log=False):
    """行级数值分布直方图（正数部分，截断到 99 分位展示主体形态）。

    log=True 时画 log1p 后的分布（验证对数变换效果）。
    """
    fig, axes = plt.subplots(1, len(cols), figsize=(13, 4))
    for ax, col in zip(axes, cols):
        data = df[col][df[col] > 0]                    # 只取正数（负值/0 为脏数据）
        if log:
            ax.hist(np.log1p(data), bins=80)
            ax.set_title(f"log1p({col}) 分布（正数部分）")
            ax.set_xlabel(f"log1p({col})")
        else:
            upper = data.quantile(0.99)                # 99 分位截断，避免长尾撑爆坐标轴
            ax.hist(data[data <= upper], bins=60, color="steelblue", edgecolor="white")
            ax.set_title(
                f"{col} 分布（正数，截断到 99 分位={upper:,.0f}）\n"
                f"均值={data.mean():,.2f} | 中位数={data.median():,.2f} | 最大={data.max():,.0f}"
            )
            ax.set_xlabel(col)
        ax.set_ylabel("频数")
    plt.tight_layout()
    plt.show()


def plot_granularity(df):
    """三种粒度 Quantity 分布对比：行级 / 订单级 / 客户级。"""
    inv_qty = df.groupby("InvoiceNo")["Quantity"].sum()
    cust_qty = df.groupby("CustomerID")["Quantity"].sum()
    datasets = [
        ("行级（每行=订单中某商品）", df["Quantity"][df["Quantity"] > 0]),
        ("订单级（每单=一次交易）", inv_qty[inv_qty > 0]),
        ("客户级（每客=一个客户）", cust_qty[cust_qty > 0]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (title, data) in zip(axes, datasets):
        upper = data.quantile(0.99)
        ax.hist(data[data <= upper], bins=50, color="steelblue", edgecolor="white")
        ax.set_title(f"{title}\n均值={data.mean():,.1f} | 中位数={data.median():,.0f} | 最大={data.max():,.0f}")
        ax.set_xlabel("数量")
        ax.set_ylabel("频数")
    plt.tight_layout()
    plt.show()


def plot_country_top(df, top=15):
    """交易行数 Top N 国家柱状图。"""
    df["Country"].value_counts().head(top).plot(kind="bar")
    plt.title(f"交易行数 Top{top} 国家")
    plt.xlabel("国家")
    plt.ylabel("行数")
    plt.tight_layout()
    plt.show()


def plot_orders_per_customer(df):
    """客户订单数分布（截断 50）。"""
    valid = df[df["CustomerID"].notna() & ~df["InvoiceNo"].astype(str).str.startswith("C")]
    orders = valid.groupby("CustomerID")["InvoiceNo"].nunique()
    orders.clip(upper=50).hist(bins=50)
    plt.title("客户订单数分布（截断 50）")
    plt.xlabel("订单数")
    plt.ylabel("客户数")
    plt.tight_layout()
    plt.show()


def plot_rfm_distribution(rfm, log=False):
    """RFM 三特征分布直方图；log=True 时画 log1p 后分布。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, rfm.columns):
        data = np.log1p(rfm[col]) if log else rfm[col]
        ax.hist(data, bins=80)
        ax.set_title(f"{col}（{'log1p' if log else '原始'}）")
        ax.set_xlabel(col)
    plt.tight_layout()
    plt.show()


# ---------------- 聚类选参 ----------------

def plot_elbow_silhouette(k_range, inertias, sils):
    """选 K 双图：左=肘部法（inertia），右=轮廓系数。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(list(k_range), inertias, marker="o")
    axes[0].set_title("肘部法（inertia 越小越好，找拐点）")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("inertia")
    axes[1].plot(list(k_range), sils, marker="o")
    axes[1].set_title("轮廓系数（越大越好）")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("silhouette")
    plt.tight_layout()
    plt.show()


def plot_knn_distance(kdist, min_pts):
    """KNN 距离图：排序后的第 k 近邻距离曲线，拐点对应 eps 参考值。"""
    plt.figure(figsize=(10, 4))
    plt.plot(kdist)
    plt.title(f"KNN 距离图（k={min_pts}），找拐点处对应的距离 → eps")
    plt.xlabel("样本（按距离排序）")
    plt.ylabel("第 k 近邻距离")
    plt.tight_layout()
    plt.show()


# ---------------- 聚类结果展示 ----------------

def plot_clusters_2d(views, rfm, km_col, db_col):
    """2×2 大图：2 种降维（PCA/t-SNE）× 2 种算法（K-means/DBSCAN）。

    views: [(视图名, 2 维坐标数组), ...]，如 [("PCA", X_pca), ("t-SNE", X_tsne)]
    注意：K-means 与 DBSCAN 的颜色「同色不同义」（DBSCAN 深紫=噪声），读图时先看列标题。
    """
    algs = [(km_col, True), (db_col, False)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for r, (vname, Xv) in enumerate(views):
        for c, (col, use_cmap) in enumerate(algs):
            ax = axes[r][c]
            if use_cmap:
                sc = ax.scatter(Xv[:, 0], Xv[:, 1], c=rfm[col], cmap="viridis", s=8, alpha=0.7)
                fig.colorbar(sc, ax=ax, label="簇")
            else:
                # DBSCAN 噪声点是 -1，转成 float 与正常簇共用同一张颜色图
                colors = np.where(rfm[col] == -1, -1.0, rfm[col].astype(float))
                sc = ax.scatter(Xv[:, 0], Xv[:, 1], c=colors, cmap="viridis", s=8, alpha=0.7)
                fig.colorbar(sc, ax=ax)
            ax.set_title(f"{vname} × {col}")
            ax.set_xlabel(f"{vname} 1")
            ax.set_ylabel(f"{vname} 2")
    plt.tight_layout()
    plt.show()


def plot_centroid_heatmap(km, k, rfm, cols=None, cluster_col="Cluster_KM"):
    """K-means 簇中心热力图：0=全体均值，红=偏高，蓝=偏低。"""
    cols = ["Recency", "Frequency", "Monetary"] if cols is None else cols
    center_df = pd.DataFrame(km.cluster_centers_, columns=cols)
    center_df.index = [f"簇{i}" for i in range(k)]
    center_df["客户数"] = rfm[cluster_col].value_counts().sort_index().values
    fig, ax = plt.subplots(figsize=(7.5, 4))
    sns.heatmap(
        center_df.iloc[:, : len(cols)], annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        yticklabels=[f"簇{i} (n={n:,})" for i, n in enumerate(center_df["客户数"])],
        linewidths=0.5, ax=ax, cbar_kws={"label": "标准化值 (0=全体均值)"},
    )
    ax.set_title("K-means 簇中心画像（log1p+标准化 空间）")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(imp_df):
    """特征区分度 η² 横向柱状图（η² 最大的排最上面）。"""
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.barh(imp_df["特征"], imp_df["η²"], color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("η²（组间方差占比，越大越能区分簇）")
    ax.set_title("RFM 特征对簇的区分度（log1p 空间）")
    for i, r in imp_df.iterrows():
        ax.text(r["η²"] + 0.01, i, f"η²={r['η²']:.3f}  F={r['F值']:.0f}", va="center")
    plt.tight_layout()
    plt.show()


def plot_boxplot_by_cluster(rfm, labels, cols=None):
    """簇间 RFM 分布箱线图（log1p 空间，三个子图）。

    箱中位线=簇的典型水平；箱高(IQR)=簇内异质性；圆点=超出 1.5×IQR 的离群个体。
    同一特征内簇间可直接比高低（log1p 单调），三个子图之间不可比量纲。
    """
    cols = ["Recency", "Frequency", "Monetary"] if cols is None else cols
    view = np.log1p(rfm[cols]).copy()
    view["Cluster"] = labels
    fig, axes = plt.subplots(1, len(cols), figsize=(15, 4.5))
    for ax, feat in zip(axes, cols):
        view.boxplot(column=feat, by="Cluster", ax=ax, grid=False, patch_artist=True)
        ax.set_title(f"{feat} by 簇 (log1p)")
        ax.set_xlabel("簇")
        ax.set_ylabel(f"log1p({feat})")
    fig.suptitle("簇间 RFM 分布对比", y=1.03)
    plt.tight_layout()
    plt.show()


def plot_business_scatter(rfm, cluster_col):
    """业务视角散点：Recency × Monetary（log 轴），簇间连续过渡直观可见。"""
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(rfm["Recency"], rfm["Monetary"], c=rfm[cluster_col],
                    cmap="viridis", s=12, alpha=0.65)
    ax.set_yscale("log")  # Monetary 跨度大，log 轴才能看清整体结构
    ax.set_xlabel("Recency（距上次购买天数）")
    ax.set_ylabel("Monetary（总消费，log 轴）")
    fig.colorbar(sc, ax=ax, label="簇")
    ax.set_title("客户分群：Recency × Monetary（业务视角）")
    plt.tight_layout()
    plt.show()


# ---------------- 诊断 ----------------

def plot_stability(sizes, sil_mean, sil_std, ari_means):
    """K-means 稳定性双轴图：轮廓系数（左轴）随样本量收敛 + ARI（右轴）趋近 1。"""
    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax1.errorbar(sizes, sil_mean, yerr=sil_std, marker="o", capsize=4,
                 color="#4C72B0", label="轮廓系数 (均值±std)")
    ax1.set_xlabel("样本量 n")
    ax1.set_ylabel("轮廓系数", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(sizes, ari_means, marker="s", color="#C44E52", label="ARI (vs 全量分群)")
    ax2.set_ylabel("ARI（与全量分群的一致性）", color="#C44E52")
    ax1.set_title("K-means 稳定性 vs 样本量（20 次重复）")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    plt.tight_layout()
    plt.show()


def plot_eps_scan(eps_grid, n_clusters, noise_ratios):
    """DBSCAN eps 扫描双轴图：簇数（左轴）+ 噪声比例（右轴）。"""
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(eps_grid, n_clusters, marker="o", color="#4C72B0", label="簇数")
    ax1.set_xlabel("eps")
    ax1.set_ylabel("簇数", color="#4C72B0")
    ax1.set_xticks(eps_grid)
    for e, nc in zip(eps_grid, n_clusters):
        ax1.annotate(f"{nc}", (e, nc), textcoords="offset points",
                     xytext=(0, 6), ha="center", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(eps_grid, noise_ratios, marker="s", color="#C44E52", label="噪声比例 %")
    ax2.set_ylabel("噪声比例 %", color="#C44E52")
    ax1.set_title("DBSCAN eps 扫描")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    plt.tight_layout()
    plt.show()
