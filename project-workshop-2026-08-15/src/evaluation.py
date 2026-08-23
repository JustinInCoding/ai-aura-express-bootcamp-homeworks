# -*- coding: utf-8 -*-
"""
评估指标脚本（evaluation.py）
==============================
对应 analysis.ipynb 第 9.1 节（内部指标）、9.3 节（簇画像）、11.3 节（特征区分度 η²）。

功能：
- evaluate_clustering(): 计算单个聚类结果的内部指标（轮廓系数 / DBI / CH / 簇数 / 噪声）
- cluster_profiles()   : 各簇 RFM 中位数画像（业务解读用，比均值更抗离群点）
- feature_importance() : η² 特征区分度 + 单因素 ANOVA（聚类版「特征重要性」）

用法：
    from evaluation import evaluate_clustering, cluster_profiles, feature_importance
    row_km  = evaluate_clustering(X_scaled, rfm["Cluster_KM"].values, "K-means")
    profile = cluster_profiles(rfm, "Cluster_KM")
    imp_df  = feature_importance(rfm, km.labels_, k=4)
"""
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def evaluate_clustering(X, labels, name="model"):
    """计算聚类结果的内部评估指标，返回 dict。

    - 簇数：唯一标签数（含 -1 时把噪声那一类减掉）
    - 噪声点：标签为 -1 的样本数（DBSCAN 专有，K-means 恒为 0）
    - 轮廓系数：越大越好（-1 ~ 1，>0.5 表示结构清晰）
    - DBI：Davies-Bouldin 指数，越小越好（簇内紧凑、簇间分离）
    - CH：Calinski-Harabasz 指数，越大越好
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return {
        "算法": name,
        "簇数": n_clusters,
        "噪声点": int((labels == -1).sum()),
        "轮廓系数": round(silhouette_score(X, labels), 4),
        "DBI": round(davies_bouldin_score(X, labels), 4),
        "CH": round(calinski_harabasz_score(X, labels), 2),
    }


def cluster_profiles(rfm, cluster_col, cols=None, drop_noise=True):
    """各簇 RFM 中位数画像（+ 客户数列），返回 DataFrame。

    参数：
    - rfm: 含簇标签列的 RFM 表
    - cluster_col: 簇标签列名（如 "Cluster_KM" / "Cluster_DB"）
    - cols: 参与画像的数值列，默认 ["Recency", "Frequency", "Monetary"]
    - drop_noise: 为 True 时排除噪声点（-1），DBSCAN 画像应保留该默认值
    """
    cols = ["Recency", "Frequency", "Monetary"] if cols is None else cols
    has_noise = (rfm[cluster_col] == -1).any()
    sub = rfm[rfm[cluster_col] != -1] if drop_noise and has_noise else rfm
    profile = sub.groupby(cluster_col)[cols].median()
    profile["客户数"] = sub.groupby(cluster_col).size()
    return profile


def feature_importance(rfm, labels, k, feats=None):
    """特征区分度：η²（组间方差占比）+ 单因素 ANOVA，按 η² 降序返回 DataFrame。

    对每个特征在 log1p 空间计算（与聚类空间保持一致）：
    - η² = 组间平方和 / 总平方和，越接近 1 说明该特征越能区分簇（效应量，不受样本量影响）
    - F / p：ANOVA 检验簇间均值是否显著不同（大样本下 p 容易趋近 0，
            比较特征重要性应只看 η²）
    """
    feats = ["Recency", "Frequency", "Monetary"] if feats is None else feats
    rows = []
    for feat in feats:
        v = np.log1p(rfm[feat].values)
        ss_tot = ((v - v.mean()) ** 2).sum()
        groups = [v[labels == c] for c in range(k)]
        ss_bet = sum(len(g) * (g.mean() - v.mean()) ** 2 for g in groups)
        F, p = f_oneway(*groups)
        rows.append({"特征": feat, "η²": ss_bet / ss_tot, "F值": F, "p值": p})
    return (
        pd.DataFrame(rows)
        .sort_values("η²", ascending=False)
        .reset_index(drop=True)
    )


def main():
    """全流程演示：清洗 → RFM → 变换 → K-means → 三套评估。"""
    from pathlib import Path

    from data_preprocessing import load_data, clean_data
    from feature_engineering import build_rfm, transform_features
    from sklearn.cluster import KMeans

    root = Path(__file__).resolve().parents[1]
    df = load_data(root / "data" / "OnlineRetail.csv")
    df_clean, _ = clean_data(df)
    rfm = build_rfm(df_clean)
    X_scaled, _, _ = transform_features(rfm)

    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_scaled)
    rfm["Cluster_KM"] = km.labels_

    print("\n内部指标对比:")
    row = evaluate_clustering(X_scaled, rfm["Cluster_KM"].values, "K-means")
    print(pd.DataFrame([row]).set_index("算法").to_string())

    print("\n簇画像（中位数）:")
    print(cluster_profiles(rfm, "Cluster_KM").round(1).to_string())

    print("\n特征区分度 η²:")
    print(feature_importance(rfm, km.labels_, k=4).round(4).to_string())


if __name__ == "__main__":
    main()
