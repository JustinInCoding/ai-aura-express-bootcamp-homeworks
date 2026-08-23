# -*- coding: utf-8 -*-
"""
DBSCAN 聚类脚本（dbscan_clustering.py）
========================================
对应 analysis.ipynb 第 8 节（选 eps + 建模）与 11.6 节（eps 扫描诊断）。

功能：
- knn_distances(): 计算每个点到其第 min_pts 近邻的距离（K-距离图数据）
- fit_dbscan()   : 训练 DBSCAN，返回 (dbs, labels, n_noise)
- eps_scan()     : 扫描 eps 观察簇数/噪声比例变化，定位参数敏感区间

用法：
    from dbscan_clustering import knn_distances, fit_dbscan, eps_scan
    kdist = knn_distances(X_scaled, min_pts=5)
    dbs, labels, n_noise = fit_dbscan(X_scaled, eps=0.6, min_samples=5)

    python dbscan_clustering.py   # 独立运行全流程演示
"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def knn_distances(X, min_pts):
    """计算每个样本到其第 min_pts 近邻的距离，返回升序排序后的数组。

    画出排序曲线后，找「拐点」处对应的距离作为 eps 的参考值。
    """
    nn = NearestNeighbors(n_neighbors=min_pts)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    return np.sort(dists[:, -1])


def fit_dbscan(X, eps, min_samples):
    """训练 DBSCAN，返回 (dbs, labels, n_noise)。

    - labels 中 -1 表示噪声点（不属于任何簇的离群样本）
    - n_noise 为噪声点个数（用于计算噪声比例）
    """
    dbs = DBSCAN(eps=eps, min_samples=min_samples)
    dbs.fit(X)
    n_noise = int((dbs.labels_ == -1).sum())
    n_clusters = len(set(dbs.labels_)) - (1 if n_noise else 0)
    print(f"簇数: {n_clusters} | 噪声点: {n_noise:,} ({n_noise / len(X):.1%})")
    return dbs, dbs.labels_, n_noise


def eps_scan(X, eps_grid=None, min_samples=5):
    """扫描 eps，观察簇数与噪声比例变化，返回 (n_clusters_list, noise_ratio_list)。

    曲线没有稳定平台（略调小就碎成微簇、略调大就全部连通）说明数据密度均匀连续，
    DBSCAN 不适用——这正是本项目 RFM 数据的情况。
    """
    eps_grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2] if eps_grid is None else eps_grid
    n_clusters, noise_ratios = [], []
    for e in eps_grid:
        dbs = DBSCAN(eps=e, min_samples=min_samples)
        labels = dbs.fit_predict(X)
        n_clusters.append(len(set(labels)) - (1 if -1 in labels else 0))
        noise_ratios.append((labels == -1).mean() * 100)
    for e, nc, nr in zip(eps_grid, n_clusters, noise_ratios):
        print(f"eps={e}: 簇数={nc}  噪声比例={nr:.1f}%")
    return n_clusters, noise_ratios


def main():
    """全流程演示：清洗 → RFM → 变换 → KNN 距离图数据 → 拟合 → eps 扫描。"""
    from pathlib import Path

    from data_preprocessing import load_data, clean_data
    from feature_engineering import build_rfm, transform_features

    root = Path(__file__).resolve().parents[1]
    df = load_data(root / "data" / "OnlineRetail.csv")
    df_clean, _ = clean_data(df)
    rfm = build_rfm(df_clean)
    X_scaled, _, _ = transform_features(rfm)

    MIN_PTS = 5
    kdist = knn_distances(X_scaled, MIN_PTS)
    print(f"\nKNN 距离（k={MIN_PTS}）：拐点处对应距离 → eps 参考值")

    print("\n拟合 DBSCAN:")
    dbs, labels, n_noise = fit_dbscan(X_scaled, eps=0.6, min_samples=MIN_PTS)

    print("\neps 扫描诊断:")
    n_clusters, noise_ratios = eps_scan(X_scaled, min_samples=MIN_PTS)


if __name__ == "__main__":
    main()
