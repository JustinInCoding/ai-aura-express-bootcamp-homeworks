# -*- coding: utf-8 -*-
"""
K-means 聚类脚本（kmeans_clustering.py）
=========================================
对应 analysis.ipynb 第 7 节（选 K + 建模）与 11.5 节（稳定性测试）。

功能：
- select_k()      : 肘部法（inertia）+ 轮廓系数双方法选 K，返回 (inertias, sils)
- fit_kmeans()    : 用选定 K 训练最终模型，返回 (km, labels)
- stability_test(): 抽样不同样本量重复聚类，检验分群是否收敛（11.5）

用法：
    from kmeans_clustering import select_k, fit_kmeans
    inertias, sils = select_k(X_scaled, range(2, 11))
    km, labels = fit_kmeans(X_scaled, k=4)

    python kmeans_clustering.py   # 独立运行全流程演示
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


def select_k(X, k_range=range(2, 11), random_state=42, n_init=10):
    """用肘部法 + 轮廓系数选 K，返回 (inertias, sils)。

    - inertia：簇内平方和（SSE），越小越紧凑，曲线「肘点」为参考
    - silhouette：轮廓系数，越大越好，取最大值对应的 K 为参考
    两个方法综合判断（如本项目都指向 K=4 附近）。
    """
    inertias, sils = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        km.fit(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, km.labels_))
    for k, i, s in zip(k_range, inertias, sils):
        print(f"K={k}: inertia={i:,.0f}  silhouette={s:.4f}")
    return inertias, sils


def fit_kmeans(X, k, random_state=42, n_init=10):
    """用选定 K 训练最终 K-means，返回 (km, labels)。

    n_init=10 表示重复 10 次初始化取最优，random_state 固定种子保证可复现。
    """
    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    km.fit(X)
    return km, km.labels_


def stability_test(X, k, sizes=None, reps=20, random_state=42):
    """K-means 稳定性测试：从全量随机抽样不同样本量重复聚类。

    返回 (sizes, sil_mean, sil_std, ari_means)：
    - 轮廓系数均值/标准差：质量是否随样本量收敛
    - ARI：抽样分群 vs 全量分群的一致性（趋近 1 说明结果可复现）
    """
    rng = np.random.default_rng(random_state)
    sizes = [500, 1000, 2000, 3000, len(X)] if sizes is None else sizes
    ref_km, ref_labels = fit_kmeans(X, k, random_state=random_state)
    sil_mean, sil_std, ari_means = [], [], []
    for n in sizes:
        sil_tmp, ari_tmp = [], []
        for _ in range(reps):
            idx = rng.choice(len(X), size=n, replace=False)
            kk = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X[idx])
            sil_tmp.append(silhouette_score(X[idx], kk.labels_))
            ari_tmp.append(adjusted_rand_score(ref_labels[idx], kk.labels_))
        sil_mean.append(np.mean(sil_tmp))
        sil_std.append(np.std(sil_tmp))
        ari_means.append(np.mean(ari_tmp))
    for n, sm, sd, a in zip(sizes, sil_mean, sil_std, ari_means):
        print(f"n={n}: 轮廓系数={sm:.4f}±{sd:.4f}  ARI={a:.3f}")
    return sizes, sil_mean, sil_std, ari_means


def main():
    """全流程演示：清洗 → RFM → 变换 → 选 K → 拟合 → 稳定性。"""
    from pathlib import Path

    from data_preprocessing import load_data, clean_data
    from feature_engineering import build_rfm, transform_features

    root = Path(__file__).resolve().parents[1]
    df = load_data(root / "data" / "OnlineRetail.csv")
    df_clean, _ = clean_data(df)
    rfm = build_rfm(df_clean)
    X_scaled, _, _ = transform_features(rfm)

    print("\n选 K:")
    inertias, sils = select_k(X_scaled, range(2, 11))

    K_BEST = 4
    print(f"\n选定 K={K_BEST}")
    km, labels = fit_kmeans(X_scaled, K_BEST)
    print("簇大小:", np.bincount(labels).tolist())

    print("\n稳定性测试:")
    sizes, sil_mean, sil_std, ari_means = stability_test(X_scaled, K_BEST)


if __name__ == "__main__":
    main()
