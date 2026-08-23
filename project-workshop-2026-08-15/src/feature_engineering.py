# -*- coding: utf-8 -*-
"""
RFM 特征工程脚本（feature_engineering.py）
============================================
对应 analysis.ipynb 第 5~6 节：RFM 构建 + log1p 变换 + 标准化。

功能：
- build_rfm()         : 按 CustomerID 聚合生成客户级 RFM 特征矩阵
- transform_features(): log1p 压缩右偏长尾 + StandardScaler 统一量纲

用法：
    from feature_engineering import build_rfm, transform_features
    rfm = build_rfm(df_clean, ref_date="2011-12-09")
    X_scaled, scaler, rfm_log = transform_features(rfm)

    python feature_engineering.py   # 独立运行全流程演示
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_preprocessing import load_data, clean_data


def build_rfm(df, ref_date="2011-12-09"):
    """按 CustomerID 聚合构建 RFM 特征矩阵，返回 rfm DataFrame。

    - Recency  : 距基准日（观察窗口末日）最近一次购买的天数，越小越活跃
    - Frequency: 不同订单数（InvoiceNo 去重计数）
    - Monetary : 总消费金额（Quantity × UnitPrice 求和）

    注意：InvoiceDate 含时间戳（末日为 2011-12-09 12:50），计算天数差前需
    normalize() 归到当日零点，否则 12-09 当天购买的客户会出现 Recency = -1。
    """
    REF_DATE = pd.Timestamp(ref_date)
    df = df.copy()
    df["Amount"] = df["Quantity"] * df["UnitPrice"]                       # 每行交易金额
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (REF_DATE - x.max().normalize()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Amount", "sum"),
    )
    print(f"RFM 客户数: {len(rfm):,}")
    return rfm


def transform_features(rfm, cols=None):
    """RFM 特征变换：先 log1p 压缩右偏长尾，再 StandardScaler 统一量纲。

    返回 (X_scaled, scaler, rfm_log)：
    - X_scaled: numpy 数组，聚类用的特征矩阵（均值0、方差1）
    - scaler  : 已拟合的 StandardScaler（可用于新数据 transform）
    - rfm_log : log1p 后的 DataFrame（可视化用）

    为什么两步都要：
    - log1p 解决分布偏态（RFM 严重右偏，极端客户会主导欧氏距离）
    - 标准化 解决量纲统一（聚类基于距离，对尺度敏感）
    """
    cols = list(rfm.columns) if cols is None else cols
    rfm_log = np.log1p(rfm[cols])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_log)
    print("X_scaled shape:", X_scaled.shape)
    return X_scaled, scaler, rfm_log


def main():
    root = Path(__file__).resolve().parents[1]
    df = load_data(root / "data" / "OnlineRetail.csv")
    df_clean, _ = clean_data(df)
    rfm = build_rfm(df_clean)
    print("\nRFM 统计摘要:")
    print(rfm.describe().to_string())
    X_scaled, scaler, rfm_log = transform_features(rfm)
    print("特征列:", list(rfm_log.columns))


if __name__ == "__main__":
    from pathlib import Path
    main()
