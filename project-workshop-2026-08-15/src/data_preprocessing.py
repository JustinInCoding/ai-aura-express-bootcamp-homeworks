# -*- coding: utf-8 -*-
"""
数据清洗与预处理脚本（data_preprocessing.py）
==============================================
对应 analysis.ipynb 第 1~4 节：环境配置、加载数据、数据清洗。

功能：
- load_data() : 读取 Online Retail 原始 CSV（latin-1 编码，转换 InvoiceDate 为 datetime）
- clean_data() : 4 类剔除（CustomerID 缺失 / Quantity≤0 / UnitPrice≤0 / C 开头取消单）
                 + 去重 + CustomerID 转 int，返回清洗后 df 与统计报告 dict

用法：
    # 作为模块复用
    from data_preprocessing import load_data, clean_data
    df = load_data("../data/OnlineRetail.csv")
    df_clean, report = clean_data(df)

    # 独立运行（打印完整清洗链路行数变化）
    python data_preprocessing.py
"""
from pathlib import Path

import pandas as pd


def load_data(path):
    """读取 Online Retail CSV 并解析日期，返回原始 DataFrame。

    文件为 latin-1 编码（含 £ 等字符），必须指定 encoding="latin-1"。
    """
    df = pd.read_csv(path, encoding="latin-1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    print(f"原始数据: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    print(f"时间跨度: {df['InvoiceDate'].min()} ~ {df['InvoiceDate'].max()}")
    return df


def clean_data(df, verbose=True):
    """数据清洗：4 类剔除 + 去重 + CustomerID 转 int。

    依次剔除：
    1) CustomerID 缺失（无法关联客户）
    2) Quantity <= 0（退货 / 负数）
    3) UnitPrice <= 0（0 元异常 / 赠品）
    4) InvoiceNo 以 "C" 开头（取消单）
    随后删除完全重复的行，并把 CustomerID 从 float 转 int（此时已无缺失，
    astype(int) 必须放在清洗之后，否则含 NaN 会报错）。

    返回 (清洗后 df, 统计报告 dict)；verbose=True 时打印每一步行数变化。
    """
    n_before = df.shape[0]
    df = df[df["CustomerID"].notna()]                                     # 条件1：客户ID非空
    df = df[df["Quantity"] > 0]                                           # 条件2：数量为正
    df = df[df["UnitPrice"] > 0]                                          # 条件3：单价为正
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]             # 条件4：非取消单
    n_after_dirty = df.shape[0]

    df = df.drop_duplicates()                                             # 删除完全重复行
    n_after_dedup = df.shape[0]
    df["CustomerID"] = df["CustomerID"].astype(int)

    report = {
        "清洗前行数": n_before,
        "剔除脏数据后行数": n_after_dirty,
        "剔除脏数据行数": n_before - n_after_dirty,
        "去重后行数": n_after_dedup,
        "去重剔除行数": n_after_dirty - n_after_dedup,
        "有效客户数": int(df["CustomerID"].nunique()),
    }
    if verbose:
        print(f"清洗前: {n_before:,} 行")
        print(f"剔除脏数据后: {n_after_dirty:,} 行 | {report['有效客户数']:,} 个客户")
        print(f"去重后: {n_after_dedup:,} 行（剔除 {n_after_dirty - n_after_dedup:,} 行完全重复）")
    return df, report


def main():
    root = Path(__file__).resolve().parents[1]          # 项目根目录（src/ 的上一级）
    df = load_data(root / "data" / "OnlineRetail.csv")
    df_clean, report = clean_data(df)
    print("\n清洗统计:")
    for k, v in report.items():
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
