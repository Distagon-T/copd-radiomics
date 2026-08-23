#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 E:/myInfo.csv 列结构（GBK 编码）与 AirQuant 2026-05 schema"""
import glob

import pandas as pd

# myInfo.csv
df = pd.read_csv(r"myInfo.csv", encoding="gb18030", nrows=3)
print(f"=== myInfo.csv ===")
print(f"列数: {len(df.columns)}")
cols = list(df.columns)
for c in cols[:25]:
    print(f"  {c!r}: {df[c].dropna().head(3).tolist()}")
print("...")
# 找关键列
print("\n--- 关键列定位 ---")
for key in ["患者id", "年龄", "主要诊断", "其他诊断", "主诉", "肌钙蛋白", "NT_ProBNP", "BNP"]:
    hit = [c for c in cols if key in str(c)]
    print(f"  {key}: {hit[:6]}")
# 是否含患者id且非空
sub = df[["患者id"]].dropna() if "患者id" in cols else None
if sub is not None:
    print(f"\n患者id 非空: {len(sub)}, 样例: {sub['患者id'].head(5).tolist()}")

# AirQuant 2026-05 schema
aq = glob.glob("airway_metrics/*_airquant/*_full_metrics.csv")
print(f"\n=== AirQuant 2026-05 ===")
print(f"metrics csv: {len(aq)}")
if aq:
    m = pd.read_csv(aq[0])
    print(f"列: {list(m.columns)}")
    print(f"行数: {len(m)}")
