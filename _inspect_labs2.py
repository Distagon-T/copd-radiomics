#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import numpy as np
import pandas as pd

XLSX = "Asthma.xlsx"
LAB_NSFC = "labels_nsfc_2026_05.csv"

df = pd.read_excel(XLSX, nrows=200)
cols = [str(c) for c in df.columns]

print("=== 所有含 '嗜酸' 的列 ===")
for i, c in enumerate(cols):
    if "嗜酸" in c:
        print(f"  [{i}] {c}")

print("\n=== 嗜酸性粒细胞比例 定量结果 前 20 例 ===")
val_col = None
for i, c in enumerate(cols):
    if c == "嗜酸性粒细胞比例-定量结果":
        val_col = c
        break
if val_col:
    v = pd.to_numeric(df[val_col], errors="coerce")
    print(v.head(20).to_string())
    print("non-null:", v.notna().sum(), "/", len(v), "| range:", v.min(), "-", v.max())

print("\n=== 嗜酸性粒细胞比例 参考范围 / 单位 前 10 ===")
for suffix in ["-参考范围", "-单位"]:
    col = "嗜酸性粒细胞比例" + suffix
    if col in cols:
        print(col, ":", df[col].dropna().head(10).tolist())

# 关联 AECOPD 队列
print("\n=== 与 AECOPD(NSFC=1) 关联 ===")
if os.path.exists(LAB_NSFC):
    lab = pd.read_csv(LAB_NSFC, dtype=str)
    lab["_pid"] = lab["patient_id"].astype(str).str.strip().str.lstrip("0")
    aecopd = lab[lab["NSFC_AE_Label"] == "1"]
    print("AECOPD 患者数(临床):", len(aecopd))
    ast = pd.DataFrame({"患者id": df["患者id"].astype(str).str.strip().str.lstrip("0"),
                        "eos_pct": pd.to_numeric(df[val_col], errors="coerce")} if val_col else {"患者id": []})
    m = aecopd.merge(ast, left_on="_pid", right_on="患者id", how="left")
    print("AECOPD 且有嗜酸比例值:", m["eos_pct"].notna().sum(), "/", len(m))
    if m["eos_pct"].notna().sum():
        vv = m["eos_pct"]
        print("eos_pct 分布: mean=%.2f median=%.2f p75=%.2f p90=%.2f max=%.2f" % (
            vv.mean(), vv.median(), vv.quantile(.75), vv.quantile(.90), vv.max()))
        for thr in [2, 3, 4, 5]:
            print(f"  >={thr}% : {int((vv >= thr).sum())} 例 ({((vv >= thr).mean()*100):.1f}%)")
