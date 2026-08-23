#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比两种 Label=0 定义：全保留队列 vs 仅 J44(COPD) 主诊断"""
import sys
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

INFO = "info.csv"
OVERLAP = "overlap.xlsx"
RAD = "radiomics_2026_05_features.csv"

info = pd.read_csv(INFO, encoding="utf-8")
ov = pd.read_excel(OVERLAP)
info["_pid"] = info["患者id"].astype(str).str.strip().str.lstrip("0")
ov["_pid"] = ov["患者id"].astype(str).str.strip().str.lstrip("0")
m = info.merge(ov[["_pid", "COPD合并支扩", "主要诊断-ICD码"]], on="_pid", how="left")
m["bcos"] = (m["COPD合并支扩"] == 1).astype(int)
m["_diag"] = m["主要诊断"].fillna("").astype(str)
m["_icd"] = m["主要诊断-ICD码"].fillna("").astype(str)
m["is_bronch"] = m["_diag"].str.startswith(("支气管扩张", "支气管扩张症", "细支气管扩张症"))

# --- 当前跑法: Label0 = 非支扩主诊断 且 非 BCOS ---
cur_kept = m[(~m["is_bronch"]) | (m["bcos"] == 1)].copy()
cur_lab0 = cur_kept[cur_kept["bcos"] == 0]
print("=== 当前跑法 Label=0 (463) ===")
print(f"n = {len(cur_lab0)}")

# --- 新定义: Label0 = 主诊断 J44 开头 且 非 BCOS ---
new_lab0 = m[(m["_icd"].str.startswith("J44")) & (m["bcos"] == 0)]
print("\n=== 新定义 Label=0 (J44 主诊断 且 非 BCOS) ===")
print(f"n = {len(new_lab0)}")

print("\n=== 当前 Label0 中 ICD 码分布（前 3 位）===")
print(cur_lab0["_icd"].str[:3].value_counts().to_string())

print("\n=== 当前 Label0 中 各主诊断 + ICD 前缀 ===")
grp = cur_lab0.groupby(["_diag", "_icd"]).size().reset_index(name="n")
print(grp.to_string(index=False))

# --- 差异: 当前有而新定义无 ---
diff = cur_lab0[~cur_lab0["_pid"].isin(new_lab0["_pid"])]
print("\n=== 被新定义排除的患者（当前 Label0 含，新定义不含）===")
print(f"n = {len(diff)}")
if len(diff):
    print(diff.groupby(["_diag", "_icd"]).size().reset_index(name="n").to_string(index=False))

# radiomics 覆盖
rad = pd.read_csv(RAD, usecols=["Patient_ID", "PatientID"])
rad["_pid"] = rad["PatientID"].astype(str).str.strip().str.lstrip("0")
pid = set(rad["_pid"])
print(f"\n=== radiomics 覆盖 ===")
print(f"当前 Label0 有 radiomics: {cur_lab0['_pid'].isin(pid).sum()}/{len(cur_lab0)}")
print(f"新 Label0 有 radiomics: {new_lab0['_pid'].isin(pid).sum()}/{len(new_lab0)}")

# BCOS(78) 里主诊断是 COPD 的 37 例
bcos = m[m["bcos"] == 1]
print("\n=== BCOS 78 的主诊断 ICD 前缀 ===")
print(bcos["_icd"].str[:3].value_counts().to_string())
print("\nBCOS 中主诊断 COPD(J44):", (bcos["_icd"].str.startswith("J44")).sum(),
      "| 主诊断支扩(J47):", (bcos["_icd"].str.startswith("J47")).sum())
