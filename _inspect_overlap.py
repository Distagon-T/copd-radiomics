#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import numpy as np
import pandas as pd

PY = sys.executable
OVERLAP = "overlap.xlsx"
INFO = "info.csv"

print("=== overlap xlsx ===")
xl = pd.ExcelFile(OVERLAP)
print("sheets:", xl.sheet_names)
ov = pd.read_excel(OVERLAP)
print("shape:", ov.shape)
print("cols:", list(ov.columns))
print(ov.head(5).to_string())

# 找 overlap 标签列（值含 0/1）
id_col = None
lab_col = None
for c in ov.columns:
    cs = str(c)
    if "id" in cs.lower() or "患者" in cs or "住院" in cs:
        id_col = c
    if "overlap" in cs.lower() or "支扩" in cs or "COPD合并" in cs or "合并支扩" in cs:
        lab_col = c
print("\nid_col:", id_col, "| lab_col:", lab_col)
if lab_col:
    print("overlap label value_counts:", ov[lab_col].value_counts().to_dict())

print("\n=== info-2026-05.csv ===")
info = pd.read_csv(INFO, encoding="utf-8")
print("shape:", info.shape, "| cols:", list(info.columns))

print("\n=== 合并 overlap 到 info（按 患者id）===")
if id_col and lab_col:
    info["_pid"] = info["患者id"].astype(str).str.strip().str.lstrip("0")
    ov["_pid"] = ov[id_col].astype(str).str.strip().str.lstrip("0")
    m = info.merge(ov[["_pid", lab_col]], on="_pid", how="left")
    print("overlap 匹配:", m[lab_col].notna().sum(), "/", len(m))
    print("overlap=1 患者数:", int((m[lab_col] == 1).sum()))
    print("overlap=0 患者数:", int((m[lab_col] == 0).sum()))
    print("overlap=NaN 患者数:", int(m[lab_col].isna().sum()))
    # 保存合并结果供后续使用
    m.to_csv("_info_with_overlap.csv", index=False, encoding="utf-8-sig")
    print("已保存 -> _info_with_overlap.csv")

print("\n=== 主要诊断 唯一值分布 (全部) ===")
vc = info["主要诊断"].fillna("").astype(str).value_counts()
print("唯一诊断数:", len(vc))
for d, n in vc.items():
    print(f"  {n:4d}  {str(d)[:60]}")
