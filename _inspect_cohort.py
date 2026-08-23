#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd

M = r"E:\DICOM\2026-05-seg\_info_with_overlap.csv"
df = pd.read_csv(M)
df["_pid"] = df["患者id"].astype(str).str.strip().str.lstrip("0")
diag = df["主要诊断"].fillna("").astype(str)
ov = df["COPD合并支扩"]  # NaN 或 1

BRONCH = ["支气管扩张", "支气管扩张症", "细支气管扩张症"]  # 前缀判定纯支扩主诊断
def is_bronch(d):
    return any(d.startswith(b) for b in BRONCH)
def is_copd(d):
    return any(k in d for k in ["慢性阻塞性肺", "慢性支气管", "阻塞性支气管炎"])

df["is_bronch"] = diag.apply(is_bronch)
df["is_copd"] = diag.apply(is_copd)
df["bcos"] = (ov == 1).astype(int)

# 划分
pure_bronch = df[df["is_bronch"] & ~df["bcos"].astype(bool)]
kept = df[~(df["is_bronch"] & ~df["bcos"].astype(bool))]
print("纯支扩(剔除):", len(pure_bronch))
print("保留队列:", len(kept), "| 其中 BCOS(COPD合并支扩=1):", int(kept["bcos"].sum()))

# BCOS 的主诊断构成
print("\nBCOS(78) 的主诊断分布:")
print(kept[kept["bcos"] == 1]["主要诊断"].value_counts().to_string())

# 标签
ACUTE_KW = ["急性加重", "急性发作", "伴急性下呼吸道感染", "合并感染"]
def label(d):
    return 1 if any(k in d for k in ACUTE_KW) else 0
kept["BCOS_AE_Label"] = diag[kept.index].apply(label)
print("\n保留队列 BCOS_AE_Label 分布:", kept["BCOS_AE_Label"].value_counts().to_dict())
print("AECOPD 中 BCOS 占比:", int(((kept["BCOS_AE_Label"]==1)&(kept["bcos"]==1)).sum()), "/",
      int((kept["BCOS_AE_Label"]==1).sum()))
print("SCOPD 中 BCOS 占比:", int(((kept["BCOS_AE_Label"]==0)&(kept["bcos"]==1)).sum()), "/",
      int((kept["BCOS_AE_Label"]==0).sum()))

# 边缘情况
print("\n被标为 0(稳定) 的诊断分布:")
print(kept[kept["BCOS_AE_Label"]==0]["主要诊断"].value_counts().to_string())

# 与建模特征交集
rad = pd.read_csv(r"E:\DICOM\2026-05-seg\radiomics_2026_05_features.csv")
rad_pid = set(rad["PatientID"].astype(str).str.strip().str.lstrip("0"))
kept["has_radiomics"] = kept["_pid"].isin(rad_pid)
model = kept[kept["has_radiomics"]]
print("\n保留队列且有 radiomics 特征:", len(model))
print("其中 BCOS_AE_Label 分布:", model["BCOS_AE_Label"].value_counts().to_dict())
print("其中 BCOS 数:", int(model["bcos"].sum()))
