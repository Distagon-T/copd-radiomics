#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Asthma.xlsx 中 80 例 radiomics 患者的行缓存为 pickle，避免反复读大文件"""
import sys

import pandas as pd

XLSX = "Asthma.xlsx"
RADI = "radiomics_all_patients.csv"
CACHE = "clinical_80_cache.pkl"

try:
    df = pd.read_excel(XLSX)
except MemoryError:
    print("MemoryError: xlsx 读取内存不足，请先暂停 compute_patient_radiomics_lite.py")
    sys.exit(2)

radi = pd.read_csv(RADI, dtype={"PatientID": str})
ids = set(radi["PatientID"].astype(str).str.lstrip("0"))
sub = df[df["患者id"].astype(str).str.lstrip("0").isin(ids)].copy()
print(f"匹配: {len(sub)} 行")
sub.to_pickle(CACHE)
print(f"缓存已写: {CACHE}")
