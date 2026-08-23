#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 2026-05 radiomics summary + E:/asthma.xlsx 结构"""
import json
import pandas as pd

# 1) radiomics summary
summ = r"E:\DICOM\2026-05-seg\radiomics_all_patients.json"
with open(summ, encoding="utf-8") as f:
    recs = json.load(f)
print(f"=== radiomics_all_patients.json: {len(recs)} 条 ===")
r0 = recs[0]
keys = list(r0.keys())
print(f"字段数: {len(keys)}")
print("前 12 个键:", keys[:12])
print("Patient_ID:", r0.get("Patient_ID"))
print("PatientID:", r0.get("PatientID"))
print("CT_Series:", r0.get("CT_Series"))
feat_keys = [k for k in keys if "::" in k]
print(f"特征列数: {len(feat_keys)}")
# PatientID 样例
pids = sorted({str(r.get("PatientID")) for r in recs if r.get("PatientID")})
print(f"有 PatientID 的患者: {len(pids)}")
print("PatientID 样例:", pids[:8])
nonempty_pid = sum(1 for r in recs if r.get("PatientID"))
print(f"PatientID 非空: {nonempty_pid}/{len(recs)}")

# 2) E:/asthma.xlsx 结构
xl = r"E:\asthma.xlsx"
df = pd.read_excel(xl, nrows=5)
print(f"\n=== {xl} ===")
print(f"列数: {len(df.columns)}")
print("前 20 列:", df.columns[:20].tolist())
# 找 ID 列
for c in df.columns[:40]:
    v = df[c].dropna().head(3).tolist()
    print(f"  {c!r}: {v}")
