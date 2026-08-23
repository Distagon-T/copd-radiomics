#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 info-2026-05.csv: 列结构、label 分布、与 dicom PatientID 匹配"""
import glob
import json

import pandas as pd

info = pd.read_csv("info-2026-05.csv", encoding="utf-8")
print(f"info-2026-05.csv: {len(info)} 行 x {len(info.columns)} 列")
print("列:", list(info.columns))
print()
print(info.head(10).to_string())
print()

lab = info["主要诊断"].astype(str).str.contains("急性加重", na=False)
print(f"主要诊断含'急性加重': {int(lab.sum())} / {len(info)}")
print("样例(阳性):")
for _, r in info[lab].head(5).iterrows():
    print(f"  {r['患者id']} | {str(r['主要诊断'])[:60]}")

ids_info = set(info["患者id"].dropna().astype(str).str.strip())
ids_info_n = {i.lstrip("0") for i in ids_info}
pids = set()
for f in glob.glob("seg_results/*_radiomics.json"):
    try:
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        if r.get("PatientID"):
            pids.add(str(r["PatientID"]).strip())
    except Exception:
        pass
pids_n = {p.lstrip("0") for p in pids}
print(f"\ninfo 患者id: {len(ids_info)} 唯一 | radiomics PatientID: {len(pids)}")
print(f"归一化交集: {len(ids_info_n & pids_n)} / {len(pids_n)}")
print("info 患者id 样例:", sorted(list(ids_info_n))[:8])
print("radiomics 样例:", sorted(list(pids_n))[:8])
