#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 2026-05 radiomics PatientID 与 myInfo.csv 患者id 的匹配情况"""
import glob
import json

import pandas as pd

# 1) radiomics PatientID
pids = []
for f in glob.glob("seg_results/*_radiomics.json"):
    try:
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        if r.get("PatientID"):
            pids.append(str(r["PatientID"]).strip())
    except Exception:
        pass
pids = sorted(set(pids))
print(f"radiomics PatientID 唯一值: {len(pids)}")
print("样例:", pids[:10])

def norm(s):
    s = str(s).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.lstrip("0")

pids_n = sorted({norm(p) for p in pids})
print("归一化样例:", pids_n[:10])

# 2) myInfo.csv 患者id
info = pd.read_csv(r"myInfo.csv", encoding="gb18030", usecols=["患者id"])
info_ids = set(info["患者id"].dropna().astype(str).str.strip())
info_n = set()
for v in info_ids:
    x = v
    if x.endswith(".0"):
        x = x[:-2]
    info_n.add(x.lstrip("0"))
print(f"\nmyInfo 患者id 唯一: {len(info_n)}")
print("myInfo 归一化样例:", sorted(list(info_n))[:10])

# 交集
both = set(pids_n) & info_n
print(f"\n归一化后交集: {len(both)}")
both_raw = set(pids) & info_ids
print(f"原始字符串交集: {len(both_raw)}")
print("\n'1006234' in myInfo 归一化:", "1006234" in info_n)
print("'33' in radiomics 归一化:", "33" in set(pids_n))
