#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用真实 2026-05 json 的 PatientID 全表搜索 myInfo.csv 所有列"""
import collections
import glob
import json

import pandas as pd

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
print(f"真实 json PatientID 数: {len(pids)}, 样例: {pids[:5]}")

df = pd.read_csv(r"myInfo.csv", encoding="gb18030", low_memory=False)
print(f"myInfo: {len(df)} 行 x {len(df.columns)} 列")

targets = pids[:10] + [p.lstrip("0") for p in pids[:10]]
print("\n前10个真实 PatientID 的命中:")
for t in targets:
    cols = []
    for c in df.columns:
        try:
            if (df[c].astype(str).str.contains(t, na=False, regex=False)).any():
                cols.append(c)
        except Exception:
            pass
    print(f"  '{t}' -> {cols[:6] if cols else '无'}")
