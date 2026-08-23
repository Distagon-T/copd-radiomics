#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 myInfo.csv 全表搜索 radiomics PatientID；并查看 radiomics json 的可识别字段"""
import glob
import json

import pandas as pd

# radiomics 样例 PatientID
sample_pids = []
for f in sorted(glob.glob("seg_results/*_radiomics.json"))[:5]:
    try:
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        sample_pids.append(str(r.get("PatientID")))
        print("=== json keys ===")
        print("Patient_ID:", r.get("Patient_ID"))
        print("PatientID:", r.get("PatientID"))
        print("CT_Series:", r.get("CT_Series"))
        print("其他非特征键:", [k for k in r.keys() if "::" not in k and k not in
                              ("Patient_ID", "PatientID", "CT_Series")][:20])
        print()
    except Exception as e:
        print(e)
print("样例 PatientID:", sample_pids)

# myInfo 全表搜索
df = pd.read_csv(r"myInfo.csv", encoding="gb18030", low_memory=False)
print(f"\nmyInfo: {len(df)} 行 x {len(df.columns)} 列")
targets = ["1006234", "1035996", "1502092", "1550034"]
for t in targets:
    found_cols = []
    for c in df.columns:
        try:
            if (df[c].astype(str).str.contains(t, na=False)).any():
                found_cols.append(c)
        except Exception:
            pass
    print(f"'{t}' 出现在列: {found_cols[:8]}")
