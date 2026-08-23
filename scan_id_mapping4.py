#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""严格验证: myInfo 每列 与 698 个 dicom PatientID 的多格式匹配（含 int/.0）"""
import glob
import json
import os

import pandas as pd

pids = set()
for f in glob.glob(r"E:\DICOM\2026-05-seg\*_radiomics.json"):
    try:
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        if r.get("PatientID"):
            pids.add(str(r["PatientID"]).strip())
    except Exception:
        pass
print(f"PatientID 数: {len(pids)}")

target_str = set(pids) | {p.lstrip("0") for p in pids} | {p + ".0" for p in pids if p.isdigit()}
target_int = set()
for p in pids:
    try:
        target_int.add(int(p))
    except Exception:
        pass
print(f"目标字符串格式: {len(target_str)}, int 格式: {len(target_int)}")

mtime = os.path.getmtime(r"E:\myInfo.csv")
print(f"myInfo.csv 修改时间: {pd.Timestamp(mtime, unit='s')}")
df = pd.read_csv(r"E:\myInfo.csv", encoding="gb18030", low_memory=False)
print(f"myInfo: {len(df)} 行 x {len(df.columns)} 列")

found = 0
for i, c in enumerate(df.columns):
    col = df[c]
    s = col.astype(str).str.strip()
    hit_s = s.isin(target_str).sum()
    hit_i = 0
    if col.dtype.kind in "iuf":
        hit_i = col.isin(target_int).sum()
    n = int(hit_s + hit_i)
    if n > 0:
        found += n
        print(f"列 {c!r}: {n} 命中 (str={int(hit_s)}, num={int(hit_i)})")
print(f"=== 扫描完成, 总命中 {found} ===")
