#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速: 用 698 个真实 PatientID 对 myInfo 所有列做 isin 等值匹配, 逐列落盘"""
import glob
import json

import pandas as pd

OUT = open("scan_map3.log", "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.write(s + "\n")
    OUT.flush()


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
log(f"PatientID: {len(pids)} (归一化 {len(pids_n)})")

df = pd.read_csv(r"myInfo.csv", encoding="gb18030", low_memory=False)
log(f"myInfo: {len(df)} 行 x {len(df.columns)} 列")

found_any = False
for i, c in enumerate(df.columns):
    try:
        vals = df[c].dropna().astype(str).str.strip()
        hit = vals.isin(pids) | vals.isin(pids_n)
        n = int(hit.sum())
        if n > 0:
            log(f"列 {c!r}: {n} 命中")
            found_any = True
    except Exception as e:
        log(f"列 {c!r} 出错: {e}")
    if (i + 1) % 200 == 0:
        log(f"[进度 {i+1}/{len(df.columns)}]")
if not found_any:
    log("结果: 所有列均无 PatientID 命中")
log("完成")
OUT.close()
