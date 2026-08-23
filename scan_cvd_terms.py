#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描 80 例 radiomics 患者的诊断类字段，统计心血管关键词命中"""
import sys
import os

import pandas as pd

OUT = open(r"E:\DICOM\2026-04-seg-part1\cvd_scan.log", "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.write(s + "\n")
    OUT.flush()


cache = r"E:\DICOM\2026-04-seg-part1\clinical_80_cache.pkl"
if os.path.exists(cache):
    log("[使用缓存 clinical_80_cache.pkl]")
    sub = pd.read_pickle(cache)
else:
    log("[缓存不存在，读 xlsx]")
    df = pd.read_excel(r'D:\copd-radiomics\Asthma.xlsx')
    radi = pd.read_csv(r'E:\DICOM\2026-04-seg-part1\radiomics_all_patients.csv',
                       dtype={'PatientID': str})
    ids = set(radi['PatientID'].astype(str).str.lstrip('0'))
    sub = df[df['患者id'].astype(str).str.lstrip('0').isin(ids)].copy()
    df = None

df = sub  # 保持下面逻辑一致

# 诊断类字段
diag_cols = [c for c in df.columns
             if ('诊断' in str(c)) or ('主诉' in str(c)) or ('现病史' in str(c))
             or ('既往史' in str(c))]
log('诊断类字段:')
for c in diag_cols:
    nn = sub[c].notna().sum()
    log(f'  {c}  非空 {nn}/80')
log('')

CVD_TERMS = [
    "心力衰竭", "心衰", "心功能不全", "心肌梗死", "心梗", "急性冠脉",
    "冠心病", "心绞痛", "心肌缺血", "房颤", "心房颤动", "心律失常",
    "心源性休克", "肺源性心脏病", "肺心病", "心包积液", "高血压", "高血压病",
    "瓣膜", "心肌病", "心室颤动", "室颤", "心脏骤停", "心肺复苏",
]
# 每个词在各列的命中数
hits = {}
for t in CVD_TERMS:
    n = 0
    for c in diag_cols:
        n += sub[c].astype(str).str.contains(t, na=False).sum()
    hits[t] = n
log('=== 关键词命中（跨所有诊断字段，可能同一患者多列命中） ===')
for t, n in sorted(hits.items(), key=lambda x: -x[1]):
    if n > 0:
        log(f'  {t}: {n}')

# 任一 CVD 词命中（含急性+慢性词）的患者数
ANY = ["心力衰竭", "心衰", "心功能不全", "心肌梗死", "心梗", "急性冠脉",
       "冠心病", "心绞痛", "心肌缺血", "房颤", "心房颤动", "心律失常",
       "心源性休克", "肺源性心脏病", "肺心病", "心包积液", "心肌病",
       "心室颤动", "室颤", "心脏骤停", "心肺复苏"]
text = sub[diag_cols].astype(str).agg(" ".join, axis=1)
hit_any = text.apply(lambda s: any(k in s for k in ANY))
log('')
log(f'任一 CVD 词命中患者数: {hit_any.sum()}/80')
OUT.close()

