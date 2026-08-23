#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 Asthma.xlsx 中搜索真正匹配 radiomics PatientID 的列"""
import pandas as pd

df = pd.read_excel('Asthma.xlsx')
radi = pd.read_csv('radiomics_all_patients.csv',
                   dtype={'PatientID': str})
ids = set(radi['PatientID'].astype(str).str.lstrip('0'))
print(f'目标 PatientID 数: {len(ids)}, 样例: {list(ids)[:5]}')

# 列名重复情况
dup = df.columns[df.columns.duplicated()].tolist()
print(f'\n重复列名 ({len(dup)}):', dup[:10])

best = []
for c in df.columns:
    try:
        s = df[c].astype(str).str.lstrip('0').str.strip()
        hit = s.isin(ids).sum()
        if hit >= 70:
            best.append((c, hit))
    except Exception:
        pass
print('\n匹配率 >= 70/80 的列:')
for c, h in sorted(best, key=lambda x: -x[1])[:10]:
    print(f'  {repr(c)}: {h}/80')
