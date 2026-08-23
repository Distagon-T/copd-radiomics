#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查心血管检验单位与分布"""
import pandas as pd

df = pd.read_excel('Asthma.xlsx')
csv = pd.read_csv('radiomics_all_patients.csv', dtype={'PatientID': str})
ids = set(csv['PatientID'].astype(str).str.lstrip('0'))
sub = df[df['患者id'].astype(str).str.lstrip('0').isin(ids)].copy()

# 单位列
print('=== 单位列 ===')
for c in df.columns:
    if ('肌钙蛋白' in str(c) or '钠尿肽' in str(c)) and '单位' in str(c):
        vals = sub[c].value_counts().head(5)
        print(f'{c} -> {dict(vals)}')
print()

# 定量结果分布
print('=== 定量结果分布 ===')
for c in ['肌钙蛋白ITnI测定-定量结果', 'N端_B型钠尿肽前体NT_ProBNP测定-定量结果']:
    v = pd.to_numeric(sub[c], errors='coerce')
    print(f'{c}: n={v.notna().sum()} min={v.min():.4g} median={v.median():.4g} '
          f'max={v.max():.4g} p90={v.quantile(.9):.4g} p95={v.quantile(.95):.4g}')
    print('  前10个值:', v.dropna().head(10).tolist())
print()

# 检查结果（定性）分布
print('=== 定性检查结果分布 ===')
for c in ['肌钙蛋白ITnI测定-检查结果', 'N端_B型钠尿肽前体NT_ProBNP测定-检查结果']:
    print(f'{c}:')
    print(sub[c].value_counts().head(8).to_string())
    print()

# 肌酸激酶同工酶
print('=== CK-MB ===')
for c in ['肌酸激酶同工酶测定-定量结果', '肌酸激酶同工酶测定-检查结果']:
    if c in df.columns:
        v = pd.to_numeric(sub[c], errors='coerce')
        print(f'{c}: n={v.notna().sum()}')
        if v.notna().sum() > 0:
            print(f'  min={v.min():.4g} median={v.median():.4g} max={v.max():.4g}')
print()

# 年龄列
print('=== 年龄列名确认 ===')
age_cols = [c for c in df.columns if '年龄' in str(c) and ('岁' in str(c) or len(str(c)) <= 6)]
print(age_cols[:5])
if age_cols:
    print(sub[age_cols[0]].describe().to_string())
