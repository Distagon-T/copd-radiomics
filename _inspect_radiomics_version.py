#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""判断 radiomics_2026_05_features.csv 是 全量版 还是 Lite 版：按特征类型统计列数"""
import sys
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CSV = r"E:\DICOM\2026-05-seg\radiomics_2026_05_features.csv"
df = pd.read_csv(CSV, nrows=3)
cols = [c for c in df.columns]
print(f"总列数: {len(cols)}")

# 特征类型分类（按 pyRadiomics 命名约定）
def classify(c):
    low = c.lower()
    if "::original_shape" in low or "::shape" in low or c.startswith("Lobe_") or \
       c.startswith("Airway_") or c.startswith("Diaphragm_") or c.startswith("PA_") or \
       c.startswith("RV_") or c.startswith("LV_") or c.startswith("CAC_") or \
       c.startswith("Lung_"):
        return "shape/自定义表型"
    if "::original_firstorder" in low:
        return "original_firstorder"
    if "glcm" in low:
        return "GLCM纹理"
    if "glrlm" in low:
        return "GLRLM纹理"
    if "glszm" in low:
        return "GLSZM纹理"
    if "gldm" in low:
        return "GLDM纹理"
    if "ngtdm" in low:
        return "NGTDM纹理"
    if "wavelet" in low:
        return "Wavelet滤波"
    if low.startswith("log") or "-log-" in low or "log-sigma" in low:
        return "LoG滤波"
    if "squareroot" in low or "square" in low or "logarithm" in low or "exponential" in low:
        return "其他滤波"
    return "其他"

from collections import Counter
cnt = Counter(classify(c) for c in cols)
print("\n=== 特征类型分布 ===")
for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

# 采样展示
print("\n=== 非 shape/firstorder 的样例列 ===")
samples = [c for c in cols if classify(c) not in ("shape/自定义表型", "original_firstorder", "其他")]
for c in samples[:20]:
    print(f"  {c}")
if not samples:
    print("  （无 —— 纯 shape + firstorder + 自定义表型，即 Lite 特征集）")
