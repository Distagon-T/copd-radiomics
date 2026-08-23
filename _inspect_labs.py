#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import pandas as pd

candidates = [
    r"E:\DICOM\2026-05\2026-5-9提取.xlsx",
    r"E:\DICOM\2026-05\2026-5-9提取.xls",
    r"D:\copd-radiomics\Asthma.xlsx",
]

KEYWORDS = ["嗜酸", "嗜酸性", "中性粒", "白细胞", "血红蛋白", "患者id", "住院号", "病历号"]

for f in candidates:
    if not os.path.exists(f):
        print(f"[missing] {f}")
        continue
    print(f"\n=== {f} ===")
    try:
        xl = pd.ExcelFile(f)
        print("sheets:", xl.sheet_names)
        # 取第一个 sheet 表头
        df0 = pd.read_excel(f, nrows=0)
        cols = [str(c) for c in df0.columns]
        print("n_cols:", len(cols))
        hits = [(i, c) for i, c in enumerate(cols) if any(k in c for k in KEYWORDS)]
        print("keyword hits (%d):" % len(hits))
        for i, c in hits[:40]:
            print(f"   [{i}] {c[:70]}")
    except Exception as e:
        print("  ERR:", e)
