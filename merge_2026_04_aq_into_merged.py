# -*- coding: utf-8 -*-
"""
merge_2026_04_aq_into_merged.py
================================
把 2026-04 新算出的 AQ 特征（airway_features_all.csv，MATLAB compute_airway_features.m
产物，474 例 x 68 特征）中「合并表尚缺失」的值，填充进合并表
radiomics_2026_04_clinical_merged.csv（500 例 x 2669 列）。

规则：
  - 连接键：合并表 Patient_ID == AQ 表 patient_folder（均为患者文件夹名）。
  - 只「填 NaN」：合并表 AQ 列中当前为空的单元格，若新表有值则填充；
    已有值一律保留（不做覆盖），避免破坏既有结果。
  - 新表 68 个 AQ 列在合并表中都已存在（无新增列），仅补值。

用法：
  python merge_2026_04_aq_into_merged.py
  python merge_2026_04_aq_into_merged.py \
      --merged D:/copd-radiomics/radiomics_2026_04_clinical_merged.csv \
      --aq E:/DICOM/2026-04-Airway_features/airway_features_all.csv
"""
import os
import sys
import shutil
import argparse

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_MERGED = r"D:\copd-radiomics\radiomics_2026_04_clinical_merged.csv"
DEFAULT_AQ = r"E:\DICOM\2026-04-Airway_features\airway_features_all.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=DEFAULT_MERGED, help="合并表 CSV 路径")
    ap.add_argument("--aq", default=DEFAULT_AQ, help="新 AQ 特征表 CSV 路径")
    ap.add_argument("--no-backup", action="store_true", help="不备份原合并表")
    args = ap.parse_args()

    if not os.path.exists(args.merged):
        sys.exit(f"[err] 找不到合并表: {args.merged}")
    if not os.path.exists(args.aq):
        sys.exit(f"[err] 找不到新 AQ 表: {args.aq}")

    print("=== 读取合并表 ===")
    m = pd.read_csv(args.merged, dtype={"Patient_ID": str, "PatientID": str}, low_memory=False)
    print(f"  合并表: {m.shape[0]} 行 x {m.shape[1]} 列")

    print("=== 读取新 AQ 表 ===")
    a = pd.read_csv(args.aq, dtype={"patient_folder": str})
    aq_cols = [c for c in a.columns if c != "patient_folder"]
    print(f"  AQ 表: {a.shape[0]} 行 x {a.shape[1]} 列 | AQ 特征列: {len(aq_cols)}")

    a = a.drop_duplicates(subset=["patient_folder"], keep="first")
    a = a.set_index("patient_folder")

    # 只处理合并表中存在的 AQ 列；新表有而合并表没有的列 => 补列（本次为 0）
    exist_cols = [c for c in aq_cols if c in m.columns]
    new_cols = [c for c in aq_cols if c not in m.columns]
    print(f"  AQ 列在合并表已存在: {len(exist_cols)} | 需新增列: {len(new_cols)}")

    # 填充：合并表该单元格为 NaN 且新表有值 => 填入
    total_filled = 0
    per_col = {}
    m_index = m.index
    m_patients = m["Patient_ID"].astype(str)

    for c in new_cols:
        m[c] = pd.NA  # 新增列占位

    for c in aq_cols:
        if c not in m.columns:
            continue
        col_series = m[c]
        if col_series.isna().sum() == 0:
            per_col[c] = 0
            continue
        # 新表该列有效值映射（患者 -> 值）
        a_vals = a[c].dropna()
        if a_vals.empty:
            per_col[c] = 0
            continue
        # 取需要填充的位置：合并表为空 且 患者在新表有值
        mask = col_series.isna() & m_patients.isin(a_vals.index)
        if mask.sum() == 0:
            per_col[c] = 0
            continue
        # 用 map 填充
        fill_map = a_vals.to_dict()
        new_vals = m_patients[mask].map(fill_map)
        m.loc[mask, c] = new_vals.values
        n = int(mask.sum())
        per_col[c] = n
        total_filled += n

    print("\n=== 填充汇总（各 AQ 列本次填充的单元格数）===")
    for c in aq_cols:
        n = per_col.get(c, 0)
        if n > 0:
            print(f"  {c:<26} +{n}")
    print(f"  共填充 {total_filled} 个单元格")

    # 关键列前后对比
    print("\n=== 关键列非空数对比 ===")
    for c in ["Pi10", "n_branches", "TD_fwhm_all", "blur_contrast_mean", "wall_hu_mean", "WA_pct_all"]:
        if c in m.columns:
            print(f"  {c:<22} 现在非空={int(m[c].notna().sum())}")

    # 备份 + 写回
    if (os.path.abspath(args.merged) == os.path.abspath(DEFAULT_MERGED)) and not args.no_backup:
        bak = args.merged + ".bak"
        if not os.path.exists(bak):
            shutil.copyfile(args.merged, bak)
            print(f"\n已备份原合并表 -> {bak}")
    m.to_csv(args.merged, index=False, encoding="utf-8-sig")
    print(f"已写回: {args.merged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
