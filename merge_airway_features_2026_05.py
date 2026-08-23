#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_airway_features_2026_05.py
================================
把 MATLAB compute_airway_features.m 输出的 airway_features_all.csv（新 schema，
含 FWHM 边界模糊度 blur_*、管壁密度 wall_hu_*、T/D 急剧变化 TD_*、FWHM 版 T/D TD_fwhm_*）
并入 2026-05 AirQuant 聚合表 airquant_2026_05_aggregated.csv（aq_* 基础 49 特征）。
新增列统一加 "aq_" 前缀，使 plot_2026_05_results.py / plot_consistency_2026_05.py /
check_pi10.py / train_fusion_model.py --airquant 零改动自动纳入。

前置步骤：
  1) 在 MATLAB 里把 compute_airway_features.m 的
     METRICS_DIR = 'E:\\DICOM\\2026-05-Airway_metrics_tmp'
     FEATURES_DIR = 'E:\\DICOM\\2026-05-Airway_features'
     跑完后生成 E:\\DICOM\\2026-05-Airway_features\\airway_features_all.csv
  2) 已有 E:\\DICOM\\2026-05-seg\\airquant_2026_05_aggregated.csv（aggregate_airquant.py 产物）

用法：
  python merge_airway_features_2026_05.py
  python merge_airway_features_2026_05.py \
      --matlab-feats E:\\DICOM\\2026-05-Airway_features\\airway_features_all.csv \
      --airquant     E:\\DICOM\\2026-05-seg\\airquant_2026_05_aggregated.csv \
      --out          E:\\DICOM\\2026-05-seg\\airquant_2026_05_aggregated.csv
"""
import argparse
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEG = r"E:\DICOM\2026-05-seg"
DEFAULT_MATLAB_FEATS = r"E:\DICOM\2026-05-Airway_features\airway_features_all.csv"
DEFAULT_AIRQUANT = os.path.join(SEG, "airquant_2026_05_aggregated.csv")

# 这些 MATLAB 列与基础 aq_* 已有等价列（aq_Pi10 / aq_n_branches），跳过避免重复
SKIP_COLS = {"patient_folder", "Pi10", "n_branches"}


def match_patient_id(name, id_set):
    """文件夹名 -> 基础表 patient 键对齐（与 aggregate_airquant.py 一致）。"""
    name = str(name)
    if name in id_set:
        return name
    # 去掉结尾 '.数字' 后缀再试（如 'XXX.1' -> 'XXX'）
    m = re.sub(r"\.\d+$", "", name)
    if m in id_set:
        return m
    # 前缀匹配
    for pid in id_set:
        if name.startswith(pid):
            return pid
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matlab-feats", default=DEFAULT_MATLAB_FEATS,
                    help="MATLAB compute_airway_features.m 输出的 airway_features_all.csv")
    ap.add_argument("--airquant", default=DEFAULT_AIRQUANT,
                    help="基础 AirQuant 聚合表（patient + aq_*）")
    ap.add_argument("--out", default=DEFAULT_AIRQUANT,
                    help="输出路径（默认覆盖 airquant 表，先自动备份 _base.csv）")
    ap.add_argument("--no-backup", action="store_true",
                    help="覆盖时不做 _base.csv 备份")
    args = ap.parse_args()

    if not os.path.exists(args.airquant):
        sys.exit(f"[err] 找不到基础 AirQuant 表: {args.airquant}")
    base = pd.read_csv(args.airquant)
    if "patient" not in base.columns:
        sys.exit(f"[err] 基础表缺少 patient 列，现有列: {list(base.columns)[:10]}")
    print(f"基础 AirQuant 表: {len(base)} 行, {base.shape[1]} 列 (aq_* 基础特征)")

    if not os.path.exists(args.matlab_feats):
        print(f"[warn] 未找到 MATLAB 特征输出: {args.matlab_feats}")
        print("       请先在 MATLAB 跑 compute_airway_features.m（METRICS_DIR/FEATURES_DIR 指向 2026-05），")
        print("       再重跑本脚本。当前仅保留基础 aq_* 特征（无新增列）。")
        return 0

    mat = pd.read_csv(args.matlab_feats)
    if "patient_folder" not in mat.columns:
        sys.exit(f"[err] MATLAB 特征表缺少 patient_folder 列，现有列: {list(mat.columns)[:10]}")
    print(f"MATLAB 特征表: {len(mat)} 行, {mat.shape[1]} 列")

    # 对齐键：MATLAB patient_folder -> 基础表 patient
    id_set = set(base["patient"].dropna().astype(str))
    mat["_key"] = mat["patient_folder"].map(lambda x: match_patient_id(x, id_set))

    # 选出新增列：去掉跳过列 + 与基础已存在 aq_* 冲突的列
    new_cols = []
    for c in mat.columns:
        if c == "_key" or c in SKIP_COLS:
            continue
        aq_c = "aq_" + c
        if aq_c in base.columns:
            print(f"[skip] {c} 与基础列 {aq_c} 冲突，跳过")
            continue
        new_cols.append(c)
    print(f"新增 aq_ 列数: {len(new_cols)}")

    if new_cols:
        add = mat[["_key"] + new_cols].rename(columns={"_key": "patient"})
        add = add.rename(columns={c: "aq_" + c for c in new_cols})
        merged = base.merge(add, on="patient", how="left")
        new_aq = ["aq_" + c for c in new_cols]
        n_matched = int(merged[new_aq].notna().any(axis=1).sum())
        n_rows_new = int(merged[new_aq].notna().all(axis=1).sum())
    else:
        merged = base.copy()
        n_matched = 0
        n_rows_new = 0
    print(f"合并后: {len(merged)} 行, {merged.shape[1]} 列")
    print(f"至少一个新特征非空的患者: {n_matched} | 全部新特征非空的患者: {n_rows_new}")

    # 覆盖输出前备份
    if (os.path.abspath(args.out) == os.path.abspath(args.airquant)
            and not args.no_backup):
        bak = args.airquant + "_base.csv"
        if not os.path.exists(bak):
            shutil.copyfile(args.airquant, bak)
            print(f"已备份基础表 -> {bak}")

    merged.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"输出: {args.out}")

    # 关键新特征预览（非缺失患者数）
    show = [c for c in [
        "aq_blur_peak_hu_mean", "aq_blur_contrast_mean", "aq_blur_trans_width_mean",
        "aq_blur_edge_sharp_mean", "aq_wall_hu_mean", "aq_TD_ratio_std_all",
        "aq_TD_slope_vs_gen", "aq_TD_outlier_ratio_z2", "aq_TD_distal_minus_proximal",
        "aq_TD_fwhm_std_gen5plus",
    ] if c in merged.columns]
    if show:
        nonnull = merged[show].notna().sum()
        print("\n新特征非缺失患者数预览:")
        for c in show:
            print(f"  {c:<32} {int(nonnull[c]):>4} / {len(merged)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
