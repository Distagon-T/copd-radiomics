#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_vessel_features_2026_05.py
===============================
评估 2026-05 新增肺血管高级特征（Vessel_*）的分类能力：
对多个标签任务计算单变量 AUC / Cohen's d / p(MWU)。
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_fusion_model import univariate_summary

SEG = r"E:\DICOM\2026-05-seg"
VESSEL_CSV = os.path.join(SEG, "radiomics_2026_05_features_vessel.csv")

VESSEL_COLS = [
    "Vessel_Fractal_Dim", "Vessel_BV5_pct", "Vessel_BV10_pct",
    "Vessel_Skeleton_Voxels", "Vessel_Skeleton_Length_mm",
    "Vessel_Branch_Count", "Vessel_Junction_Count", "Vessel_Endpoint_Count",
    "Vessel_Branching_Density_per_mm", "Vessel_Tortuosity_Mean",
    "Vessel_Tortuosity_Max",
]

TASKS = [
    ("labels_bronch_hemoptysis_2026_05.csv", "HEMO_Label", "支扩: 咯血 vs 无咯血"),
    ("labels_copd_ae_cause_2026_05.csv", "AE_CAUSE_Label", "急性COPD: 感染型 vs 非感染型"),
    ("labels_bcos_phenotype_2026_05.csv", "PHENO_Label", "BCOS 表型: BCOS vs PureCOPD"),
    ("labels_copd_acute_2026_05.csv", "COPD_AE_Label", "纯COPD: 急性加重 vs 稳定"),
    ("labels_nsfc_2026_05.csv", "NSFC_AE_Label", "NSFC: 急性 vs 稳定"),
]


def main():
    rad = pd.read_csv(VESSEL_CSV)
    idcol = "Patient_ID" if "Patient_ID" in rad.columns else rad.columns[0]
    rad[idcol] = rad[idcol].astype(str)
    print(f"vessel CSV: {len(rad)} 行, {VESSEL_COLS[0]} 非空 {rad[VESSEL_COLS[0]].notna().sum()}")

    summary_rows = []
    for fname, label_col, task in TASKS:
        lab = pd.read_csv(os.path.join(SEG, fname))
        lab["Patient_ID"] = lab["Patient_ID"].astype(str)
        lab = lab[lab[label_col].notna()].copy()
        lab[label_col] = lab[label_col].astype(int)
        df = rad.merge(lab[["Patient_ID", label_col]], on="Patient_ID", how="inner")
        if len(df) < 20:
            print(f"\n[{task}] 样本过少 {len(df)}，跳过")
            continue
        y = df[label_col].values
        pos = int(y.sum())
        print(f"\n[{task}] n={len(df)} pos={pos} neg={len(y)-pos}")
        uni = univariate_summary(df, VESSEL_COLS, y, top=30)
        if len(uni):
            uni = uni.sort_values("auc_univ", key=lambda s: (s - 0.5).abs(), ascending=False)
            uni["task"] = task
            uni["n"] = len(df)
            uni.to_csv(os.path.join(SEG, f"vessel_univariate_{fname}.csv"),
                       index=False, encoding="utf-8-sig")
            for _, r in uni.head(6).iterrows():
                print(f"  {r['feature']:34s} AUC={r['auc_univ']:.3f} d={r['cohens_d']:+.2f} "
                      f"p={r['p_mwu']:.2g}")
            summary_rows.append(uni)
        else:
            print("  (无有效特征)")

    if summary_rows:
        allu = pd.concat(summary_rows, ignore_index=True)
        allu.to_csv(os.path.join(SEG, "vessel_features_univariate_ALL.csv"),
                    index=False, encoding="utf-8-sig")
        # 每个任务的最佳特征
        best = []
        for _, g in allu.groupby("task"):
            g = g.sort_values("auc_univ", key=lambda s: (s - 0.5).abs(), ascending=False)
            r = g.iloc[0]
            best.append({"task": r["task"], "n": r["n"],
                         "feature": r["feature"], "auc": round(r["auc_univ"], 3),
                         "cohens_d": round(r["cohens_d"], 2), "p_mwu": r["p_mwu"]})
        bdf = pd.DataFrame(best)
        print("\n=== 每任务最佳 Vessel_* 特征 ===")
        print(bdf.to_string(index=False))
        bdf.to_csv(os.path.join(SEG, "vessel_features_best_per_task.csv"),
                   index=False, encoding="utf-8-sig")
        print(f"\n已保存: vessel_features_univariate_ALL.csv / vessel_features_best_per_task.csv")


if __name__ == "__main__":
    main()
