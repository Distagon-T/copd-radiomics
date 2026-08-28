#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法: python missing_stats.py   # 缺失统计 / 需要补跑清单
missing_stats.py
=================
统计 2026-01/02/05 特征表的缺失情况，输出"需要补跑"清单：
  1) 标签患者 vs 特征表覆盖（哪些患者完全没特征）
  2) 每列缺失率分布 / 全空列（按特征块 rad/aq/vessel/lobe）
  3) 每行缺失率 / 全空行
  4) 2026-01 气道缺失患者名单
输出: E:\DICOM\reports\missing_stats.log + missing_patients.csv
"""
import os
import sys
import glob
import json

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOG = open(r"E:\DICOM\reports\missing_stats.log", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()

META = {"Patient_ID", "PatientID", "PatientID_raw", "Patient_ID_long", "CT_Series",
        "patient_id", "ICD", "main_diagnosis", "AECOPD", "COPD_BCOS", "患者id"}
AQ_PREFIX = ("TD_", "blur_", "wall_", "WA_", "Din_", "Dout_", "mean_",
             "Pi10", "Vessel_", "Lobe_", "Lung_", "Airway_", "PA_",
             "Diaphragm_", "pca_", "RV_", "LV_", "CAC_")


def feat_block(c):
    if c.startswith("Vessel_"):
        return "vessel"
    if c.startswith("Lobe_") or c.startswith("Lung_"):
        return "lobe/lung(肺气肿)"
    if c.startswith(AQ_PREFIX) and "::" not in c:
        return "aq(气道)"
    if "::" in c:
        return "rad(影像组学)"
    return "其他"


def ana(tag, feat_csv, lab_path=None, lab_id_col="患者id", lab_pid_col="Patient_ID"):
    df = pd.read_csv(feat_csv)
    feat = [c for c in df.columns if c not in META]
    X = df[feat].apply(pd.to_numeric, errors="coerce")
    log(f"\n========== 2026-{tag} ==========")
    log(f"特征表: {len(df)} 行 x {len(feat)} 列")

    # 标签覆盖
    if lab_path:
        lab = pd.read_excel(lab_path) if lab_path.endswith("xlsx") else pd.read_csv(lab_path)
        lab_n = len(lab)
        feat_ids = set(df["Patient_ID"].astype(str).str.strip())
        if lab_pid_col in lab.columns:
            lab_ids = set(lab[lab_pid_col].astype(str).str.strip())
        elif lab_id_col in lab.columns:
            lab_ids = set(lab[lab_id_col].astype(str).str.strip().str.lstrip("0"))
        else:
            lab_ids = set()
        covered = len(lab_ids & feat_ids)
        missing_ids = lab_ids - feat_ids
        log(f"标签 {lab_n} 例; 特征表覆盖 {covered} 例; 完全无特征 {len(missing_ids)} 例")
    else:
        missing_ids = set()
        log("(无独立标签文件，特征表自带标签列)")

    # 列缺失
    mc = X.isna().mean()
    log(f"列缺失率分布: 全空={int((mc==1).sum())}  >50%={int((mc>0.5).sum())}  10-50%={int(((mc>=0.1)&(mc<0.5)).sum())}  <10%={int((mc<0.1).sum())}")
    allna = [c for c in feat if mc[c] == 1]
    if allna:
        from collections import Counter
        blk = Counter(feat_block(c) for c in allna)
        log(f"全空列共 {len(allna)}: 按块 {dict(blk)}")
        for c in allna[:8]:
            log(f"    {c[:60]}")
    # 高缺失列
    hi = [c for c in feat if 0.5 < mc[c] < 1]
    if hi:
        from collections import Counter
        blk2 = Counter(feat_block(c) for c in hi)
        log(f"50-100%缺失列 {len(hi)}: 按块 {dict(blk2)} 示例 {hi[:3]}")

    # 行缺失
    rm = X.isna().mean(axis=1)
    log(f"行缺失率: 中位 {rm.median()*100:.0f}%, 最大 {rm.max()*100:.0f}%, 全空行 {int((rm==1).sum())}, >50% {int((rm>0.5).sum())}")
    # 每块缺失
    for blk_name in ["aq(气道)", "vessel", "lobe/lung(肺气肿)"]:
        cols = [c for c in feat if feat_block(c) == blk_name]
        if cols:
            bm = X[cols].isna().mean(axis=1)
            log(f"  {blk_name}: {len(cols)}列, 有患者的块缺失率>50%: {int((bm>0.5).sum())} 例")
    return X, df, missing_ids


def main():
    log("===== 三队列特征缺失统计 / 需要补跑清单 =====")
    # 01
    X1, df1, miss1 = ana("01", r"E:\DICOM\2026-01-seg\2026-01-integrated_radiomics_aq.csv")
    # 01 气道覆盖
    af = sorted(glob.glob(r"E:\DICOM\2026-01-Airway_features\*.csv"))
    af = [a for a in af if not os.path.basename(a).lower().endswith("all.csv")]
    aq_ids = set(os.path.basename(a).split("_airway_features.csv")[0] for a in af)
    feat1 = set(df1["Patient_ID"].astype(str).str.strip())
    no_aq = feat1 - aq_ids
    log(f"\n[2026-01] airway 特征患者 {len(aq_ids)}/{len(feat1)}; 缺气道特征 {len(no_aq)} 例:")
    for pid in sorted(no_aq):
        log(f"    {pid}")
    # 01 无 radiomics
    js1 = glob.glob(r"E:\DICOM\2026-01-seg\*_radiomics.json")
    rad1 = set(os.path.basename(j).replace("_radiomics.json", "") for j in js1)
    no_rad1 = feat1 - rad1
    log(f"[2026-01] radiomics 患者 {len(rad1)}/{len(feat1)}; 缺 radiomics {len(no_rad1)} 例: {sorted(no_rad1)[:10]}")

    # 02
    X2, df2, miss2 = ana("02", r"E:\DICOM\2026-02-seg\2026-02-integrated_radiomics_aq.csv",
                         r"E:\DICOM\2026-02-seg\labels_ae_bcos_2026_02.csv", lab_pid_col="Patient_ID")
    # 05
    X5, df5, miss5 = ana("05", r"E:\DICOM\2026-05-seg\2026-05-integrated_radiomics_aq.csv",
                         r"E:\DICOM\2026-05-seg\labels_ae_bcos_2026_05.csv", lab_pid_col="Patient_ID")

    # 汇总需要补跑的名单
    rows = []
    for tag, miss in [("01", miss1), ("02", miss2), ("05", miss5)]:
        for pid in sorted(miss):
            rows.append({"cohort": tag, "patient": pid, "reason": "完全无特征(标签在但特征表无)"})
    for pid in sorted(no_aq):
        rows.append({"cohort": "01", "patient": pid, "reason": "缺气道(aq)特征"})
    for pid in sorted(no_rad1):
        rows.append({"cohort": "01", "patient": pid, "reason": "缺 radiomics"})
    pd.DataFrame(rows).to_csv(r"E:\DICOM\reports\missing_patients.csv",
                              index=False, encoding="utf-8-sig")
    log(f"\n需要补跑名单已写 -> E:\\DICOM\\reports\\missing_patients.csv ({len(rows)} 条)")
    LOG.close()


if __name__ == "__main__":
    main()
