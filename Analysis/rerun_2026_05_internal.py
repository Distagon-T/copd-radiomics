#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rerun_2026_05_internal.py
==========================
在 2026-05 内用整合特征表重现"原任务"的内部 CV，检验之前的好特征效果是否还在。

对比基线：
  - 原 BCOS 报告(run_bcos_2026_05.py)：AECOPD(文本标签 BCOS_AE_Label) CV AUC ≈ 0.584±0.064
  - COPD+BCOS 研究(run_bcos_study_2026_05_v2.py)：COPD_BCOS Top100 CV AUC ≈ 0.698±0.068

任务/标签来源：
  A. AECOPD(原文标签)   : labels_bcos_2026_05.csv 的 BCOS_AE_Label
  B. AECOPD(一致ICD)    : labels_ae_bcos_2026_05.csv 的 AECOPD (J44.1|J44.0)
  C. COPD_BCOS          : labels_ae_bcos_2026_05.csv 的 COPD_BCOS

特征：整合表 rad | rad+aq；5 折 CV LR（与报告同参）。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOG = open(r"E:\DICOM\2026-05-seg\rerun_internal.log", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()

SEED = 42
SEG5 = r"E:\DICOM\2026-05-seg"
F5 = os.path.join(SEG5, "2026-05-integrated_radiomics_aq.csv")
L_ORIG = os.path.join(SEG5, "labels_bcos_2026_05.csv")
L_CONS = os.path.join(SEG5, "labels_ae_bcos_2026_05.csv")

AQ_PREFIX = ("TD_", "blur_", "wall_", "WA_", "Din_", "Dout_", "mean_",
             "Pi10", "Vessel_", "Lobe_", "Lung_", "Airway_", "PA_",
             "Diaphragm_", "pca_", "RV_", "LV_", "CAC_")
RAD_EXTRA = ("Lobe_", "Lung_", "Airway_", "PA_", "Diaphragm_", "heart",
             "aorta", "trachea", "pulmonary_artery")
META = {"Patient_ID", "PatientID", "Patient_ID_long", "PatientID_raw", "CT_Series",
        "patient_id", "ICD", "main_diagnosis", "AECOPD", "COPD_BCOS",
        "BCOS_AE_Label", "bcos", "患者id", "主要诊断", "n"}


def split_feats(cols):
    rad, aq = [], []
    for c in cols:
        if "::" in c:
            rad.append(c)
        elif any(c.startswith(p) for p in AQ_PREFIX):
            aq.append(c)
        elif any(c.startswith(p) for p in RAD_EXTRA):
            rad.append(c)
    return rad, aq


def run_cv(X, y, tag):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    scaler = StandardScaler()
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                 class_weight="balanced", max_iter=2000, random_state=SEED)
        clf.fit(scaler.fit_transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(scaler.transform(X[te]))[:, 1]))
    log(f"[{tag}] n_feat={X.shape[1]} n={len(y)} pos={int(y.sum())} "
        f"CV AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}")
    return np.mean(aucs), np.std(aucs)


def main():
    t0 = time.time()
    log("===== 2026-05 内部重现原任务（整合特征表） =====")
    df = pd.read_csv(F5)
    log(f"整合特征表: {df.shape[0]} 行 x {df.shape[1]} 列")

    shared = [c for c in df.columns if c not in META]
    rad, aq = split_feats(shared)
    log(f"rad={len(rad)} aq={len(aq)}")

    # 特征矩阵（一次构建）
    Xf = df[shared].apply(pd.to_numeric, errors="coerce")
    med = Xf.median().fillna(0)
    Xf = Xf.fillna(med).fillna(0).values.astype(np.float64)
    rad_idx = [shared.index(c) for c in rad]
    Xrad = Xf[:, rad_idx]

    # Patient_ID -> 特征表行号 映射
    pid2row = {p: i for i, p in enumerate(df["Patient_ID"].astype(str))}

    # ---- A. AECOPD 原文标签 ----
    lab = pd.read_csv(L_ORIG)
    m = df.merge(lab[["Patient_ID", "BCOS_AE_Label", "COPD_BCOS"]], on="Patient_ID", how="inner")
    m = m.drop_duplicates(subset=["Patient_ID"])
    y = m["BCOS_AE_Label"].values
    rowsA = np.array([pid2row[p] for p in m["Patient_ID"].astype(str)])
    log(f"\nA. AECOPD(原文文本标签 BCOS_AE_Label): join {len(m)} 行")
    log(f"   分布: pos={int(y.sum())} neg={int((y==0).sum())} (阳性率 {y.mean():.1%})")
    run_cv(Xrad[rowsA], y, "A|rad")
    run_cv(Xf[rowsA], y, "A|rad+aq")

    # ---- B. AECOPD 一致ICD标签 ----
    lc = pd.read_csv(L_CONS)
    m2 = df.merge(lc[["Patient_ID", "AECOPD", "COPD_BCOS"]], on="Patient_ID", how="inner")
    m2 = m2.drop_duplicates(subset=["Patient_ID"])
    yb = m2["AECOPD"].values
    k = ~np.isnan(yb)
    yb = yb[k].astype(int)
    rowsB = np.array([pid2row[p] for p in m2["Patient_ID"].astype(str)])
    log(f"\nB. AECOPD(一致ICD标签 J44.1|J44.0): join {len(m2)} 行, 有效 {int(k.sum())}")
    log(f"   分布: pos={int(yb.sum())} neg={int((yb==0).sum())} (阳性率 {yb.mean():.1%})")
    run_cv(Xrad[rowsB[k]], yb, "B|rad")
    run_cv(Xf[rowsB[k]], yb, "B|rad+aq")

    # ---- C. COPD_BCOS ----
    yc = m2["COPD_BCOS"].fillna(0).values.astype(int)
    rowsC = rowsB
    log(f"\nC. COPD_BCOS(合并支扩): join {len(m2)} 行")
    log(f"   分布: pos={int(yc.sum())} neg={int((yc==0).sum())} (阳性率 {yc.mean():.1%})")
    run_cv(Xrad[rowsC], yc, "C|rad")
    run_cv(Xf[rowsC], yc, "C|rad+aq")

    log(f"\n对比基线: 原报告 AECOPD CV≈0.584±0.064 ; COPD+BCOS 研究 CV≈0.698±0.068")
    log(f"总耗时 {time.time()-t0:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
