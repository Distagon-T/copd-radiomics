#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rerun_2026_05_bcos_task.py
==========================
2026-05 内 COPD_BCOS 分类任务复现（独立小脚本，防静默 kill）：
  标签   : labels_ae_bcos_2026_05.csv 的 COPD_BCOS (合并支扩==1)
  特征   : (a) 研究筛出的 Top100 特征 (b) 整合表 rad+aq 全量
  方法   : 5 折 CV LR（每折即 log，中途被杀也能看到进度）
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

LOG = open(r"E:\DICOM\2026-05-seg\rerun_bcos_task.log", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()

SEED = 42
SEG5 = r"E:\DICOM\2026-05-seg"
F5 = os.path.join(SEG5, "2026-05-integrated_radiomics_aq.csv")
L_CONS = os.path.join(SEG5, "labels_ae_bcos_2026_05.csv")
TOP = os.path.join(SEG5, "bcos_screening_univariate_top.csv")

AQ_PREFIX = ("TD_", "blur_", "wall_", "WA_", "Din_", "Dout_", "mean_",
             "Pi10", "Vessel_", "Lobe_", "Lung_", "Airway_", "PA_",
             "Diaphragm_", "pca_", "RV_", "LV_", "CAC_")
RAD_EXTRA = ("Lobe_", "Lung_", "Airway_", "PA_", "Diaphragm_", "heart",
             "aorta", "trachea", "pulmonary_artery")
META = {"Patient_ID", "PatientID", "Patient_ID_long", "PatientID_raw", "CT_Series",
        "patient_id", "ICD", "main_diagnosis", "AECOPD", "COPD_BCOS"}


def run_cv_foldwise(X, y, tag):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    scaler = StandardScaler()
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                 class_weight="balanced", max_iter=2000, random_state=SEED)
        clf.fit(scaler.fit_transform(X[tr]), y[tr])
        a = roc_auc_score(y[te], clf.predict_proba(scaler.transform(X[te]))[:, 1])
        aucs.append(a)
        log(f"   [{tag}] fold{fold}: AUC={a:.3f}")
    log(f"[{tag}] n_feat={X.shape[1]} n={len(y)} pos={int(y.sum())} "
        f"CV AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}")
    return np.mean(aucs)


def main():
    t0 = time.time()
    log("===== 2026-05 内 COPD_BCOS 任务复现 =====")
    df = pd.read_csv(F5)
    lab = pd.read_csv(L_CONS)
    m = df.merge(lab[["Patient_ID", "COPD_BCOS"]], on="Patient_ID", how="inner")
    m = m.drop_duplicates(subset=["Patient_ID"])
    y = m["COPD_BCOS"].fillna(0).values.astype(int)
    log(f"join {len(m)} 行, pos={int(y.sum())} neg={int((y==0).sum())} (阳性率 {y.mean():.1%})")
    pid2row = {p: i for i, p in enumerate(df["Patient_ID"].astype(str))}
    rows = np.array([pid2row[p] for p in m["Patient_ID"].astype(str)])

    allf = [c for c in df.columns if c not in META]
    Xfull = df[allf].apply(pd.to_numeric, errors="coerce")
    med = Xfull.median().fillna(0)
    Xfull = Xfull.fillna(med).fillna(0).values.astype(np.float64)
    X = Xfull[rows]
    log(f"全特征矩阵 {X.shape}")

    # (a) 研究 Top100 特征
    top = pd.read_csv(TOP).head(100)
    feat100 = [f for f in top["feature"] if f in df.columns]
    log(f"研究Top100特征可用: {len(feat100)}")
    idx100 = [allf.index(c) for c in feat100]
    log("--- (a) 研究 Top100 特征 ---")
    run_cv_foldwise(X[:, idx100], y, "Top100")

    # (b) 全量 rad+aq
    log("--- (b) 全量 rad+aq ---")
    run_cv_foldwise(X, y, "rad+aq")

    log(f"总耗时 {time.time()-t0:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
