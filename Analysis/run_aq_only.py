#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法: python run_aq_only.py   # aq(AirQuant)特征单独评估
run_aq_only.py
===============
评估 aq（AirQuant 气道）特征到底能不能用：
  - aq-only LR（只用 aq 特征）: 05 内部 CV + 外验 02/01
  - aq TopK 特征筛选 -> 外验
  - aq 单变量 Top
任务: AECOPD / COPD_BCOS / J44.0_vs_J44.9
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

LOG = open(r"E:\DICOM\2026-02-seg\aq_only.log", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()

SEED = 42
META = {"Patient_ID", "PatientID", "PatientID_raw", "Patient_ID_long", "CT_Series",
        "patient_id", "ICD", "main_diagnosis", "AECOPD", "COPD_BCOS", "患者id"}
PATHS = {
    "05": (r"E:\DICOM\2026-05-seg\2026-05-integrated_radiomics_aq.csv",
           r"E:\DICOM\2026-05-seg\labels_ae_bcos_2026_05.csv"),
    "02": (r"E:\DICOM\2026-02-seg\2026-02-integrated_radiomics_aq.csv",
           r"E:\DICOM\2026-02-seg\labels_ae_bcos_2026_02.csv"),
    "01": (r"E:\DICOM\2026-01-seg\2026-01-integrated_radiomics_aq.csv", None),
}
AQ_PREFIX = ("TD_", "blur_", "wall_", "WA_", "Din_", "Dout_", "mean_",
             "Pi10", "Vessel_", "Lobe_", "Lung_", "Airway_", "PA_",
             "Diaphragm_", "pca_", "RV_", "LV_", "CAC_")


def load(tag):
    f, l = PATHS[tag]
    df = pd.read_csv(f)
    if l:
        lab = pd.read_csv(l)
        m = df.merge(lab[["Patient_ID", "ICD", "AECOPD", "COPD_BCOS"]], on="Patient_ID", how="inner")
    else:
        m = df
    return m.drop_duplicates(subset=["Patient_ID"])


def make_y(m, task):
    icd = m["ICD"].astype(str).str.strip()
    if task == "AECOPD":
        return m["AECOPD"].values.astype(float)
    if task == "COPD_BCOS":
        return m["COPD_BCOS"].fillna(0).values.astype(float)
    return np.array([1 if x.startswith("J44.0") else (0 if x.startswith("J44.9") else np.nan)
                     for x in icd]).astype(float)


def uni_auc(X, y):
    from scipy.stats import rankdata
    pos = y == 1
    np_ = int(pos.sum()); nn = int((~pos).sum())
    Z = np.vstack([X[pos], X[~pos]])
    R = rankdata(Z, axis=0)
    Rpos = R[:np_].sum(0)
    return (Rpos - np_ * (np_ + 1) / 2) / (np_ * nn)


def run_cv(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    sc = StandardScaler()
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                 class_weight="balanced", max_iter=2000, random_state=SEED)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return np.mean(aucs), np.std(aucs)


def ext(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                             class_weight="balanced", max_iter=2000, random_state=SEED)
    clf.fit(sc.fit_transform(Xtr), ytr)
    return roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1])


def main():
    t0 = time.time()
    log("===== aq(AirQuant) 特征单独评估 =====")
    m5 = load("05"); m2 = load("02"); m1 = load("01")
    shared_all = [c for c in m1.columns if c in m2.columns and c in m5.columns and c not in META]
    aq_all = [c for c in shared_all if any(c.startswith(p) for p in AQ_PREFIX)]
    log(f"共享特征 {len(shared_all)}，其中 aq 特征 {len(aq_all)}")

    def build(m):
        X = m[shared_all].apply(pd.to_numeric, errors="coerce")
        med = X.median().fillna(0)
        return X.fillna(med).fillna(0).values.astype(np.float64)
    X5 = build(m5); X2 = build(m2); X1 = build(m1)
    aq_idx = [shared_all.index(c) for c in aq_all]

    for task in ["AECOPD", "COPD_BCOS", "J44.0_vs_J44.9"]:
        y5 = make_y(m5, task); y2 = make_y(m2, task); y1 = make_y(m1, task)
        k5 = ~np.isnan(y5); k2 = ~np.isnan(y2); k1 = ~np.isnan(y1)
        y5 = y5[k5].astype(int); y2 = y2[k2].astype(int); y1 = y1[k1].astype(int)
        X5q = X5[k5][:, aq_idx]; X2q = X2[k2][:, aq_idx]; X1q = X1[k1][:, aq_idx]
        log(f"\n##### {task} | aq-only (n_aq={len(aq_all)}) #####")
        log(f"  05 n={len(y5)} pos={int(y5.sum())} | 02 n={len(y2)} pos={int(y2.sum())} | 01 n={len(y1)} pos={int(y1.sum())}")

        # aq 单变量 top
        au = uni_auc(X5q, y5)
        order = np.argsort(-np.abs(au - 0.5))
        log("  aq 单变量 Top10:")
        for i in order[:10]:
            log(f"     {aq_all[i][:44]:<46} AUC={au[i]:.3f}")
        log(f"  aq 单变量 AUC>0.55 或 <0.45 的数量: "
            f"{int((np.abs(au-0.5)>0.05).sum())}/{len(au)}")

        # aq-only 全量
        c, s = run_cv(X5q, y5)
        e2 = ext(X5q, y5, X2q, y2); e1 = ext(X5q, y5, X1q, y1)
        log(f"  [aq-only 全量 n={len(aq_all)}] CV={c:.3f}±{s:.3f} | 02={e2:.3f} 01={e1:.3f}")
        # aq TopK
        for K in [10, 20, 50]:
            idx = order[:K]
            c, s = run_cv(X5q[:, idx], y5)
            e2 = ext(X5q[:, idx], y5, X2q[:, idx], y2)
            e1 = ext(X5q[:, idx], y5, X1q[:, idx], y1)
            log(f"  [aq Top{K}] CV={c:.3f}±{s:.3f} | 02={e2:.3f} 01={e1:.3f}")

    log(f"总耗时 {time.time()-t0:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
