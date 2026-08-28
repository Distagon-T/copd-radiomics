#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_3cohort_validate.py
========================
训练 2026-05（一致 ICD 标签）-> 外部验证 2026-01 与 2026-02（分开 + 合并）。
任务: AECOPD (J44.1|J44.0) 和 COPD_BCOS。
特征: 三队列共享（2026-01 ⊆ 02 ⊆ 05，共享 = 01 的特征集 ~2273）。
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

LOG = open(r"E:\DICOM\2026-02-seg\validate_3cohort.log", "w", encoding="utf-8")
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
    "01": (r"E:\DICOM\2026-01-seg\2026-01-integrated_radiomics_aq.csv",
           None),  # 01 表内已含 AECOPD/COPD_BCOS
}


def load(tag):
    f, l = PATHS[tag]
    df = pd.read_csv(f)
    if l:
        lab = pd.read_csv(l)
        m = df.merge(lab[["Patient_ID", "AECOPD", "COPD_BCOS"]], on="Patient_ID", how="inner")
    else:
        m = df
    m = m.drop_duplicates(subset=["Patient_ID"])
    return m


def train_ext(Xtr, ytr, Xte, yte, tag):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                             class_weight="balanced", max_iter=2000, random_state=SEED)
    clf.fit(scaler.fit_transform(Xtr), ytr)
    p = clf.predict_proba(scaler.transform(Xte))[:, 1]
    a = roc_auc_score(yte, p)
    log(f"  -> 2026-{tag} 外部 AUC = {a:.3f}")
    return a


def cv5(X, y):
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
    log(f"  2026-05 内部 CV AUC = {np.mean(aucs):.3f}±{np.std(aucs):.3f}")
    return np.mean(aucs), np.std(aucs)


def main():
    t0 = time.time()
    log("===== 三队列外部验证: 训练2026-05 -> 验证2026-01/2026-02 =====")
    m5 = load("05"); m2 = load("02"); m1 = load("01")
    log(f"样本: 05={len(m5)} 02={len(m2)} 01={len(m1)}")

    shared = [c for c in m1.columns if c in m2.columns and c in m5.columns and c not in META]
    log(f"三队列共享特征: {len(shared)}")

    def build(m):
        X = m[shared].apply(pd.to_numeric, errors="coerce")
        med = X.median().fillna(0)
        return X.fillna(med).fillna(0).values.astype(np.float64)
    X5 = build(m5); X2 = build(m2); X1 = build(m1)
    log(f"矩阵: 05={X5.shape} 02={X2.shape} 01={X1.shape}")

    for task in ["AECOPD", "COPD_BCOS"]:
        y5 = m5[task].values
        k5 = ~np.isnan(y5)
        y5 = y5[k5].astype(int); X5t = X5[k5]
        y2 = m2[task].values
        k2 = ~np.isnan(y2)
        y2 = y2[k2].astype(int); X2t = X2[k2]
        y1 = m1[task].values
        k1 = ~np.isnan(y1)
        y1 = y1[k1].astype(int); X1t = X1[k1]

        log(f"\n##### {task} #####")
        log(f"  2026-05 训练: n={len(y5)} pos={int(y5.sum())} (阳性率 {y5.mean():.1%})")
        log(f"  2026-02 测试: n={len(y2)} pos={int(y2.sum())} (阳性率 {y2.mean():.1%})")
        log(f"  2026-01 测试: n={len(y1)} pos={int(y1.sum())} (阳性率 {y1.mean():.1%})")
        cv5(X5t, y5)
        train_ext(X5t, y5, X2t, y2, "02")
        train_ext(X5t, y5, X1t, y1, "01")
        # 合并 01+02
        Xc = np.vstack([X1t, X2t]); yc = np.concatenate([y1, y2])
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                 class_weight="balanced", max_iter=2000, random_state=SEED)
        clf.fit(sc.fit_transform(X5t), y5)
        ac = roc_auc_score(yc, clf.predict_proba(sc.transform(Xc))[:, 1])
        log(f"  -> 2026-01+02 合并 外部 AUC = {ac:.3f} (n={len(yc)})")

    log(f"总耗时 {time.time()-t0:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
