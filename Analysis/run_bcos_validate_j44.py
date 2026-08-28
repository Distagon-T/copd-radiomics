#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_bcos_validate_j44.py
=========================
COPD_BCOS 任务，仅限 J44(COPD) 患者（AECOPD 非空即 J44）：
  2026-05 内部 5 折 CV + 2026-05 -> 2026-02 外部验证
特征：rad | rad+aq | 研究Top100
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

LOG = open(r"E:\DICOM\2026-02-seg\validate_j44.log", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()

SEED = 42
SEG5 = r"E:\DICOM\2026-05-seg"
SEG2 = r"E:\DICOM\2026-02-seg"
F5 = os.path.join(SEG5, "2026-05-integrated_radiomics_aq.csv")
F2 = os.path.join(SEG2, "2026-02-integrated_radiomics_aq.csv")
L5 = os.path.join(SEG5, "labels_ae_bcos_2026_05.csv")
L2 = os.path.join(SEG2, "labels_ae_bcos_2026_02.csv")
TOP = os.path.join(SEG5, "bcos_screening_univariate_top.csv")

META = {"Patient_ID", "PatientID", "Patient_ID_long", "PatientID_raw", "CT_Series",
        "patient_id", "ICD", "main_diagnosis", "AECOPD", "COPD_BCOS"}


def load_j44(feat_path, lab_path, tag):
    df = pd.read_csv(feat_path)
    lab = pd.read_csv(lab_path)
    m = df.merge(lab[["Patient_ID", "AECOPD", "COPD_BCOS"]], on="Patient_ID", how="inner")
    m = m.drop_duplicates(subset=["Patient_ID"])
    k = ~pd.isna(m["AECOPD"])  # 仅 J44
    m = m[k].reset_index(drop=True)
    y = m["COPD_BCOS"].fillna(0).astype(int).values
    log(f"2026-{tag} J44: n={len(m)} COPD_BCOS pos={int(y.sum())} neg={int((y==0).sum())} "
        f"阳性率 {y.mean():.1%}")
    return m, y


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
    return np.mean(aucs), np.std(aucs)


def external(Xtr, ytr, Xte, yte, tag):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                             class_weight="balanced", max_iter=2000, random_state=SEED)
    clf.fit(scaler.fit_transform(Xtr), ytr)
    p = clf.predict_proba(scaler.transform(Xte))[:, 1]
    a = roc_auc_score(yte, p)
    log(f"[{tag}] 外部(2026-05->2026-02) AUC={a:.3f}")
    return a


def main():
    t0 = time.time()
    log("===== COPD_BCOS 任务 (J44 内) 内部CV + 外验 =====")
    m5, y5 = load_j44(F5, L5, "05")
    m2, y2 = load_j44(F2, L2, "02")

    shared = [c for c in m2.columns if c in m5.columns and c not in META]
    log(f"共享特征列: {len(shared)}")

    def build(Xm):
        X = Xm[shared].apply(pd.to_numeric, errors="coerce")
        med = X.median().fillna(0)
        return X.fillna(med).fillna(0).values.astype(np.float64)
    X5 = build(m5); X2 = build(m2)
    log(f"矩阵: 05={X5.shape} 02={X2.shape}")

    top = pd.read_csv(TOP).head(100)
    feat100 = [f for f in top["feature"] if f in shared]
    idx100 = [shared.index(c) for c in feat100]
    log(f"研究Top100可用: {len(feat100)}")

    for tag, X5s, X2s in [
            ("rad+aq", X5, X2),
            ("Top100", X5[:, idx100], X2[:, idx100])]:
        run_cv_foldwise(X5s, y5, f"{tag}|CV")
        external(X5s, y5, X2s, y2, f"{tag}")

    log(f"总耗时 {time.time()-t0:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
