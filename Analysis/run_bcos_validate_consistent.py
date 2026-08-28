#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_bcos_validate_consistent.py
================================
基于一致 ICD 标签的 2026-05 -> 2026-02 外部验证。

标签来源: labels_ae_bcos_2026_05.csv / labels_ae_bcos_2026_02.csv（由 build_bcos_labels_consistent.py 生成）
  AECOPD    : J44.1*|J44.0* -> 1, J44.9*|J44.8* -> 0 (仅 J44)
  COPD_BCOS : COPD合并支扩==1 -> 1, else 0 (仅 J44)

对每个任务：
  (a) 2026-05 内部 5 折 CV LR（rad | rad+aq | rad+aq+bcos 消融）
  (b) 外部验证：2026-05 全量训练 -> 2026-02 预测 AUC

join 键: Patient_ID（长串目录名）。
输出: E:\DICOM\2026-02-seg\validate_consistent.log + validate_consistent_results.csv
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

LOG = open(r"E:\DICOM\2026-02-seg\validate_consistent.log", "w", encoding="utf-8")
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

AQ_PREFIX = ("TD_", "blur_", "wall_", "WA_", "Din_", "Dout_", "mean_",
             "Pi10", "Vessel_", "Lobe_", "Lung_", "Airway_", "PA_",
             "Diaphragm_", "pca_", "RV_", "LV_", "CAC_")
RAD_EXTRA = ("Lobe_", "Lung_", "Airway_", "PA_", "Diaphragm_", "heart",
             "aorta", "trachea", "pulmonary_artery")
META = {"Patient_ID", "PatientID", "Patient_ID_long", "PatientID_raw", "CT_Series",
        "_nid", "AECOPD", "COPD_BCOS", "ICD", "main_diagnosis", "patient_id",
        "患者id", "主要诊断"}


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


def load(feat_path, lab_path, tag):
    df = pd.read_csv(feat_path)
    lab = pd.read_csv(lab_path)
    m = df.merge(lab[["Patient_ID", "AECOPD", "COPD_BCOS"]], on="Patient_ID", how="inner")
    m = m.drop_duplicates(subset=["Patient_ID"])
    log(f"2026-{tag}: 特征表 {len(df)} 行 -> join 标签后 {len(m)} 行")
    return m


def run_cv(X, y, n_splits=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aucs, accs, sens, specs = [], [], [], []
    scaler = StandardScaler()
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                 class_weight="balanced", max_iter=2000, random_state=SEED)
        clf.fit(scaler.fit_transform(X[tr]), y[tr])
        Xte = scaler.transform(X[te])
        p = clf.predict_proba(Xte)[:, 1]
        pred = clf.predict(Xte)
        aucs.append(roc_auc_score(y[te], p))
        accs.append(accuracy_score(y[te], pred))
        tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
        sens.append(tp / (tp + fn) if tp + fn else 0.0)
        specs.append(tn / (tn + fp) if tn + fp else 0.0)
    return (float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(accs)),
            float(np.mean(sens)), float(np.mean(specs)))


def external(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                             class_weight="balanced", max_iter=2000, random_state=SEED)
    clf.fit(scaler.fit_transform(Xtr), ytr)
    p = clf.predict_proba(scaler.transform(Xte))[:, 1]
    return float(roc_auc_score(yte, p))


def main():
    t0 = time.time()
    log("===== 一致标签下的 2026-05 -> 2026-02 外部验证 =====")
    m5 = load(F5, L5, "05")
    m2 = load(F2, L2, "02")

    shared = [c for c in m2.columns if c in m5.columns and c not in META]
    log(f"共享特征列: {len(shared)}")
    rad, aq = split_feats(shared)
    log(f"rad={len(rad)} aq={len(aq)}")

    results = []

    for task in ["AECOPD", "COPD_BCOS"]:
        log(f"\n########## 任务: {task} ##########")
        y5 = m5[task].values
        y2 = m2[task].values
        keep5 = ~np.isnan(y5)
        keep2 = ~np.isnan(y2)
        m5t = m5[keep5].reset_index(drop=True)
        m2t = m2[keep2].reset_index(drop=True)
        y5 = m5t[task].values.astype(int)
        y2 = m2t[task].values.astype(int)
        log(f"2026-05(训练): n={len(y5)} pos={int(y5.sum())} neg={int((y5==0).sum())} "
            f"阳性率 {y5.mean():.1%}")
        log(f"2026-02(测试): n={len(y2)} pos={int(y2.sum())} neg={int((y2==0).sum())} "
            f"阳性率 {y2.mean():.1%}")

        X5all = m5t[shared].apply(pd.to_numeric, errors="coerce")
        X2all = m2t[shared].apply(pd.to_numeric, errors="coerce")
        med5 = X5all.median().fillna(0); med2 = X2all.median().fillna(0)
        X5all = X5all.fillna(med5).fillna(0).values.astype(np.float64)
        X2all = X2all.fillna(med2).fillna(0).values.astype(np.float64)

        # bcos 特征（AECOPD 任务消融用）
        b5 = m5t["COPD_BCOS"].fillna(0).values.astype(np.float64).reshape(-1, 1)
        b2 = m2t["COPD_BCOS"].fillna(0).values.astype(np.float64).reshape(-1, 1)

        sets = [("rad", X5all[:, [shared.index(c) for c in rad]], X2all[:, [shared.index(c) for c in rad]]),
                ("rad+aq", X5all, X2all)]
        if task == "AECOPD":
            sets.append(("rad+aq+bcos",
                         np.hstack([X5all, b5]), np.hstack([X2all, b2])))

        for tag, X5, X2 in sets:
            auc, sd, acc, sens, spec = run_cv(X5, y5)
            ext = external(X5, y5, X2, y2)
            log(f"[{task}|{tag}] n_feat={X5.shape[1]} "
                f"2026-05 CV AUC={auc:.3f}±{sd:.3f} Acc={acc:.3f} Sens={sens:.3f} Spec={spec:.3f} "
                f"| 2026-02 外部 AUC={ext:.3f}")
            results.append({"task": task, "feats": tag, "n_feat": X5.shape[1],
                            "cv_auc": auc, "cv_auc_std": sd, "cv_acc": acc,
                            "cv_sens": sens, "cv_spec": spec, "ext_auc": ext})

    pd.DataFrame(results).to_csv(
        r"E:\DICOM\2026-02-seg\validate_consistent_results.csv",
        index=False, encoding="utf-8-sig")
    log(f"\n结果 -> validate_consistent_results.csv")
    log(f"总耗时 {time.time()-t0:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
