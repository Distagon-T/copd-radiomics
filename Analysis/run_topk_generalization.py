#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法: python run_topk_generalization.py   # TopK 特征筛选泛化实验
run_topk_generalization.py
===========================
尝试改善泛化：在 2026-05 训练集上做单变量特征筛选（TopK），再用 TopK 特征训练
LR 去外验 2026-01 / 2026-02。
任务: AECOPD / COPD_BCOS / J44.0-vs-J44.9（更均衡的候选任务）
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

LOG = open(r"E:\DICOM\2026-02-seg\topk_gen.log", "w", encoding="utf-8")
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


def load(tag):
    f, l = PATHS[tag]
    df = pd.read_csv(f)
    if l:
        lab = pd.read_csv(l)
        m = df.merge(lab[["Patient_ID", "ICD", "AECOPD", "COPD_BCOS"]], on="Patient_ID", how="inner")
    else:
        m = df
    return m.drop_duplicates(subset=["Patient_ID"])


def uni_auc(X, y):
    """向量化单变量 AUC（scipy rankdata，正确版）"""
    from scipy.stats import rankdata
    pos = y == 1
    np_ = int(pos.sum()); nn = int((~pos).sum())
    Z = np.vstack([X[pos], X[~pos]])
    R = rankdata(Z, axis=0)
    Rpos = R[:np_].sum(0)
    return (Rpos - np_ * (np_ + 1) / 2) / (np_ * nn)


def ext(Xtr, ytr, Xte, yte, tag):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                             class_weight="balanced", max_iter=2000, random_state=SEED)
    clf.fit(sc.fit_transform(Xtr), ytr)
    a = roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1])
    log(f"     2026-{tag} 外部 AUC = {a:.3f}")
    return a


def cv5(X, y):
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


def main():
    t0 = time.time()
    log("===== TopK 特征筛选 改善泛化 =====")
    m5 = load("05"); m2 = load("02"); m1 = load("01")
    shared = [c for c in m1.columns if c in m2.columns and c in m5.columns and c not in META]
    log(f"共享特征 {len(shared)}; 样本 05={len(m5)} 02={len(m2)} 01={len(m1)}")

    def build(m):
        X = m[shared].apply(pd.to_numeric, errors="coerce")
        med = X.median().fillna(0)
        return X.fillna(med).fillna(0).values.astype(np.float64)
    X5 = build(m5); X2 = build(m2); X1 = build(m1)

    # 校验向量化 AUC 与 sklearn 一致
    ychk = m5["AECOPD"].values
    kchk = ~np.isnan(ychk)
    a_vec = uni_auc(X5[kchk], ychk[kchk].astype(int))
    from sklearn.metrics import roc_auc_score
    a_skl = np.array([roc_auc_score(ychk[kchk].astype(int), X5[kchk][:, j]) for j in range(5)])
    log(f"向量化AUC vs sklearn 前5特征: {np.round(a_vec[:5],4)} vs {np.round(a_skl,4)}")

    def make_label(m, task):
        icd = m["ICD"].astype(str).str.strip()
        if task == "AECOPD":
            y = m["AECOPD"].values
        elif task == "COPD_BCOS":
            y = m["COPD_BCOS"].fillna(0).values
        elif task == "J44.0_vs_J44.9":
            y = np.where(icd.str.startswith("J44.0"), 1,
                         np.where(icd.str.startswith("J44.9"), 0, np.nan))
        else:
            raise ValueError(task)
        return y.astype(float)

    for task in ["AECOPD", "COPD_BCOS", "J44.0_vs_J44.9"]:
        y5 = make_label(m5, task); y2 = make_label(m2, task); y1 = make_label(m1, task)
        k5 = ~np.isnan(y5); k2 = ~np.isnan(y2); k1 = ~np.isnan(y1)
        y5 = y5[k5].astype(int); y2 = y2[k2].astype(int); y1 = y1[k1].astype(int)
        X5t = X5[k5]; X2t = X2[k2]; X1t = X1[k1]
        log(f"\n##### {task} #####")
        log(f"  05 n={len(y5)} pos={int(y5.sum())} ({y5.mean():.1%}) | "
            f"02 n={len(y2)} pos={int(y2.sum())} ({y2.mean():.1%}) | "
            f"01 n={len(y1)} pos={int(y1.sum())} ({y1.mean():.1%})")

        auc = uni_auc(X5t, y5)
        order = np.argsort(-np.abs(auc - 0.5))
        for K in [20, 50, 100, 200]:
            idx = order[:K]
            c, s = cv5(X5t[:, idx], y5)
            log(f"  [K={K}] 05 CV AUC={c:.3f}±{s:.3f}")
            ext(X5t[:, idx], y5, X2t[:, idx], y2, "02")
            ext(X5t[:, idx], y5, X1t[:, idx], y1, "01")

    log(f"总耗时 {time.time()-t0:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
