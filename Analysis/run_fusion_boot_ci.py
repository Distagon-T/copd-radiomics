#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法: python run_fusion_boot_ci.py   # 融合最优模型 bootstrap CI
run_fusion_boot_ci.py
======================
对 radTop100 + aqTop20 融合模型补 bootstrap 95%CI（外验 02/01，测试集重采样 500 次）
任务: AECOPD / COPD_BCOS / J44.0_vs_J44.9
输出: E:\DICOM\reports\fusion_boot_ci.csv + figs/fig_fusion_boot_ci.png
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

SEED = 42
REPORTS = r"E:\DICOM\reports"
FIGD = os.path.join(REPORTS, "figs")
os.makedirs(FIGD, exist_ok=True)
LOG = open(r"E:\DICOM\2026-02-seg\fusion_boot_ci.log", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()

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
RAD_EXTRA = ("Lobe_", "Lung_", "Airway_", "PA_", "Diaphragm_", "heart",
             "aorta", "trachea", "pulmonary_artery")
B = 500


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


def uni_auc(X, y):
    from scipy.stats import rankdata
    pos = y == 1
    np_ = int(pos.sum()); nn = int((~pos).sum())
    Z = np.vstack([X[pos], X[~pos]])
    R = rankdata(Z, axis=0)
    Rpos = R[:np_].sum(0)
    return (Rpos - np_ * (np_ + 1) / 2) / (np_ * nn)


def boot_auc(sc, clf, Xte, yte, rng, B=B):
    from sklearn.metrics import roc_auc_score
    p_all = clf.predict_proba(sc.transform(Xte))[:, 1]
    pt = roc_auc_score(yte, p_all)
    n = len(yte)
    bs = np.array([roc_auc_score(yte[idx], p_all[idx])
                   for idx in [rng.choice(n, n, replace=True) for _ in range(B)]])
    return pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fit_lr(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                             class_weight="balanced", max_iter=2000, random_state=SEED)
    clf.fit(sc.fit_transform(X), y)
    return sc, clf


def main():
    t0 = time.time()
    log("===== radTop100+aqTop20 融合模型 bootstrap CI =====")
    m5 = load("05"); m2 = load("02"); m1 = load("01")
    shared = [c for c in m1.columns if c in m2.columns and c in m5.columns and c not in META]
    rad, aq = split_feats(shared)

    def build(m):
        X = m[shared].apply(pd.to_numeric, errors="coerce")
        med = X.median().fillna(0)
        return X.fillna(med).fillna(0).values.astype(np.float64)
    X5 = build(m5); X2 = build(m2); X1 = build(m1)
    rad_idx = np.array([shared.index(c) for c in rad])
    aq_idx = np.array([shared.index(c) for c in aq])

    results = []
    for task in ["AECOPD", "COPD_BCOS", "J44.0_vs_J44.9"]:
        y5 = make_y(m5, task); y2 = make_y(m2, task); y1 = make_y(m1, task)
        k5 = ~np.isnan(y5); k2 = ~np.isnan(y2); k1 = ~np.isnan(y1)
        y5 = y5[k5].astype(int); y2 = y2[k2].astype(int); y1 = y1[k1].astype(int)
        X5r = X5[k5][:, rad_idx]; X2r = X2[k2][:, rad_idx]; X1r = X1[k1][:, rad_idx]
        X5q = X5[k5][:, aq_idx]; X2q = X2[k2][:, aq_idx]; X1q = X1[k1][:, aq_idx]
        aur = uni_auc(X5r, y5); auq = uni_auc(X5q, y5)
        o_r = np.argsort(-np.abs(aur - 0.5))[:100]
        o_q = np.argsort(-np.abs(auq - 0.5))[:20]
        X5c = np.hstack([X5r[:, o_r], X5q[:, o_q]])
        X2c = np.hstack([X2r[:, o_r], X2q[:, o_q]])
        X1c = np.hstack([X1r[:, o_r], X1q[:, o_q]])

        rng = np.random.default_rng(SEED)
        sc5, clf5 = fit_lr(X5c, y5)
        e2, lo2, hi2 = boot_auc(sc5, clf5, X2c, y2, rng)
        e1, lo1, hi1 = boot_auc(sc5, clf5, X1c, y1, rng)
        log(f"[{task}] radTop100+aqTop20 (n=120): "
            f"02={e2:.3f} CI[{lo2:.3f},{hi2:.3f}] | 01={e1:.3f} CI[{lo1:.3f},{hi1:.3f}]")
        results.append({"task": task, "ext_02": e2, "ci02_lo": lo2, "ci02_hi": hi2,
                        "ext_01": e1, "ci01_lo": lo1, "ci01_hi": hi1})

    pd.DataFrame(results).to_csv(os.path.join(REPORTS, "fusion_boot_ci.csv"),
                                 index=False, encoding="utf-8-sig")

    # 森林图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, col, lo_c, hi_c, tit in [
            (axes[0], "ext_02", "ci02_lo", "ci02_hi", "2026-02 外验"),
            (axes[1], "ext_01", "ci01_lo", "ci01_hi", "2026-01 外验")]:
        tasks = df["task"].tolist()[::-1]
        v = df[col].tolist()[::-1]
        lo = df[lo_c].tolist()[::-1]
        hi = df[hi_c].tolist()[::-1]
        y = np.arange(len(tasks))
        cols = ["#4c72b0", "#dd8452", "#55a868"][::-1]
        for i, t in enumerate(tasks):
            ax.errorbar([v[i]], [y[i]], xerr=[[v[i] - lo[i]], [hi[i] - v[i]]],
                        fmt="o", color=cols[i], ecolor=cols[i], capsize=4, ms=7)
        ax.axvline(0.5, color="k", ls="--", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(tasks, fontsize=9)
        ax.set_xlim(0.2, 0.85); ax.set_xlabel(f"{tit} AUC (95%CI)")
        ax.set_title(f"radTop100+aqTop20 - {tit}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGD, "fig_fusion_boot_ci.png"), dpi=150); plt.close()
    log(f"森林图 -> {FIGD}\\fig_fusion_boot_ci.png")
    log(f"总耗时 {time.time()-t0:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
