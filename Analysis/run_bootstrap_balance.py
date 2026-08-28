#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法: python run_bootstrap_balance.py   # bootstrap CI + 下采样平衡
run_bootstrap_balance.py
=========================
1) 为每个任务的外验 AUC 补 bootstrap 95% CI（对测试集重采样 500 次，模型固定）
2) 对不平衡任务做多数类下采样(balanced)训练，对比外验 AUC 与 CI
任务: AECOPD / COPD_BCOS / J44.0_vs_J44.9
输出: log + 森林图 figs/fig_bal_boot_ci.png
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

LOG = open(r"E:\DICOM\2026-02-seg\bootstrap_balance.log", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()

SEED = 42
SEG2 = r"E:\DICOM\2026-02-seg"
FIGD = os.path.join(SEG2, "figs")
os.makedirs(FIGD, exist_ok=True)
META = {"Patient_ID", "PatientID", "PatientID_raw", "Patient_ID_long", "CT_Series",
        "patient_id", "ICD", "main_diagnosis", "AECOPD", "COPD_BCOS", "患者id"}
PATHS = {
    "05": (r"E:\DICOM\2026-05-seg\2026-05-integrated_radiomics_aq.csv",
           r"E:\DICOM\2026-05-seg\labels_ae_bcos_2026_05.csv"),
    "02": (r"E:\DICOM\2026-02-seg\2026-02-integrated_radiomics_aq.csv",
           r"E:\DICOM\2026-02-seg\labels_ae_bcos_2026_02.csv"),
    "01": (r"E:\DICOM\2026-01-seg\2026-01-integrated_radiomics_aq.csv", None),
}
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


def fit_lr(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                             class_weight="balanced", max_iter=2000, random_state=SEED)
    clf.fit(sc.fit_transform(X), y)
    return sc, clf


def balanced_idx(y, rng):
    """多数类下采样到与少数类等量"""
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    k = min(len(pos), len(neg))
    return np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])


def boot_auc(sc, clf, Xte, yte, rng, B=B):
    from sklearn.metrics import roc_auc_score
    p_all = clf.predict_proba(sc.transform(Xte))[:, 1]
    pt = roc_auc_score(yte, p_all)
    n = len(yte)
    bs = np.array([roc_auc_score(yte[idx], p_all[idx])
                   for idx in [rng.choice(n, n, replace=True) for _ in range(B)]])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return pt, lo, hi


def main():
    t0 = time.time()
    log("===== Bootstrap CI + 下采样平衡 对比 =====")
    m5 = load("05"); m2 = load("02"); m1 = load("01")
    shared = [c for c in m1.columns if c in m2.columns and c in m5.columns and c not in META]
    log(f"共享特征 {len(shared)}")

    def build(m):
        X = m[shared].apply(pd.to_numeric, errors="coerce")
        med = X.median().fillna(0)
        return X.fillna(med).fillna(0).values.astype(np.float64)
    X5 = build(m5); X2 = build(m2); X1 = build(m1)

    results = []
    for task in ["AECOPD", "COPD_BCOS", "J44.0_vs_J44.9"]:
        y5 = make_y(m5, task); y2 = make_y(m2, task); y1 = make_y(m1, task)
        k5 = ~np.isnan(y5); k2 = ~np.isnan(y2); k1 = ~np.isnan(y1)
        y5 = y5[k5].astype(int); y2 = y2[k2].astype(int); y1 = y1[k1].astype(int)
        X5t = X5[k5]; X2t = X2[k2]; X1t = X1[k1]
        log(f"\n##### {task} (05 pos={int(y5.sum())}/{len(y5)}) #####")

        for mode in ["full", "balanced"]:
            rng = np.random.default_rng(SEED)
            if mode == "full":
                tr = np.arange(len(y5)); ytr = y5
                tag_mode = "全量"
            else:
                tr = balanced_idx(y5, rng); ytr = y5[tr]
                tag_mode = "下采样平衡"
            sc, clf = fit_lr(X5t[tr], ytr)
            line = f"[{task}|{mode}] 训练 n={len(tr)} (pos={int(ytr.sum())})"
            for name, Xte, yte in [("02", X2t, y2), ("01", X1t, y1)]:
                pt, lo, hi = boot_auc(sc, clf, Xte, yte, rng)
                line += f" | 外验{name}={pt:.3f} CI[{lo:.3f},{hi:.3f}]"
                results.append({"task": task, "mode": mode, "test": name,
                                "auc": pt, "ci_lo": lo, "ci_hi": hi,
                                "train_n": len(tr), "train_pos": int(ytr.sum())})
            log(line)

    pd.DataFrame(results).to_csv(os.path.join(SEG2, "bootstrap_balance_results.csv"),
                                 index=False, encoding="utf-8-sig")

    # 森林图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i, task in enumerate(["AECOPD", "COPD_BCOS", "J44.0_vs_J44.9"]):
        ax = axes[i]
        sub = df[df["task"] == task]
        ypos = []
        for i, (mode, test) in enumerate([("full", "02"), ("full", "01"),
                                          ("balanced", "02"), ("balanced", "01")]):
            r = sub[(sub["mode"] == mode) & (sub["test"] == test)].iloc[0]
            y = len(ypos)
            col = "#4c72b0" if test == "02" else "#dd8452"
            ax.errorbar([r["auc"]], [y], xerr=[[r["auc"] - r["ci_lo"]], [r["ci_hi"] - r["auc"]]],
                        fmt="o", color=col, ecolor=col, capsize=3, ms=6)
            ypos.append(f"{mode}/{test}")
        ax.axvline(0.5, color="k", ls="--", lw=0.8)
        ax.set_yticks(range(len(ypos))); ax.set_yticklabels(ypos, fontsize=8)
        ax.set_xlim(0.2, 0.9)
        ax.set_title(f"{task}")
        ax.set_xlabel("外验 AUC (95%CI)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGD, "fig_bal_boot_ci.png"), dpi=150); plt.close()
    log(f"\n森林图 -> figs/fig_bal_boot_ci.png ; 结果 -> bootstrap_balance_results.csv")
    log(f"总耗时 {time.time()-t0:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
