#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_bcos_study_2026_05.py (v2)
==========================
慢阻肺合并支扩（COPD+BCOS）研究：
  训练集 = 2026-05（COPD_BCOS label）
  外部验证 = 2026-02（医生标注 COPD合并支扩）
流程: 单变量筛选 -> TopK 建模(LR 5折CV) -> 2026-02 外部验证 -> 一致性
"""
import os
import sys
import time

import numpy as np
import pandas as pd

from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

LOG = open(r"E:\DICOM\2026-05-seg\bcos_run.log", "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def norm(s):
    return (s.astype(str).str.replace(r"\.0$", "", regex=True)
            .str.strip().str.lstrip("0"))


def load_cohort(feat_path, lab_path, lab_col, feat_id, lab_id):
    t0 = time.time()
    feat = pd.read_csv(feat_path)
    lab = pd.read_excel(lab_path) if lab_path.endswith(".xlsx") else pd.read_csv(lab_path)
    feat["_nid"] = norm(feat[feat_id])
    lab["_nid"] = norm(lab[lab_id])
    m = feat.merge(lab[["_nid", lab_col]], on="_nid", how="inner").drop_duplicates(subset=["_nid"])
    y = m[lab_col].astype(int).values
    log(f"  加载 {os.path.basename(feat_path)}: {len(m)} 例, 阳性 {int(y.sum())} ({time.time()-t0:.0f}s)")
    return m, y


def main():
    SEG = r"E:\DICOM\2026-05-seg"
    SEG2 = r"E:\DICOM\2026-02-seg"
    TOPK = 100

    log("===== COPD+BCOS 研究 (训练 2026-05, 外部验证 2026-02) =====")
    tr, y_tr = load_cohort(os.path.join(SEG, "2026-05-integrated_radiomics_aq.csv"),
                           os.path.join(SEG, "labels_bcos_2026_05.csv"),
                           "COPD_BCOS", "PatientID", "patient_id")
    va, y_va = load_cohort(os.path.join(SEG2, "2026-02-integrated_radiomics_aq.csv"),
                           os.path.join(SEG2, "2026-2提取.xlsx"),
                           "COPD合并支扩", "PatientID", "患者id")

    ID = ["Patient_ID", "PatientID", "CT_Series", "_nid"]
    common = [c for c in tr.columns if c not in ID and c in set(va.columns)]
    log(f"共同特征: {len(common)}")

    def prep(df):
        X = df[common].apply(pd.to_numeric, errors="coerce")
        return X.fillna(X.median())

    X_tr = prep(tr)
    X_va = prep(va)
    keep = [c for c in common
            if X_tr[c].isna().mean() <= 0.30 and X_tr[c].std() > 1e-9]
    log(f"过滤后特征: {len(keep)}")
    X_tr = X_tr[keep]
    X_va = X_va[keep]

    # ---------- 1) 单变量筛选 ----------
    t0 = time.time()
    rows = []
    for c in keep:
        x = X_tr[c].values
        try:
            auc = roc_auc_score(y_tr, x)
            u, p = sps.mannwhitneyu(x[y_tr == 0], x[y_tr == 1], alternative="two-sided")
        except ValueError:
            continue
        d = (x[y_tr == 1].mean() - x[y_tr == 0].mean()) / np.sqrt(
            (x[y_tr == 0].std()**2 + x[y_tr == 1].std()**2) / 2 + 1e-9)
        rows.append({"feature": c, "auc": auc, "cohens_d": d, "p_mwu": p})
    uni = pd.DataFrame(rows).sort_values("auc", ascending=False,
                                         key=lambda s: (s - 0.5).abs())
    uni.to_csv(os.path.join(SEG, "bcos_screening_univariate_top.csv"),
               index=False, encoding="utf-8-sig")
    log(f"单变量筛选完成 ({len(uni)} 特征, {time.time()-t0:.0f}s)")
    log("\n=== 单变量 AUC Top 20 (训练 2026-05) ===")
    for _, r in uni.head(20).iterrows():
        log(f"  {r['feature'][:48]:50s} AUC={r['auc']:.3f} d={r['cohens_d']:+.2f} p={r['p_mwu']:.2g}")

    # ---------- 2) TopK 建模 + LR 5折CV ----------
    topk = uni.head(TOPK)["feature"].tolist()
    Xk = X_tr[topk].values.astype(np.float64)
    Xk_va = X_va[topk].values.astype(np.float64)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    aucs = []
    for tr_i, te_i in skf.split(Xk, y_tr):
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                 class_weight="balanced", max_iter=2000, random_state=42)
        clf.fit(scaler.fit_transform(Xk[tr_i]), y_tr[tr_i])
        p = clf.predict_proba(scaler.transform(Xk[te_i]))[:, 1]
        aucs.append(roc_auc_score(y_tr[te_i], p))
    log(f"\n=== 2026-05 训练 Top{TOPK} 5折CV AUC: {np.mean(aucs):.3f}±{np.std(aucs):.3f} ===")

    # ---------- 3) 外部验证 2026-02 ----------
    clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                             class_weight="balanced", max_iter=2000, random_state=42)
    clf.fit(scaler.fit_transform(Xk), y_tr)
    p_va = clf.predict_proba(scaler.transform(Xk_va))[:, 1]
    auc_va = roc_auc_score(y_va, p_va)
    log(f"=== 外部验证 2026-02 AUC: {auc_va:.3f} (n={len(y_va)}, pos={int(y_va.sum())}) ===")

    coef = pd.DataFrame({"feature": topk, "coef": clf.coef_[0]})
    coef["abs"] = coef["coef"].abs()
    coef = coef.sort_values("abs", ascending=False)
    coef.to_csv(os.path.join(SEG, "bcos_lr_coefficients.csv"),
                index=False, encoding="utf-8-sig")
    log("\n=== LR 系数 Top 15 (相关性非因果) ===")
    for _, r in coef.head(15).iterrows():
        log(f"  {r['feature'][:48]:50s} coef={r['coef']:+.3f}")

    # ---------- 4) 一致性 (top10 bootstrap) ----------
    rng = np.random.default_rng(42)
    ip = np.where(y_tr == 1)[0]
    ig = np.where(y_tr == 0)[0]
    cons = []
    for c in uni.head(10)["feature"]:
        x = X_tr[c].values
        bs = []
        for _ in range(200):
            si = np.concatenate([rng.choice(ip, len(ip), True), rng.choice(ig, len(ig), True)])
            bs.append(roc_auc_score(y_tr[si], x[si]))
        pt = uni.loc[uni["feature"] == c, "auc"].values[0]
        cons.append((c, np.mean(bs), *np.percentile(bs, [2.5, 97.5]),
                     np.mean(bs > 0.5) if pt >= 0.5 else np.mean(bs < 0.5)))
    log("\n=== top10 bootstrap 一致性 ===")
    for c, mu, lo, hi, stab in cons:
        log(f"  {c[:48]:50s} AUC={mu:.3f} CI[{lo:.3f},{hi:.3f}] 稳定={stab:.0%}")

    # ---------- 5) 图 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    figdir = os.path.join(SEG, "figs")
    os.makedirs(figdir, exist_ok=True)

    plt.figure(figsize=(6.5, 6.5))
    tprs, base = [], np.linspace(0, 1, 101)
    for tr_i, te_i in skf.split(Xk, y_tr):
        clf2 = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                  class_weight="balanced", max_iter=2000, random_state=42)
        clf2.fit(scaler.fit_transform(Xk[tr_i]), y_tr[tr_i])
        p = clf2.predict_proba(scaler.transform(Xk[te_i]))[:, 1]
        fpr, tpr, _ = roc_curve(y_tr[te_i], p)
        tprs.append(np.interp(base, fpr, tpr))
        tprs[-1][0] = 0
    mt = np.mean(tprs, axis=0)
    mt[-1] = 1
    plt.plot(base, mt, "b-", lw=2, label=f"Train CV (AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f})")
    fpr_v, tpr_v, _ = roc_curve(y_va, p_va)
    plt.plot(fpr_v, tpr_v, "r-", lw=2, label=f"External 2026-02 (AUC={auc_va:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.title(f"COPD+BCOS (Top{TOPK}): Train 2026-05 CV vs External 2026-02")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "fig_bcos_roc.png"), dpi=150)
    plt.close()

    top = uni.head(20).iloc[::-1]
    plt.figure(figsize=(9, 7))
    colors = ["#d62728" if a >= 0.5 else "#1f77b4" for a in top["auc"]]
    plt.barh(range(len(top)), top["auc"], color=colors)
    plt.axvline(0.5, color="k", ls="--", lw=1)
    plt.yticks(range(len(top)), [f[:46] for f in top["feature"]], fontsize=8)
    plt.xlabel("Univariate AUC (train 2026-05)")
    plt.xlim(0, 1)
    plt.title("COPD+BCOS feature screening top20")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "fig_bcos_univariate_auc.png"), dpi=150)
    plt.close()

    cons = sorted(cons, key=lambda t: -abs(t[1] - 0.5))
    ys = np.arange(len(cons))[::-1]
    plt.figure(figsize=(8, 4.5))
    for i, (c, mu, lo, hi, stab) in enumerate(cons):
        col = "#d62728" if mu >= 0.5 else "#1f77b4"
        plt.errorbar([mu], [ys[i]], xerr=[[mu - lo], [hi - mu]], fmt="o",
                     color=col, ecolor=col, capsize=3, ms=5)
        plt.text(0.5, ys[i] + 0.2, f"{stab:.0%}", ha="center", fontsize=7, color="gray")
    plt.axvline(0.5, color="k", ls="--", lw=1)
    plt.yticks(ys, [t[0][:44] for t in cons], fontsize=8)
    plt.xlabel("Bootstrap univariate AUC (95%CI)")
    plt.xlim(0, 1)
    plt.title("COPD+BCOS top10 consistency (bootstrap n=200)")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "fig_bcos_consistency.png"), dpi=150)
    plt.close()
    log("\n图已保存到 figs/ (fig_bcos_roc.png, fig_bcos_univariate_auc.png, fig_bcos_consistency.png)")
    log("完成")


if __name__ == "__main__":
    main()
