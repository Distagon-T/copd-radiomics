#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_bcos_study_2026_05.py
==========================
慢阻肺合并支扩（COPD+BCOS）研究：
  训练集 = 2026-05（COPD_BCOS label, 534 例）
  外部验证 = 2026-02（医生标注 COPD合并支扩, 370 例）

输出:
  bcos_screening_univariate_top.csv    单变量筛选 Top（训练集）
  bcos_lr_coefficients.csv             LR 系数（训练集, 相关性非因果）
  bcos_cv_log.txt                      5折CV + 外部验证结果
  figs/fig_bcos_roc.png                训练CV平均ROC + 2026-02外部ROC
  figs/fig_bcos_univariate_auc.png     单变量AUC柱状图
  figs/fig_bcos_consistency.png        训练集 top 特征 bootstrap 一致性
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEG = r"E:\DICOM\2026-05-seg"
SEG2 = r"E:\DICOM\2026-02-seg"
FIGDIR = os.path.join(SEG, "figs")
os.makedirs(FIGDIR, exist_ok=True)

TRAIN_FEAT = os.path.join(SEG, "2026-05-integrated_radiomics_aq.csv")
TRAIN_LAB = os.path.join(SEG, "labels_bcos_2026_05.csv")
VAL_FEAT = os.path.join(SEG2, "2026-02-integrated_radiomics_aq.csv")
VAL_LAB = os.path.join(SEG2, "2026-2提取.xlsx")

ID_COLS = ["Patient_ID", "PatientID", "CT_Series"]


def norm(s):
    return (s.astype(str).str.replace(r"\.0$", "", regex=True)
            .str.strip().str.lstrip("0"))


def load_cohort(feat_path, lab_path, lab_col, feat_id, lab_id):
    feat = pd.read_csv(feat_path)
    if lab_path.endswith(".xlsx"):
        lab = pd.read_excel(lab_path)
    else:
        lab = pd.read_csv(lab_path)
    feat["_nid"] = norm(feat[feat_id])
    lab["_nid"] = norm(lab[lab_id])
    m = feat.merge(lab[[lab_id if lab_id != "_nid" else "_nid", lab_col, "_nid"]],
                   on="_nid", how="inner").drop_duplicates(subset=["_nid"])
    y = m[lab_col].astype(int).values
    return m, y


def main():
    # ---------- 数据 ----------
    tr, y_tr = load_cohort(TRAIN_FEAT, TRAIN_LAB, "COPD_BCOS", "PatientID", "patient_id")
    va, y_va = load_cohort(VAL_FEAT, VAL_LAB, "COPD合并支扩", "PatientID", "患者id")
    print(f"训练 2026-05: {len(tr)} 例, 阳性 {int(y_tr.sum())}")
    print(f"验证 2026-02: {len(va)} 例, 阳性 {int(y_va.sum())}")

    # ---------- 共同特征 ----------
    tr_feats = [c for c in tr.columns if c not in ID_COLS and "_nid" != c]
    va_feats = [c for c in va.columns if c not in ID_COLS and "_nid" != c]
    common = [c for c in tr_feats if c in set(va_feats)]
    print(f"共同特征: {len(common)}")

    # 数值化 + 过滤
    def prep(df, feats):
        X = df[feats].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median())
        return X

    X_tr = prep(tr, common)
    X_va = prep(va, common)
    # 剔除训练集高缺失/零方差
    keep = []
    for c in common:
        if X_tr[c].isna().mean() > 0.30:
            continue
        if X_tr[c].std() < 1e-9:
            continue
        keep.append(c)
    print(f"筛选后特征: {len(keep)}")
    X_tr = X_tr[keep]; X_va = X_va[keep]

    # ---------- 1) 单变量筛选（训练集）----------
    rows = []
    for c in keep:
        x = X_tr[c].values
        x0 = x[y_tr == 0]; x1 = x[y_tr == 1]
        try:
            auc = roc_auc_score(y_tr, x)
            u, p = sps.mannwhitneyu(x0, x1, alternative="two-sided")
        except ValueError:
            continue
        d = (x1.mean() - x0.mean()) / np.sqrt((x0.std()**2 + x1.std()**2) / 2 + 1e-9)
        rows.append({"feature": c, "auc": auc, "cohens_d": d, "p_mwu": p})
    uni = pd.DataFrame(rows).sort_values("auc", ascending=False, key=lambda s: (s - 0.5).abs())
    uni_out = os.path.join(SEG, "bcos_screening_univariate_top.csv")
    uni.to_csv(uni_out, index=False, encoding="utf-8-sig")
    print(f"\n=== 单变量 AUC Top 20 (训练 2026-05) ===")
    for _, r in uni.head(20).iterrows():
        print(f"  {r['feature'][:50]:52s} AUC={r['auc']:.3f} d={r['cohens_d']:+.2f} p={r['p_mwu']:.2g}")

    # ---------- 2) LR 5折 CV（训练集）----------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    scaler = StandardScaler()
    for tr_i, te_i in skf.split(X_tr, y_tr):
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                 class_weight="balanced", max_iter=2000, random_state=42)
        clf.fit(scaler.fit_transform(X_tr.iloc[tr_i]), y_tr[tr_i])
        p = clf.predict_proba(scaler.transform(X_tr.iloc[te_i]))[:, 1]
        aucs.append(roc_auc_score(y_tr[te_i], p))
    print(f"\n=== 2026-05 训练 5折CV AUC: {np.mean(aucs):.3f}±{np.std(aucs):.3f} ===")

    # ---------- 3) 外部验证 2026-02 ----------
    clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                             class_weight="balanced", max_iter=2000, random_state=42)
    clf.fit(scaler.fit_transform(X_tr.values), y_tr)
    p_va = clf.predict_proba(scaler.transform(X_va.values))[:, 1]
    auc_va = roc_auc_score(y_va, p_va)
    print(f"=== 外部验证 2026-02 AUC: {auc_va:.3f} (n={len(y_va)}, pos={int(y_va.sum())}) ===")

    # 系数
    coef = pd.DataFrame({"feature": keep, "coef": clf.coef_[0]})
    coef["abs"] = coef["coef"].abs()
    coef = coef.sort_values("abs", ascending=False)
    coef_out = os.path.join(SEG, "bcos_lr_coefficients.csv")
    coef.to_csv(coef_out, index=False, encoding="utf-8-sig")
    print("\n=== LR 系数 Top 15 (相关性非因果) ===")
    for _, r in coef.head(15).iterrows():
        print(f"  {r['feature'][:50]:52s} coef={r['coef']:+.3f}")

    # ---------- 4) 图：ROC（训练CV平均 + 外部）----------
    plt.figure(figsize=(6.5, 6.5))
    # 训练 CV ROC
    tprs, base = [], np.linspace(0, 1, 101)
    for tr_i, te_i in skf.split(X_tr, y_tr):
        clf2 = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                  class_weight="balanced", max_iter=2000, random_state=42)
        clf2.fit(scaler.fit_transform(X_tr.iloc[tr_i]), y_tr[tr_i])
        p = clf2.predict_proba(scaler.transform(X_tr.iloc[te_i]))[:, 1]
        fpr, tpr, _ = roc_curve(y_tr[te_i], p)
        tprs.append(np.interp(base, fpr, tpr)); tprs[-1][0] = 0
    mt = np.mean(tprs, axis=0); mt[-1] = 1
    plt.plot(base, mt, "b-", lw=2, label=f"Train CV (AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f})")
    # 外部 ROC
    fpr_v, tpr_v, _ = roc_curve(y_va, p_va)
    plt.plot(fpr_v, tpr_v, "r-", lw=2, label=f"External 2026-02 (AUC={auc_va:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(loc="lower right")
    plt.title("COPD+BCOS: Train 2026-05 CV vs External 2026-02")
    plt.tight_layout()
    roc_p = os.path.join(FIGDIR, "fig_bcos_roc.png")
    plt.savefig(roc_p, dpi=150); plt.close()
    print(f"\nROC 图 -> {roc_p}")

    # ---------- 5) 图：单变量 AUC Top20 ----------
    top = uni.head(20).iloc[::-1]
    plt.figure(figsize=(9, 7))
    colors = ["#d62728" if a >= 0.5 else "#1f77b4" for a in top["auc"]]
    plt.barh(range(len(top)), top["auc"], color=colors)
    plt.axvline(0.5, color="k", ls="--", lw=1)
    plt.yticks(range(len(top)), [f[:46] for f in top["feature"]], fontsize=8)
    plt.xlabel("Univariate AUC (train 2026-05)"); plt.xlim(0, 1)
    plt.title("COPD+BCOS feature screening top20 (red=pos dir)")
    plt.tight_layout()
    u_p = os.path.join(FIGDIR, "fig_bcos_univariate_auc.png")
    plt.savefig(u_p, dpi=150); plt.close()
    print(f"单变量AUC图 -> {u_p}")

    # ---------- 6) 一致性：训练集 top10 bootstrap ----------
    rng = np.random.default_rng(42)
    idx_pos = np.where(y_tr == 1)[0]; idx_neg = np.where(y_tr == 0)[0]
    cons = []
    for c in uni.head(10)["feature"]:
        x = X_tr[c].values
        bs = []
        for _ in range(200):
            sp = rng.choice(idx_pos, len(idx_pos), replace=True)
            sn = rng.choice(idx_neg, len(idx_neg), replace=True)
            ix = np.concatenate([sp, sn])
            bs.append(roc_auc_score(y_tr[ix], x[ix]))
        cons.append((c, np.mean(bs), *np.percentile(bs, [2.5, 97.5]),
                     np.mean(bs > 0.5) if uni.loc[uni['feature']==c,'auc'].values[0] >= 0.5 else np.mean(bs < 0.5)))
    plt.figure(figsize=(8, 4.5))
    cons = sorted(cons, key=lambda t: -abs(t[1] - 0.5))
    ys = np.arange(len(cons))[::-1]
    for i, (c, mu, lo, hi, stab) in enumerate(cons):
        col = "#d62728" if mu >= 0.5 else "#1f77b4"
        plt.errorbar([mu], [ys[i]], xerr=[[mu - lo], [hi - mu]], fmt="o",
                     color=col, ecolor=col, capsize=3, ms=5)
        plt.text(0.5, ys[i] + 0.2, f"{stab:.0%}", ha="center", fontsize=7, color="gray")
    plt.axvline(0.5, color="k", ls="--", lw=1)
    plt.yticks(ys, [t[0][:44] for t in cons], fontsize=8)
    plt.xlabel("Bootstrap univariate AUC (95%CI)"); plt.xlim(0, 1)
    plt.title("COPD+BCOS top10 consistency (bootstrap n=200)")
    plt.tight_layout()
    c_p = os.path.join(FIGDIR, "fig_bcos_consistency.png")
    plt.savefig(c_p, dpi=150); plt.close()
    print(f"一致性图 -> {c_p}")

    # ---------- 7) 日志 ----------
    with open(os.path.join(SEG, "bcos_cv_log.txt"), "w", encoding="utf-8") as f:
        f.write(f"训练 2026-05: {len(tr)} 例, 阳性 {int(y_tr.sum())}\n")
        f.write(f"验证 2026-02: {len(va)} 例, 阳性 {int(y_va.sum())}\n")
        f.write(f"共同特征 {len(common)}, 筛选后 {len(keep)}\n")
        f.write(f"训练5折CV AUC: {np.mean(aucs):.3f}±{np.std(aucs):.3f}\n")
        f.write(f"外部验证2026-02 AUC: {auc_va:.3f}\n")
    print("\n日志 ->", os.path.join(SEG, "bcos_cv_log.txt"))


if __name__ == "__main__":
    main()
