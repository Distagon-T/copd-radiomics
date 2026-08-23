#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_2026_05_results.py
=======================
1) 把 airquant 聚合特征 CSV 输出到 AirQuant 文件夹
2) 复现融合 LR（radiomics + AirQuant）并画图:
   - 5 折 CV 平均 ROC 曲线 (fig_roc_2026_05.png)
   - 单变量 AUC 柱状图 Top20 (fig_univariate_auc_2026_05.png)
   - Top8 特征按类别箱线图 (fig_boxplot_top8_2026_05.png)
   - AirQuant 特征单独的单变量 AUC 柱状图 (fig_univariate_auc_airquant_2026_05.png)
"""
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_fusion_model import load_and_join, select_features, norm_id

SEG = r"E:\DICOM\2026-05-seg"
AQ_DIR = r"E:\DICOM\2026-05-Airway_metrics_tmp"
FIGDIR = os.path.join(SEG, "figs")
os.makedirs(FIGDIR, exist_ok=True)

RAD = os.path.join(SEG, "radiomics_2026_05_features.csv")
AQ = os.path.join(SEG, "airquant_2026_05_aggregated.csv")
LAB = os.path.join(SEG, "labels_2026_05.csv")

FEAT_COLS = ["Patient_ID", "PatientID", "CT_Series"]


def shorten(name, n=44):
    return name if len(name) <= n else name[: n - 3] + "..."


class Args:
    radiomics = RAD
    airquant = AQ
    labels = LAB


def main():
    # ---------- 1) 输出 airquant 特征 CSV 到 AirQuant 文件夹 ----------
    aq_out = os.path.join(AQ_DIR, "airquant_2026_05_features.csv")
    shutil.copyfile(AQ, aq_out)
    print(f"airquant 特征 CSV -> {aq_out}")
    # radiomics 特征 CSV 已在 seg 文件夹
    print(f"radiomics 特征 CSV   -> {RAD}")

    # ---------- 2) 构建数据 ----------
    df = load_and_join(Args())
    y = df["cvd_exacerbation_label"].values
    feats, dropped = select_features(df, "cvd_exacerbation_label")
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(
        df[feats].apply(pd.to_numeric, errors="coerce").median())
    Xv = X.values.astype(np.float64)
    print(f"样本 {len(y)}, 阳性 {int(y.sum())}, 特征 {len(feats)}")
    aq_feats = [c for c in feats if c.startswith("aq_")]

    # ---------- 3) 5 折 CV + 平均 ROC ----------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tprs, aucs, base_fpr = [], [], np.linspace(0, 1, 101)
    plt.figure(figsize=(6.5, 6.5))
    scaler = StandardScaler()
    for tr, te in skf.split(Xv, y):
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                 class_weight="balanced", max_iter=1000, random_state=42)
        clf.fit(scaler.fit_transform(Xv[tr]), y[tr])
        proba = clf.predict_proba(scaler.transform(Xv[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], proba))
        fpr, tpr, _ = roc_curve(y[te], proba)
        tprs.append(np.interp(base_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    plt.plot(base_fpr, mean_tpr, "b-", lw=2,
             label=f"Mean ROC (AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f})")
    plt.fill_between(base_fpr, np.clip(mean_tpr - std_tpr, 0, 1),
                     np.clip(mean_tpr + std_tpr, 0, 1), color="b", alpha=0.15,
                     label="±1 std")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"2026-05 Acute Exacerbation (n={len(y)}, pos={int(y.sum())})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_p = os.path.join(FIGDIR, "fig_roc_2026_05.png")
    plt.savefig(roc_p, dpi=150); plt.close()
    print(f"ROC 图 -> {roc_p}")

    # ---------- 4) 单变量 AUC 柱状图 ----------
    uni = []
    for c in feats:
        x = pd.to_numeric(df[c], errors="coerce").values
        if np.isnan(x).mean() > 0.3:
            continue
        try:
            a = roc_auc_score(y, np.nan_to_num(x, nan=np.nanmedian(x)))
        except ValueError:
            continue
        uni.append((c, a))
    uni_df = pd.DataFrame(uni, columns=["feature", "auc"])
    uni_df["auc_dev"] = (uni_df["auc"] - 0.5).abs()
    top = uni_df.sort_values("auc_dev", ascending=False).head(20)

    plt.figure(figsize=(9, 7))
    colors = ["#d62728" if a >= 0.5 else "#1f77b4" for a in top["auc"]]
    plt.barh(range(len(top)), top["auc"], color=colors)
    plt.axvline(0.5, color="k", ls="--", lw=1)
    plt.yticks(range(len(top)), [shorten(f, 46) for f in top["feature"]], fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel("Univariate AUC"); plt.xlim(0, 1)
    plt.title("2026-05 Top 20 features by |AUC-0.5| (red=pos dir, blue=neg dir)")
    plt.tight_layout()
    u_p = os.path.join(FIGDIR, "fig_univariate_auc_2026_05.png")
    plt.savefig(u_p, dpi=150); plt.close()
    print(f"单变量AUC图 -> {u_p}")

    # ---------- 5) AirQuant 单变量 AUC ----------
    aq_top = uni_df[uni_df["feature"].str.startswith("aq_")].sort_values(
        "auc_dev", ascending=False).head(15)
    if len(aq_top):
        plt.figure(figsize=(8, 6))
        colors = ["#d62728" if a >= 0.5 else "#1f77b4" for a in aq_top["auc"]]
        plt.barh(range(len(aq_top)), aq_top["auc"], color=colors)
        plt.axvline(0.5, color="k", ls="--", lw=1)
        plt.yticks(range(len(aq_top)), [shorten(f, 46) for f in aq_top["feature"]], fontsize=8)
        plt.gca().invert_yaxis()
        plt.xlabel("Univariate AUC"); plt.xlim(0, 1)
        plt.title("2026-05 AirQuant features |AUC-0.5| Top 15")
        plt.tight_layout()
        aq_p = os.path.join(FIGDIR, "fig_univariate_auc_airquant_2026_05.png")
        plt.savefig(aq_p, dpi=150); plt.close()
        print(f"AirQuant单变量AUC图 -> {aq_p}")

    # ---------- 6) Top8 特征箱线图（按类别） ----------
    top8 = uni_df.sort_values("auc_dev", ascending=False).head(8)["feature"].tolist()
    ncol = 4
    nrow = int(np.ceil(len(top8) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, top8):
        x = pd.to_numeric(df[c], errors="coerce")
        d0 = x[y == 0].dropna(); d1 = x[y == 1].dropna()
        ax.boxplot([d0, d1], labels=["Neg", "Pos"], widths=0.6)
        try:
            a = roc_auc_score(y, np.nan_to_num(x.values, nan=np.nanmedian(x.values)))
            ax.set_title(f"AUC={a:.3f}\n{shorten(c, 30)}", fontsize=8)
        except ValueError:
            ax.set_title(shorten(c, 30), fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(top8):]:
        ax.axis("off")
    fig.suptitle("2026-05 Top 8 features by class (Neg vs Pos)", fontsize=12)
    plt.tight_layout()
    bx_p = os.path.join(FIGDIR, "fig_boxplot_top8_2026_05.png")
    plt.savefig(bx_p, dpi=150); plt.close()
    print(f"箱线图 -> {bx_p}")

    print(f"\n全部图已保存到: {FIGDIR}")


if __name__ == "__main__":
    main()
