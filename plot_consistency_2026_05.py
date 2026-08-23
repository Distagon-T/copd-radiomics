#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_consistency_2026_05.py
===========================
一致性分析（bootstrap）：对显著特征的单变量判别能力（AUC）做重采样，
评估其稳定性，输出森林图（均值 ± 95%CI + 方向稳定性百分比）。

- radiomics 显著特征  -> fig_consistency_radiomics_2026_05.png
- AirQuant 特征(含 Pi10) -> fig_consistency_airquant_2026_05.png
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_fusion_model import load_and_join, select_features

SEG = r"E:\DICOM\2026-05-seg"
FIGDIR = os.path.join(SEG, "figs")
os.makedirs(FIGDIR, exist_ok=True)

RAD = os.path.join(SEG, "radiomics_2026_05_features.csv")
AQ = os.path.join(SEG, "airquant_2026_05_aggregated.csv")
LAB = os.path.join(SEG, "labels_2026_05.csv")

N_BOOT = 200
SEED = 42
TOP_N = 15


class Args:
    radiomics = RAD
    airquant = AQ
    labels = LAB


def bootstrap_auc(x, y, n_iter=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    aucs = []
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    xf = np.where(np.isnan(x), med, x)
    for _ in range(n_iter):
        sp = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        sn = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        idx = np.concatenate([sp, sn])
        if np.unique(y[idx]).size < 2:
            continue
        aucs.append(roc_auc_score(y[idx], xf[idx]))
    return np.array(aucs)


def forest_plot(items, title, out_path):
    """items: list of (feature, mean_auc, ci_lo, ci_hi, stability, direction)"""
    items = sorted(items, key=lambda t: -abs(t[1] - 0.5))
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(items) + 1.2))
    ypos = np.arange(len(items))[::-1]
    for i, (name, mu, lo, hi, stab, dirn) in enumerate(items):
        color = "#d62728" if dirn >= 0 else "#1f77b4"
        ax.errorbar([mu], [ypos[i]], xerr=[[mu - lo], [hi - mu]],
                    fmt="o", color=color, ecolor=color, capsize=3, ms=5)
        ax.text(0.5, ypos[i] + 0.22, f"{stab:.0%}", ha="center",
                fontsize=7, color="gray")
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([t[0] for t in items], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Bootstrap univariate AUC (mean ± 95% CI)")
    ax.set_title(title + f"  (bootstrap n={N_BOOT})", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"一致性图 -> {out_path}")


def main():
    df = load_and_join(Args())
    y = df["cvd_exacerbation_label"].values
    feats, dropped = select_features(df, "cvd_exacerbation_label")
    print(f"样本 {len(y)}, 阳性 {int(y.sum())}")

    def point_auc(feature_list):
        out = []
        for c in feature_list:
            x = pd.to_numeric(df[c], errors="coerce").values
            if np.isnan(x).mean() > 0.3:
                continue
            try:
                a = roc_auc_score(y, np.nan_to_num(x, nan=np.nanmedian(x)))
            except ValueError:
                continue
            out.append((c, a))
        return out

    def analyze(feature_list, label):
        # 先快速算点估计，只对 |AUC-0.5| 最大的 TOP_N 做 bootstrap
        pts = sorted(point_auc(feature_list), key=lambda t: -abs(t[1] - 0.5))
        out = []
        for c, point in pts[:TOP_N]:
            x = pd.to_numeric(df[c], errors="coerce").values
            b = bootstrap_auc(x, y)
            if len(b) < 50:
                continue
            lo, hi = np.percentile(b, [2.5, 97.5])
            stab = float(np.mean((b > 0.5) if point >= 0.5 else (b < 0.5)))
            out.append((c, point, lo, hi, stab, np.sign(point - 0.5)))
        return out

    # ---- radiomics 显著特征（非 AQ）----
    rad_feats = [c for c in feats if not c.startswith("aq_")]
    rad_top = analyze(rad_feats, "radiomics")
    forest_plot(rad_top,
                "2026-05 Consistency: significant radiomics features",
                os.path.join(FIGDIR, "fig_consistency_radiomics_2026_05.png"))
    print("--- radiomics 一致性(前%d) ---" % len(rad_top))
    for t in rad_top:
        print(f"  {t[0][:52]:54s} AUC={t[1]:.3f} CI[{t[2]:.3f},{t[3]:.3f}] "
              f"同向稳定={t[4]:.0%}")

    # ---- AirQuant 特征（含 aq_Pi10，强制加入）----
    aq_feats = [c for c in df.columns if c.startswith("aq_")]
    aq_top = analyze(aq_feats, "airquant")

    # 强制纳入用户点名的 Pi10 与分支数（即使判别力低也展示其一致性）
    force = ["aq_Pi10", "aq_n_branches"]
    for c in force:
        if c in aq_top:
            continue
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce").values
        try:
            point = roc_auc_score(y, np.nan_to_num(x, nan=np.nanmedian(x)))
        except ValueError:
            continue
        b = bootstrap_auc(x, y)
        if len(b) >= 50:
            lo, hi = np.percentile(b, [2.5, 97.5])
            stab = float(np.mean((b > 0.5) if point >= 0.5 else (b < 0.5)))
            aq_top.append((c, point, lo, hi, stab, np.sign(point - 0.5)))

    forest_plot(aq_top,
                "2026-05 Consistency: AirQuant features (incl. Pi10)",
                os.path.join(FIGDIR, "fig_consistency_airquant_2026_05.png"))
    print("\n--- AirQuant 一致性(%d) ---" % len(aq_top))
    for t in aq_top:
        print(f"  {t[0][:52]:54s} AUC={t[1]:.3f} CI[{t[2]:.3f},{t[3]:.3f}] "
              f"同向稳定={t[4]:.0%}")

    print(f"\n图目录: {FIGDIR}")


if __name__ == "__main__":
    main()
