#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_vessel_boost_2026_05.py
===========================
验证新增 Vessel_* 特征是否提升各任务判别力：
  任务：支扩咯血 / 急性COPD加重病因 / BCOS表型
  对比：top8(rad+aq)  vs  top8(rad+aq+vessel)  vs  top8(rad+aq)+top4(vessel)
  指标：5 折 CV AUC + bootstrap 均值/95%CI/稳定性
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_fusion_model import select_features, univariate_summary

SEG = "seg_results"
RAD_CSV = os.path.join(SEG, "radiomics_2026_05_features_vessel.csv")
SEED = 42
N_BOOT = 150

TASKS = [
    ("labels_bronch_hemoptysis_2026_05.csv", "HEMO_Label", "支扩: 咯血 vs 无咯血"),
    ("labels_copd_ae_cause_2026_05.csv", "AE_CAUSE_Label", "急性COPD: 感染型 vs 非感染型"),
    ("labels_bcos_phenotype_2026_05.csv", "PHENO_Label", "BCOS 表型: BCOS vs PureCOPD"),
]
TOP_RADAQ = 8
TOP_VESSEL = 4


def model_auc_cv(Xv, y, seed=SEED):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs, scaler = [], StandardScaler()
    for tr, te in skf.split(Xv, y):
        Xtr = scaler.fit_transform(Xv[tr]); Xte = scaler.transform(Xv[te])
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                 class_weight="balanced", max_iter=1000, random_state=seed)
        clf.fit(Xtr, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(aucs)), aucs


def bootstrap_model_auc(Xv, y, n_iter=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]; idx_neg = np.where(y == 0)[0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out, scaler = [], StandardScaler()
    for _ in range(n_iter):
        sp = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        sn = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        idx = np.concatenate([sp, sn])
        if np.unique(y[idx]).size < 2:
            continue
        fa = []
        for tr, te in skf.split(Xv[idx], y[idx]):
            Xtr = scaler.fit_transform(Xv[idx][tr]); Xte = scaler.transform(Xv[idx][te])
            clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                     class_weight="balanced", max_iter=1000, random_state=seed)
            clf.fit(Xtr, y[idx][tr])
            try:
                fa.append(roc_auc_score(y[idx][te], clf.predict_proba(Xte)[:, 1]))
            except ValueError:
                continue
        if fa:
            out.append(np.mean(fa))
    return np.array(out)


def prep(df, fs):
    X = df[fs].apply(pd.to_numeric, errors="coerce").fillna(
        df[fs].apply(pd.to_numeric, errors="coerce").median())
    return X.values.astype(np.float64)


def main():
    rad = pd.read_csv(RAD_CSV)
    rad["Patient_ID"] = rad["Patient_ID"].astype(str)
    print(f"vessel CSV: {len(rad)} 行, Vessel_* 列数 "
          f"{sum(c.startswith('Vessel_') for c in rad.columns)}")

    all_rows = []
    for fname, label_col, task in TASKS:
        lab = pd.read_csv(os.path.join(SEG, fname))
        lab["Patient_ID"] = lab["Patient_ID"].astype(str)
        lab = lab[lab[label_col].notna()].copy()
        lab[label_col] = lab[label_col].astype(int)
        df = rad.merge(lab[["Patient_ID", label_col]], on="Patient_ID", how="inner")
        y = df[label_col].values
        if len(y) < 30 or int(y.sum()) < 10:
            print(f"\n[{task}] 样本不足，跳过"); continue
        print(f"\n[{task}] n={len(y)} pos={int(y.sum())}")

        feats_all, _ = select_features(df, label_col)
        feats_radaq = [c for c in feats_all if not c.startswith("Vessel_")]
        feats_vessel = [c for c in feats_all if c.startswith("Vessel_")]
        uni = univariate_summary(df, feats_all, y, top=100)
        uni["auc_dev"] = (uni["auc_univ"] - 0.5).abs()

        top_radaq = uni[uni["feature"].isin(feats_radaq)].sort_values(
            "auc_dev", ascending=False).head(TOP_RADAQ)["feature"].tolist()
        top_all = uni.sort_values("auc_dev", ascending=False).head(TOP_RADAQ)["feature"].tolist()
        top_vessel = uni[uni["feature"].isin(feats_vessel)].sort_values(
            "auc_dev", ascending=False).head(TOP_VESSEL)["feature"].tolist()
        top_radaq_vessel = list(dict.fromkeys(top_radaq + top_vessel))

        print("  top_radaq:", [c[:40] for c in top_radaq])
        print("  top_vessel:", top_vessel)
        print("  top_all 前3:", [c[:40] for c in top_all[:3]])

        for tag, fs in [("radaq_top8", top_radaq),
                        ("radaq_top8+vessel", top_radaq_vessel),
                        ("all_top8(含vessel)", top_all)]:
            Xv = prep(df, fs)
            pt, folds = model_auc_cv(Xv, y)
            b = bootstrap_model_auc(Xv, y)
            lo, hi = np.percentile(b, [2.5, 97.5])
            stab = float(np.mean(b > 0.5))
            all_rows.append({"task": task, "model": tag, "n_feat": len(fs),
                             "cv_auc": round(pt, 3), "boot_mean": round(float(np.mean(b)), 3),
                             "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3),
                             "stability": round(stab, 2)})
            print(f"  [{tag}] n_feat={len(fs)} CV_AUC={pt:.3f} "
                  f"boot={np.mean(b):.3f} (95%CI {lo:.3f}-{hi:.3f}) stab={stab:.0%}")

    res = pd.DataFrame(all_rows)
    print("\n=== 汇总 ===")
    print(res.to_string(index=False))
    res.to_csv(os.path.join(SEG, "vessel_boost_comparison.csv"),
               index=False, encoding="utf-8-sig")
    print(f"已保存 -> {os.path.join(SEG, 'vessel_boost_comparison.csv')}")


if __name__ == "__main__":
    main()
