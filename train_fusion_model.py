#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_fusion_model.py
=====================
融合 pyRadiomics 特征 + AirQuant 聚合特征 + 规则 Label，
跑逻辑回归分类（分层 5 折 CV），输出:
  - CV AUC / Acc / Sens / Spec
  - 单变量描述统计（阳性 vs 阴性，按效应量排序）
  - LR 系数 Top 特征（相关性解释，非因果）

用法:
  python train_fusion_model.py \
      --radiomics radiomics_all_patients.csv \
      --airquant  airquant_patient_aggregated.csv \
      --labels    patient_cvd_labels.csv \
      --outdir    <output_dir>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def norm_id(s):
    """统一 ID: 去 .0 后缀、去前导 0"""
    return (s.astype(str)
             .str.replace(r"\.0$", "", regex=True)
             .str.strip()
             .str.lstrip("0")
             .replace("", np.nan))


def load_and_join(args):
    radi = pd.read_csv(args.radiomics)
    lab = pd.read_csv(args.labels)

    # 建 ID 映射: radiomics PatientID <-> Patient_ID(文件夹名)
    # labels 的 patient_id 是 xlsx 患者id，用 PatientID 关联
    radi["PatientID"] = norm_id(radi["PatientID"])
    if "patient_id" in lab.columns:
        lab["patient_id"] = norm_id(lab["patient_id"])
    elif "PatientID" in lab.columns:
        lab["patient_id"] = norm_id(lab["PatientID"])
    radi = radi.drop_duplicates(subset=["PatientID"], keep="first")

    # AirQuant 可选
    if args.airquant:
        aq = pd.read_csv(args.airquant)
        aq = aq.rename(columns={"patient": "Patient_ID"})
    else:
        aq = None

    # labels 关联 radiomics 的 PatientID（丢弃 lab 里的 Patient_ID，避免列冲突）
    if "Patient_ID" in lab.columns:
        lab = lab.drop(columns=["Patient_ID"])
    df = radi.merge(lab.rename(columns={"patient_id": "PatientID"}),
                    on="PatientID", how="inner")
    # AirQuant 用 Patient_ID 关联
    if aq is not None:
        df = df.merge(aq, on="Patient_ID", how="left")
        print(f"AirQuant 合并成功: {df['aq_n_branches'].notna().sum()}/{len(df)}")
    print(f"radiomics: {len(radi)} | 合并 labels 后: {len(df)}")
    return df


def select_features(df, label_col, max_missing=0.30, min_var=1e-6):
    """挑数值特征列：排除 ID/label/规则列；剔除高缺失、零方差"""
    excl_prefix = ("patient", "label", "rule", "hit", "CT_", "Series",
                   "PatientID", "Patient_ID", "tni_", "bnp_", "age", "aq_Pi10")
    excl_exact = {str(label_col).lower()}   # 大小写不敏感，避免 'NSFC_AE_Label' 泄漏
    feats, dropped = [], []
    for c in df.columns:
        low = str(c).lower()
        if low in excl_exact or low.startswith(excl_prefix) or low in ("patient",):
            continue
        if df[c].dtype not in (np.float64, np.int64, float, int):
            continue
        miss = df[c].isna().mean()
        if miss > max_missing:
            dropped.append((c, f"missing={miss:.0%}"))
            continue
        if df[c].std() < min_var:
            dropped.append((c, "const"))
            continue
        feats.append(c)
    return feats, dropped


def univariate_summary(df, feats, y, top=20):
    """阳性 vs 阴性 描述统计 + 单变量 AUC + Cohen's d"""
    from scipy import stats as sps
    rows = []
    for c in feats:
        x = pd.to_numeric(df[c], errors="coerce").values
        x0 = x[y == 0]
        x1 = x[y == 1]
        if len(x0) < 3 or len(x1) < 3:
            continue
        m0, m1 = np.nanmean(x0), np.nanmean(x1)
        sd = np.sqrt(((np.nanstd(x0) ** 2 + np.nanstd(x1) ** 2) / 2))
        d = (m1 - m0) / sd if sd > 0 else 0.0
        try:
            u, p = sps.mannwhitneyu(x0[~np.isnan(x0)], x1[~np.isnan(x1)],
                                    alternative="two-sided")
        except ValueError:
            continue
        try:
            auc = roc_auc_score(y, np.nan_to_num(x, nan=np.nanmedian(x)))
        except ValueError:
            auc = np.nan
        rows.append({"feature": c, "mean_neg": m0, "mean_pos": m1,
                     "cohens_d": d, "auc_univ": auc, "p_mwu": p})
    return pd.DataFrame(rows).sort_values("auc_univ", ascending=False, key=abs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radiomics", required=True)
    ap.add_argument("--airquant", default=None,
                    help="AirQuant 聚合 CSV（可选；不提供则纯 radiomics）")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--label-col", default="cvd_exacerbation_label",
                    help="labels CSV 中用作分类目标的列")
    ap.add_argument("--tag", default=None,
                    help="输出文件前缀标签（如 acute/comorbidity），默认取 label-col")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = Path(args.outdir) if args.outdir else Path(args.labels).parent
    if not HAS_SKLEARN:
        sys.exit("sklearn 未安装，请先: pip install scikit-learn")
    tag = args.tag or args.label_col

    df = load_and_join(args)
    y = df[args.label_col].values
    n_pos = int(y.sum())
    print(f"Label 列: {args.label_col} | 分布: 阳性 {n_pos} / {len(y)}")
    if n_pos < 5:
        print("警告: 阳性数 < 5，CV 结果不可靠，仅输出描述统计")
    if n_pos < 2:
        sys.exit("阳性数不足，无法训练")

    feats, dropped = select_features(df, args.label_col)
    print(f"入选特征: {len(feats)} (剔除 {len(dropped)})")
    if len(dropped) > 0:
        print("  剔除示例:", dropped[:5])

    X = df[feats].apply(pd.to_numeric, errors="coerce")
    # 中位数填充
    X = X.fillna(X.median())
    Xv = X.values.astype(np.float64)

    # ---------- 1) 单变量描述统计 ----------
    uni = univariate_summary(df, feats, y, top=20)
    uni_out = outdir / f"fusion_{tag}_univariate_top.csv"
    uni.head(30).to_csv(uni_out, index=False, encoding="utf-8-sig")
    print("\n=== 单变量 AUC Top 15 (|AUC-0.5| 排序) ===")
    uni["auc_dev"] = (uni["auc_univ"] - 0.5).abs()
    top15 = uni.sort_values("auc_dev", ascending=False).head(15)
    for _, r in top15.iterrows():
        print(f"  {r['feature'][:48]:50s} AUC={r['auc_univ']:.3f} "
              f"d={r['cohens_d']:+.2f} p={r['p_mwu']:.3g}")

    # ---------- 2) 逻辑回归 CV ----------
    print("\n=== LogisticRegression 分层 5 折 CV ===")
    n_splits = 5 if n_pos >= 5 else n_pos
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, accs, sens, specs, coefs = [], [], [], [], []
    scaler = StandardScaler()
    for fold, (tr, te) in enumerate(skf.split(Xv, y)):
        Xtr = scaler.fit_transform(Xv[tr])
        Xte = scaler.transform(Xv[te])
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                 class_weight="balanced", max_iter=1000,
                                 random_state=42)
        clf.fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)[:, 1]
        pred = clf.predict(Xte)
        aucs.append(roc_auc_score(y[te], proba))
        accs.append(accuracy_score(y[te], pred))
        tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
        sens.append(tp / (tp + fn) if (tp + fn) else 0.0)
        specs.append(tn / (tn + fp) if (tn + fp) else 0.0)
        coefs.append(clf.coef_[0])
        print(f"  fold{fold}: AUC={aucs[-1]:.3f} Acc={accs[-1]:.3f} "
              f"Sens={sens[-1]:.3f} Spec={specs[-1]:.3f} "
              f"(train {int(y[tr].sum())} pos / test {int(y[te].sum())} pos)")
    print(f"  平均: AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f} "
          f"Acc={np.mean(accs):.3f} Sens={np.mean(sens):.3f} Spec={np.mean(specs):.3f}")

    # ---------- 3) 平均系数 Top (相关性语言) ----------
    coef_mean = np.mean(coefs, axis=0)
    coef_std = np.std(coefs, axis=0)
    imp = pd.DataFrame({"feature": feats, "coef_mean": coef_mean,
                        "coef_std": coef_std})
    imp["abs"] = imp["coef_mean"].abs()
    imp = imp.sort_values("abs", ascending=False)
    imp_out = outdir / f"fusion_{tag}_lr_coefficients.csv"
    imp.to_csv(imp_out, index=False, encoding="utf-8-sig")
    print("\n=== LR 平均系数 Top 15 (相关性，非因果) ===")
    for _, r in imp.head(15).iterrows():
        tag = "AQ " if r["feature"].startswith("aq_") else "    "
        print(f"  [{tag}] {r['feature'][:46]:48s} coef={r['coef_mean']:+.3f}±{r['coef_std']:.3f}")

    print(f"\n单变量表: {uni_out}")
    print(f"系数表:   {imp_out}")


if __name__ == "__main__":
    main()
