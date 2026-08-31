# -*- coding: utf-8 -*-
"""
旧版特征选择的优化版（基座 = lasso_svm_binary.py）
=====================================================
背景：当前 glmnet/elastic-net 的 top-25 特征被 firstorder_Energy/TotalEnergy 等
强度特征主导（20/17 个），而这些强度特征受 CT 扫描协议混杂污染（队内标准化后
AUC 塌到 ~0.55）。旧版 `lasso_svm_binary.py` 的嵌套筛选（Mann-Whitney + FDR +
Spearman 去冗余 + L1 logistic LASSO）在 Strategy B 选出 17 个特征里只有 1 个
强度特征，LASSO AUC=0.741 —— 特征组成健康得多。

本脚本以旧版特征选择为基座，做如下优化并做受控对比：
  1) scaler: standard vs robust（中位数/IQR）——阻止巨大方差强度特征霸占 L1 系数
  2) penalty: l1 (LASSO) vs elasticnet (l1_ratio∈{0.5,0.8,1.0}，即 glmnet 的 alpha)
  3) scoring: neg_log_loss vs roc_auc（折内选 λ 的准则）
  4) SVM 核：linear（AUC 用 decision function；linear 只调 C）
输出：每 config×strategy 的 pooled OOF 指标（LASSO+SVM，decision AUC）、
每折选出的特征组成（强度/形态/纹理/气道血管）、跨折选择稳定性(Jaccard)，
最后汇总对比报告。

用法：
  python Analysis/optimize_lasso_svm_binary.py --configs old_std_l1,robust_l1,robust_enet_auc --strategies A,B
输出目录：E:\DICOM\reports\feature_selection_ordinal_ae\optimize_lasso_svm\
"""
import argparse
import base64
import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, roc_auc_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from lasso_svm_binary import (GRAY_HIGH, GRAY_LOW, binary_metrics, bootstrap_binary_ci,
                              load_frame, risk_score_summary, screen_features_binary)
from lasso_svm_nested import PARAM_GRID, SEED, bh_fdr, find_col, is_feature

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_OUT = Path(r"E:\DICOM\reports\feature_selection_ordinal_ae")
OUT = BASE_OUT / "optimize_lasso_svm"

CONFIGS = {
    "old_std_l1": dict(scaler="standard", penalty="l1", scoring="neg_log_loss"),
    "robust_l1": dict(scaler="robust", penalty="l1", scoring="neg_log_loss"),
    "robust_enet_auc": dict(scaler="robust", penalty="elasticnet", scoring="roc_auc"),
}

# 保护区特征：无论单变量筛选/FDR/去冗余/LASSO 结果如何，都强制进入候选池并保留在最终模型里
PROTECTED = []


def get_scaler(name):
    return RobustScaler() if name == "robust" else StandardScaler()


def screen_protected(X, y, names, protected):
    """包装 screen_features_binary，把保护区特征强制加入候选列表。"""
    selected, stats_df = screen_features_binary(X, y, names)
    if protected:
        for nm in protected:
            if nm in names and nm not in selected:
                selected.append(nm)
    return selected, stats_df


def fit_lasso_cfg(X, y, names, cfg, protected=()):
    """L1 或 elastic-net logistic 回归，可配置 scaler 与折内评分准则。
    注意：调用方应传入<已筛选>的候选特征（~30 个），不要在全部 2267 特征上直接跑。
    protected: 保护区特征名列表，即使 LASSO 系数为 0 也强制保留在最终选中集里。
    """
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    if cfg["penalty"] == "elasticnet":
        lr = LogisticRegressionCV(
            Cs=12, cv=inner, penalty="elasticnet", l1_ratios=[0.5, 0.8, 1.0],
            solver="saga", class_weight="balanced", scoring=cfg["scoring"],
            tol=1e-3, max_iter=3000, n_jobs=1, random_state=SEED, refit=True,
        )
    else:
        lr = LogisticRegressionCV(
            Cs=12, cv=inner, penalty="l1", solver="liblinear",
            class_weight="balanced", scoring=cfg["scoring"],
            max_iter=2000, random_state=SEED, refit=True,
        )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", get_scaler(cfg["scaler"])),
        ("lasso", lr),
    ])
    pipe.fit(X, y)
    coef = pipe.named_steps["lasso"].coef_[:, : len(names)]
    keep = np.flatnonzero(np.max(np.abs(coef), axis=0) > 1e-7)
    if len(keep) == 0:
        keep = np.argsort(np.max(np.abs(coef), axis=0))[::-1][: min(3, len(names))]
    keep = list(keep)
    if protected:
        for i, nm in enumerate(names):
            if nm in protected and i not in keep:
                keep.append(i)
    keep = sorted(set(keep))
    return pipe, [names[i] for i in keep]


C_GRID_LINEAR = [0.01, 0.1, 1, 10, 100]
SVM_KERNEL = "linear"


def make_svm_cfg(C, cfg, probability=False):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", get_scaler(cfg["scaler"])),
        ("svm", SVC(kernel=SVM_KERNEL, C=C, probability=probability,
                    class_weight="balanced", random_state=SEED)),
    ])


def tune_svm_cfg(X, y, feature_names, cfg, protected=()):
    """折内 3-fold balanced accuracy 选 C（linear SVM）。
    关键：先在训练折内做单变量筛选（~30 候选）→ LASSO 选择 → 再用选出的特征训 linear SVM。
    旧版 bug 是在全部 2267 特征上直接跑 LogisticRegressionCV，导致极慢。
    protected: 保护区特征（如 Pi10）强制进候选并保留。
    """
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = {C: [] for C in C_GRID_LINEAR}
    for tr, va in inner.split(X, y):
        sel, _ = screen_protected(X[tr], y[tr], feature_names, protected)
        if not sel:
            sel = feature_names[:3]
        idx = [feature_names.index(f) for f in sel]
        _, lasso_sel = fit_lasso_cfg(X[tr][:, idx], y[tr], sel, cfg, protected)
        if not lasso_sel:
            lasso_sel = sel[:3]
        sv_idx = [sel.index(f) for f in lasso_sel]
        for C in C_GRID_LINEAR:
            m = make_svm_cfg(C, cfg)
            m.fit(X[tr][:, idx][:, sv_idx], y[tr])
            scores[C].append(balanced_accuracy_score(y[va], m.predict(X[va][:, idx][:, sv_idx])))
    means = {C: float(np.mean(v)) if v else -np.inf for C, v in scores.items()}
    best_C = max(means, key=means.get)
    return {"C": float(best_C)}, means


def feature_composition(features):
    """把特征分为 强度(Energy/TotalEnergy)/其他firstorder/形态学shape/纹理/结构指标。"""
    n_intensity = n_shape = n_texture = n_fo_other = n_struct = 0
    for f in features:
        if re.search(r"firstorder_(TotalEnergy|Energy)(?:_|$)", f) or re.search(r"firstorder_(Energy|TotalEnergy)$", f):
            n_intensity += 1
        elif "shape_" in f:
            n_shape += 1
        elif re.search(r"_(glcm|glrlm|glszm|gldm|ngtdm)_", f):
            n_texture += 1
        elif "firstorder_" in f:
            n_fo_other += 1
        else:
            n_struct += 1
    return {"intensity": n_intensity, "shape": n_shape, "texture": n_texture,
            "firstorder_other": n_fo_other, "structural": n_struct}


def nested_loco_cfg(work, feature_names, X, groups, y_bin, train_idx, cfg, strategy, protected=()):
    """留一队列嵌套 LASSO/elastic-net -> SVM。返回 oof, fold_df, selection_df。"""
    train_groups = groups[train_idx]
    outer = GroupKFold(n_splits=len(np.unique(train_groups)))
    oof_rows, fold_rows, sel_rows = [], [], []
    label = work["Label"].to_numpy(int)
    for fold, (tr, te) in enumerate(outer.split(X[train_idx], y_bin, train_groups), 1):
        trr, ter = train_idx[tr], train_idx[te]
        # 折内特征选择（旧版：screen + LASSO），保护区特征强制保留
        best, _ = tune_svm_cfg(X[trr], y_bin[tr], feature_names, cfg, protected)
        sel, screen_df = screen_protected(X[trr], y_bin[tr], feature_names, protected)
        screen_df["outer_fold"] = fold
        idx = [feature_names.index(f) for f in sel]
        lasso, lasso_sel = fit_lasso_cfg(X[trr][:, idx], y_bin[tr], sel, cfg, protected)
        if not lasso_sel:
            lasso_sel = sel[:3]
        sv_idx = [sel.index(f) for f in lasso_sel]
        sel_rows.extend({"fold": fold, "feature": f} for f in lasso_sel)
        # 测试折 + （策略B）held-out Label1
        test_cohorts = set(groups[ter])
        pred_idx = list(ter)
        if strategy == "B":
            extra = [i for i in range(len(work)) if i not in set(train_idx) and groups[i] in test_cohorts]
            pred_idx = pred_idx + extra
        pred_idx = np.array(pred_idx, dtype=int)
        lasso_pred = lasso.predict(X[pred_idx][:, idx]).astype(int)
        lasso_score = lasso.decision_function(X[pred_idx][:, idx])
        svm = make_svm_cfg(best["C"], cfg).fit(X[trr][:, idx][:, sv_idx], y_bin[tr])
        svm_pred = svm.predict(X[pred_idx][:, idx][:, sv_idx]).astype(int)
        svm_score = svm.decision_function(X[pred_idx][:, idx][:, sv_idx])
        pos_of = {int(r): i for i, r in enumerate(pred_idx)}
        te_pos = np.array([pos_of[int(r)] for r in ter])
        y_te = y_bin[te]
        lm = binary_metrics(y_te, lasso_pred[te_pos], lasso_score[te_pos])
        sm = binary_metrics(y_te, svm_pred[te_pos], svm_score[te_pos])
        for model, met in [("LASSO", lm), ("SVM", sm)]:
            fold_rows.append({"fold": fold, "test_cohort": ";".join(sorted(test_cohorts)), "model": model,
                              "best_params": json.dumps(best), "n_screened": len(sel), "n_lasso_selected": len(lasso_sel),
                              "comp": json.dumps(feature_composition(lasso_sel)), **met})
        for i, row_i in enumerate(pred_idx):
            oof_rows.append({"row_index": int(row_i), "PatientID": str(work.iloc[row_i]["PatientID"]),
                             "cohort": groups[row_i], "Label": int(label[row_i]),
                             "in_train": bool(row_i in set(train_idx)),
                             "lasso_pred": int(lasso_pred[i]), "svm_pred": int(svm_pred[i]),
                             "lasso_score": float(lasso_score[i]), "svm_score": float(svm_score[i])})
    oof = pd.DataFrame(oof_rows).sort_values("row_index").reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows)
    sf = pd.DataFrame({"feature": feature_names})
    counts = Counter(r["feature"] for r in sel_rows)
    sf["frequency"] = sf["feature"].map(counts).fillna(0).astype(int)
    sf["selection_rate"] = sf["frequency"] / len(np.unique(train_groups))
    sf.sort_values(["frequency", "feature"], ascending=[False, True], inplace=True)
    return oof, fold_df, sf


def run_one(cfg_name, cfg, strategy, protected=()):
    work, feature_names, pid, _ = load_frame()
    X = work[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    groups = work["cohort"].to_numpy(str)
    label = work["Label"].to_numpy(int)
    if strategy == "A":
        y_bin = (label >= 1).astype(int)
        train_idx = np.arange(len(work))
    else:
        train_idx = np.flatnonzero(np.isin(label, [0, 2]))
        y_bin = (label[train_idx] == 2).astype(int)
    oof, fold_df, sf = nested_loco_cfg(work, feature_names, X, groups, y_bin, train_idx, cfg, strategy, protected)
    # pooled 指标：A 用全部；B 只用 train 折内的 0/2
    if strategy == "A":
        y_pool = (oof["Label"].to_numpy(int) >= 1).astype(int)
        oof_use = oof
    else:
        oof_use = oof[oof["in_train"]].reset_index(drop=True)
        y_pool = (oof_use["Label"].to_numpy(int) == 2).astype(int)
    pooled = pd.DataFrame([
        {"model": "LASSO", **binary_metrics(y_pool, oof_use["lasso_pred"].to_numpy(int), oof_use["lasso_score"].to_numpy(float))},
        {"model": "SVM", **binary_metrics(y_pool, oof_use["svm_pred"].to_numpy(int), oof_use["svm_score"].to_numpy(float))},
    ])
    # 特征组成（按选择频率 top 取整折并集）
    sel_final = sf[sf["frequency"] > 0]["feature"].tolist()
    comp = feature_composition(sel_final)
    return {"config": cfg_name, "strategy": strategy, "pooled": pooled, "fold_df": fold_df,
            "sf": sf, "sel_final": sel_final, "comp": comp, "oof": oof}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="old_std_l1,robust_l1,robust_enet_auc")
    ap.add_argument("--strategies", default="A,B")
    ap.add_argument("--protected", default="", help="逗号分隔的保护区特征名（如 Pi10），强制进候选并保留")
    ap.add_argument("--tag", default="", help="输出子目录名（覆盖自动生成的长名），如 pi10bv_declared9")
    args = ap.parse_args()
    protected = tuple(p.strip() for p in args.protected.split(",") if p.strip())
    global PROTECTED, OUT
    PROTECTED = list(protected)
    if protected:
        OUT = BASE_OUT / (args.tag if args.tag else "optimize_lasso_svm_" + "_".join(protected))
    cfg_list = [c.strip() for c in args.configs.split(",") if c.strip() in CONFIGS]
    strat_list = [s.strip().upper() for s in args.strategies.split(",") if s.strip().upper() in ("A", "B")]
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    done = set()
    res_csv = OUT / "config_results.csv"
    if res_csv.exists():
        try:
            done = set((str(r["config"]), str(r["strategy"])) for _, r in pd.read_csv(res_csv).iterrows())
            print("[cache] done:", done)
        except Exception:
            pass
    for cn in cfg_list:
        for st in strat_list:
            if (cn, st) in done:
                continue
            print(f"=== config={cn} strategy={st} protected={protected} ===", flush=True)
            r = run_one(cn, CONFIGS[cn], st, protected)
            sub = OUT / f"{cn}"
            sub.mkdir(exist_ok=True)
            r["pooled"].assign(config=cn, strategy=st).to_csv(sub / f"pooled_{st}.csv", index=False)
            r["fold_df"].assign(config=cn, strategy=st).to_csv(sub / f"fold_{st}.csv", index=False)
            r["sf"].assign(config=cn, strategy=st).to_csv(sub / f"selection_{st}.csv", index=False)
            r["oof"].assign(config=cn, strategy=st).to_csv(sub / f"oof_{st}.csv", index=False)
            results.append(r)
            # 追加到汇总
            row = {"config": cn, "strategy": st}
            for _, m in r["pooled"].iterrows():
                row[f"{m['model']}_auc"] = m["auc"]
                row[f"{m['model']}_balacc"] = m["balanced_accuracy"]
            row.update({f"n_sel_{k}": v for k, v in r["comp"].items()})
            row["n_selected"] = len(r["sel_final"])
            pr = pd.DataFrame([row])
            if res_csv.exists():
                pr.to_csv(res_csv, mode="a", header=False, index=False)
            else:
                pr.to_csv(res_csv, index=False)
            print(f"  done: LASSO auc={row.get('LASSO_auc', float('nan')):.4f} SVM auc={row.get('SVM_auc', float('nan')):.4f} "
                  f"n_selected={row['n_selected']} comp={r['comp']}", flush=True)
    build_report(res_csv)


def build_report(res_csv):
    df = pd.read_csv(res_csv)
    html = ["<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>",
            "<title>旧版特征选择优化对比</title>",
            "<style>body{font-family:'Microsoft YaHei',Arial;margin:24px;color:#222}h1{font-size:22px}"
            "h2{font-size:17px;border-bottom:2px solid #bbb;padding-bottom:4px;margin-top:30px}"
            "table{border-collapse:collapse;font-size:13px;margin:10px 0}th,td{border:1px solid #ccc;padding:5px 8px}"
            "th{background:#f0f0f0}.best{background:#d4edda}.warn{background:#fff3cd}.note{color:#555;font-size:13px}</style></head><body>"]
    html.append("<h1>LASSO/glmnet 特征选择优化对比（基座：旧版 lasso_svm_binary.py）</h1>")
    html.append("<p class='note'>对比配置：scaler(standard/robust) × penalty(l1/elasticnet) × 折内评分(neg_log_loss/roc_auc)。"
                "AUC 均用 decision function；外层留一队列(2026-01/02/04/05)。</p>")
    for st in ["A", "B"]:
        sub = df[df["strategy"] == st]
        html.append(f"<h2>Strategy {st}</h2>")
        html.append(sub.to_html(index=False, border=0, classes="", float_format=lambda x: f"{x:.4f}"))
    html.append("<h2>说明</h2><ul>")
    html.append("<li>n_sel_intensity = 选中特征里 firstorder Energy/TotalEnergy 的数量；shape=形态学；texture=纹理；structural=气道/血管/肺叶等指标。</li>")
    html.append("<li>优化目标：在 AUC 不降的前提下，让特征选择摆脱强度特征主导（即 n_sel_intensity 明显变小、组成更均衡），以降低扫描协议混杂风险。</li>")
    html.append("</ul></body></html>")
    out_html = OUT / "optimize_lasso_svm_report.html"
    out_html.write_text("\n".join(html), encoding="utf-8")
    print("report =", out_html)


if __name__ == "__main__":
    main()
