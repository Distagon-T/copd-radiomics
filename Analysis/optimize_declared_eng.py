# -*- coding: utf-8 -*-
"""
保护区特征工程降维版：L1 LASSO + Linear SVM（留一队列）
=======================================================
用 declared_eng.py 把 9 个原始声明特征压缩成 5 个复合分（PAH_vascular / CAC /
FatInflam / CTR / BAR），保护区 = 5 复合分 + Pi10 + BV5 + BV10（共 8 项），
替代「原始 12 项保护区」（上一版 B SVM 0.7495，比 Pi10BV 的 0.7602 差，因
冗余 CAC 对 + 噪声 FAI/EpiFat_Mean_HU 被强制保留）。

复用 optimize_lasso_svm_binary.nested_loco_cfg 的嵌套流程：
每外层折内 Mann-Whitney+BH-FDR 筛选 → Spearman 去冗余 → L1 LASSO →
Linear SVM（C 折内选），保护区特征强制进候选并保留。

用法：
  python Analysis/optimize_declared_eng.py --configs old_std_l1 --strategies A,B
输出：E:\DICOM\reports\feature_selection_ordinal_ae\pi10bv_declared_eng\old_std_l1\*.csv
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lasso_svm_binary import load_frame, binary_metrics  # noqa: E402
from optimize_lasso_svm_binary import (BASE_OUT, CONFIGS, feature_composition,  # noqa: E402
                                       nested_loco_cfg, tune_svm_cfg, screen_protected, fit_lasso_cfg)
from declared_eng import PROTECTED_ENG, build_eng, engineered_feature_names  # noqa: E402

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEED = 20260830


def run_one_eng(cfg_name, cfg, strategy, protected):
    work, raw_feats, pid, _ = load_frame()
    work = build_eng(work)
    feature_names = engineered_feature_names(raw_feats)
    X = work[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    groups = work["cohort"].to_numpy(str)
    label = work["Label"].to_numpy(int)
    if strategy == "A":
        y_bin = (label >= 1).astype(int)
        train_idx = np.arange(len(work))
    else:
        train_idx = np.flatnonzero(np.isin(label, [0, 2]))
        y_bin = (label[train_idx] == 2).astype(int)
    oof, fold_df, sf = nested_loco_cfg(work, feature_names, X, groups, y_bin, train_idx,
                                       cfg, strategy, protected)
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
    sel_final = sf[sf["frequency"] > 0]["feature"].tolist()
    comp = feature_composition(sel_final)
    return {"config": cfg_name, "strategy": strategy, "pooled": pooled, "fold_df": fold_df,
            "sf": sf, "sel_final": sel_final, "comp": comp, "oof": oof}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="old_std_l1")
    ap.add_argument("--strategies", default="A,B")
    ap.add_argument("--protected", default=",".join(PROTECTED_ENG),
                    help="保护区子集（逗号分隔，须在 PROTECTED_ENG 内），默认全部 8 项")
    ap.add_argument("--tag", default="pi10bv_declared_eng", help="输出子目录名")
    args = ap.parse_args()
    protected = tuple(p.strip() for p in args.protected.split(",") if p.strip() and p.strip() in PROTECTED_ENG)
    if not protected:
        protected = tuple(PROTECTED_ENG)
    OUT = BASE_OUT / args.tag
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_list = [c.strip() for c in args.configs.split(",") if c.strip() in CONFIGS]
    strat_list = [s.strip().upper() for s in args.strategies.split(",") if s.strip().upper() in ("A", "B")]
    rows = []
    for cn in cfg_list:
        for st in strat_list:
            print(f"=== config={cn} strategy={st} protected_eng={protected} ===", flush=True)
            r = run_one_eng(cn, CONFIGS[cn], st, protected)
            sub = OUT / cn
            sub.mkdir(exist_ok=True)
            r["pooled"].assign(config=cn, strategy=st).to_csv(sub / f"pooled_{st}.csv", index=False)
            r["fold_df"].assign(config=cn, strategy=st).to_csv(sub / f"fold_{st}.csv", index=False)
            r["sf"].assign(config=cn, strategy=st).to_csv(sub / f"selection_{st}.csv", index=False)
            r["oof"].assign(config=cn, strategy=st).to_csv(sub / f"oof_{st}.csv", index=False)
            row = {"config": cn, "strategy": st}
            for _, m in r["pooled"].iterrows():
                row[f"{m['model']}_auc"] = m["auc"]
                row[f"{m['model']}_balacc"] = m["balanced_accuracy"]
            row.update({f"n_sel_{k}": v for k, v in r["comp"].items()})
            row["n_selected"] = len(r["sel_final"])
            rows.append(row)
            print(f"  done: LASSO auc={row.get('LASSO_auc', float('nan')):.4f} SVM auc={row.get('SVM_auc', float('nan')):.4f} "
                  f"n_selected={row['n_selected']} comp={r['comp']}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "config_results.csv", index=False)
    (OUT / "protected_eng.json").write_text(
        json.dumps({"tag": args.tag, "protected_used": list(protected), "eng_spec": {
            k: v for k, v in
            {"PAH_vascular": ["Vessel_CSA_mean_mm2", "PA_Equivalent_Diameter_mm"],
             "CAC_score": ["CAC_Agatston"],
             "FatInflam_score": ["EpiFat_Volume_mm3", "FAI_pericoronary_HU", "EpiFat_Mean_HU"],
             "CTR_score": ["CardioThoracic_Ratio"],
             "BAR_score": ["BronchoArtery_Ratio"]}.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved ->", OUT)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
