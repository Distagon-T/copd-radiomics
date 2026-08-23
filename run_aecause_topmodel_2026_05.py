#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_aecause_topmodel_2026_05.py
===============================
急性期 COPD 感染型 vs 非感染型：Top 显著特征精简模型 + Bootstrap 验证，
并融合进 report_copd_ae_cause_2026_05。

 1) 读取 rad+aq 单变量 CSV，按 |AUC-0.5| 取 top5/top10/top20
 2) 手工临床集（6 特征）：肺血管 Mean + 肺气肿 LAA950 + Perc15_HU + 管壁峰态 + 边界模糊 + 内径
 3) 每个特征集：5 折 CV 点估计 + 分层 bootstrap 重采样 CV（AUC mean ± 95%CI + 稳定性）
 4) 与全特征 rad+aq 对比，出图，插入报告新章节并重生成 HTML
"""
import base64
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_fusion_model import load_and_join, select_features
from run_copd_ae_cause_2026_05 import SEG, FIGDIR, Args, LABEL_COL, build_labels

SEED = 42
N_BOOT = 150
MD_OUT = os.path.join(SEG, "report_copd_ae_cause_2026_05.md")
HTML_OUT = os.path.join(SEG, "report_copd_ae_cause_2026_05.html")
UNI_CSV = os.path.join(SEG, "fusion_aecause_rad+aq_univariate_top.csv")

# 手工临床集（假说驱动）：肺血管灌注 + 肺气肿 + 肺密度 + 管壁峰态 + 边界模糊 + 内径
CLINICAL_SET = [
    "lung_vessels::original_firstorder_Mean",
    "Lobe_LLL_LAA950_pct",
    "Lobe_RLL_Perc15_HU",
    "aq_wall_hu_kurt",
    "aq_blur_trans_width_std",
    "aq_Din_mean_gen3",
]


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


def plot_roc2(models, out_path, n, n_pos):
    plt.figure(figsize=(6.5, 6.5))
    for tag, Xv, y in models:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        tprs, base_fpr = [], np.linspace(0, 1, 101)
        scaler = StandardScaler()
        for tr, te in skf.split(Xv, y):
            Xtr = scaler.fit_transform(Xv[tr]); Xte = scaler.transform(Xv[te])
            clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                     class_weight="balanced", max_iter=1000, random_state=SEED)
            clf.fit(Xtr, y[tr])
            fpr, tpr, _ = roc_curve(y[te], clf.predict_proba(Xte)[:, 1])
            tprs.append(np.interp(base_fpr, fpr, tpr)); tprs[-1][0] = 0.0
        mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
        mean_auc = float(np.trapz(mean_tpr, base_fpr))
        plt.plot(base_fpr, mean_tpr, lw=2, label=f"{tag} (AUC={mean_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"Acute COPD cause Top-model ROC (n={n}, pos={n_pos})")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print("ROC ->", out_path)


def md_to_html(md_text, b64img):
    lines = md_text.splitlines()
    html = ["<h1>急性期 COPD：感染型 vs 非感染型加重（Radiomics + AirQuant）</h1>"]
    in_table = False
    for ln in lines[3:]:
        if ln.startswith("## "):
            html.append(f"<h2>{ln[3:]}</h2>")
        elif ln.startswith("### "):
            html.append(f"<h3>{ln[4:]}</h3>")
        elif ln.startswith("|") and "|---|---|" not in ln:
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if not in_table:
                html.append("<table><thead><tr>")
                for c in cells:
                    html.append(f"<th>{c}</th>")
                html.append("</tr></thead><tbody>"); in_table = True
            else:
                html.append("<tr>")
                for c in cells:
                    html.append(f"<td>{c}</td>")
                html.append("</tr>")
        elif ln.startswith("![") and "](" in ln:
            cap = ln[2:].split("](")[0]
            path = ln.split("](")[1].rstrip(")")
            full = os.path.join(SEG, path)
            if os.path.exists(full):
                html.append(f'<img src="data:image/png;base64,{b64img(full)}" '
                            f'style="max-width:95%;height:auto;display:block;margin:10px auto;" '
                            f'alt="{cap}"/>')
                html.append(f'<p style="text-align:center;color:#555;font-size:0.9em">{cap}</p>')
        elif ln.strip() == "":
            if in_table:
                html.append("</tbody></table>"); in_table = False
        else:
            if in_table:
                html.append("</tbody></table>"); in_table = False
            html.append(f"<p>{ln}</p>")
    return "\n".join(html)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    build_labels()
    df = load_and_join(Args())
    df = df[df[LABEL_COL].notna()].copy()
    y = df[LABEL_COL].values.astype(int)
    n = len(y); n_pos = int(y.sum())
    print(f"[model] 样本 {n}, 感染型 {n_pos} / 非感染型 {n - n_pos}")

    feats_all, _ = select_features(df, LABEL_COL)

    uni = pd.read_csv(UNI_CSV)
    uni["auc_dev"] = (uni["auc_univ"] - 0.5).abs()
    uni = uni.sort_values("auc_dev", ascending=False)
    top5 = uni.head(5)["feature"].tolist()
    top10 = uni.head(10)["feature"].tolist()
    top20 = uni.head(20)["feature"].tolist()
    clinical = [c for c in CLINICAL_SET if c in feats_all]

    sets = [
        ("top5", top5),
        ("top10", top10),
        ("top20", top20),
        ("clinical6", clinical),
        ("rad+aq(full)", feats_all),
    ]

    def prep(fs):
        X = df[fs].apply(pd.to_numeric, errors="coerce").fillna(
            df[fs].apply(pd.to_numeric, errors="coerce").median())
        return X.values.astype(np.float64)

    rows = []
    for tag, fs in sets:
        Xv = prep(fs)
        pt, folds = model_auc_cv(Xv, y)
        b = bootstrap_model_auc(Xv, y)
        lo, hi = np.percentile(b, [2.5, 97.5])
        stab = float(np.mean(b > 0.5))
        rows.append({"model": tag, "n_feat": len(fs), "cv_auc": pt,
                     "boot_mean": float(np.mean(b)), "ci_lo": lo, "ci_hi": hi,
                     "stability": stab, "Xv": Xv})
        print(f"[{tag}] n_feat={len(fs)} CV_AUC={pt:.3f} "
              f"boot_AUC={np.mean(b):.3f} (95%CI {lo:.3f}-{hi:.3f}) stability={stab:.0%}")

    # 图1: 模型 AUC 误差条（bootstrap 均值 ± 95%CI，另标注点估计）
    tags = [r["model"] for r in rows]
    bmeans = [r["boot_mean"] for r in rows]
    los = [r["ci_lo"] for r in rows]
    his = [r["ci_hi"] for r in rows]
    cvs = [r["cv_auc"] for r in rows]
    plt.figure(figsize=(8.5, 5))
    xpos = np.arange(len(rows))
    colors = ["#1f77b4"] * (len(rows) - 1) + ["#d62728"]
    yerr_lo = np.clip([b - lo for b, lo in zip(bmeans, los)], 0, None)
    yerr_hi = np.clip([hi - b for hi, b in zip(his, bmeans)], 0, None)
    plt.errorbar(xpos, bmeans, yerr=[yerr_lo, yerr_hi],
                 fmt="none", ecolor="#999", capsize=4, zorder=2)
    plt.scatter(xpos, bmeans, c=colors, s=120, zorder=4, label="bootstrap mean")
    plt.scatter(xpos, cvs, marker="x", s=90, c="#222", zorder=5, label="5-fold CV point")
    for xp, b, c in zip(xpos, bmeans, cvs):
        plt.text(xp, b + 0.015, f"CV={c:.3f}", ha="center", fontsize=8)
    plt.axhline(0.5, color="k", ls="--", lw=1)
    plt.xticks(xpos, tags)
    plt.ylabel("AUC"); plt.ylim(0.4, 1.0)
    plt.title(f"Acute COPD cause: reduced-model AUC (errorbar=bootstrap 95%CI, n={n})")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "fig_aecause_model_auc_2026_05.png"), dpi=150)
    plt.close()
    print("AUC 图 ->", os.path.join(FIGDIR, "fig_aecause_model_auc_2026_05.png"))

    # 图2: top10 vs full ROC
    plot_roc2([(r["model"], r["Xv"], y) for r in rows if r["model"] in ("top10", "rad+aq(full)")],
              os.path.join(FIGDIR, "fig_roc_aecause_top_2026_05.png"), n, n_pos)

    # ---- 融合进报告 ----
    with open(MD_OUT, "r", encoding="utf-8") as f:
        md = f.read()

    new_sec = []
    new_sec.append("## 8. 精简模型验证（Top 显著特征 + Bootstrap）\n")
    new_sec.append(f"> 分层 5 折 CV 点估计 + 分层 bootstrap 重采样 CV（{N_BOOT} 次，AUC 95%CI + 稳定性 = P(AUC>0.5)）\n")
    new_sec.append("| 模型 | 特征数 | 5折CV AUC | bootstrap AUC 均值 | 95%CI | 稳定性 |")
    new_sec.append("|---|---|---|---|---|---|")
    for r in rows:
        new_sec.append(f"| {r['model']} | {r['n_feat']} | {r['cv_auc']:.3f} | "
                       f"{r['boot_mean']:.3f} | {r['ci_lo']:.3f}–{r['ci_hi']:.3f} | {r['stability']:.0%} |")
    new_sec.append("")
    new_sec.append("**top10 特征**：")
    new_sec.append("")
    for c in top10:
        new_sec.append(f"- `{c}`")
    new_sec.append("")
    new_sec.append("**手工临床集（假说驱动，6 特征）**：肺血管灌注(Mean) + 肺气肿(LAA950) + 肺密度(Perc15) + 管壁峰态(kurt) + 边界模糊(blur_width) + 内径(Din)。")
    new_sec.append("")
    new_sec.append("![图 7. 精简模型 AUC（误差条 = bootstrap 95%CI）](figs/fig_aecause_model_auc_2026_05.png)\n")
    new_sec.append("*图 7. 精简模型 AUC（误差条 = bootstrap 95%CI）*\n")
    new_sec.append("![图 8. top10 vs 全特征 5 折 CV 平均 ROC](figs/fig_roc_aecause_top_2026_05.png)\n")
    new_sec.append("*图 8. top10 vs 全特征 5 折 CV 平均 ROC*\n")

    new_md = "\n".join(new_sec)

    old_concl = "## 7. 结论"
    if old_concl in md:
        md = md.replace(old_concl, new_md + "## 8. 结论")
    else:
        md = md.rstrip() + "\n\n" + new_md

    best_top = min(rows, key=lambda r: -r["cv_auc"])
    concl_note = (f"\n- 精简模型验证：**{best_top['model']}** 保持最高判别力 "
                  f"(5折CV AUC = {best_top['cv_auc']:.3f}, bootstrap 95%CI {best_top['ci_lo']:.3f}–{best_top['ci_hi']:.3f}, "
                  f"稳定性 {best_top['stability']:.0%})，证明加重病因信号主要由少量显著特征承载。\n")
    md = md.replace("- 临床意义：感染型加重常伴实变/浸润/气道炎症",
                    concl_note + "- 临床意义：感染型加重常伴实变/浸润/气道炎症")

    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write(md)

    def b64img(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(md_to_html(md, b64img))
    print(f"[report] {MD_OUT}\n[report] {HTML_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
