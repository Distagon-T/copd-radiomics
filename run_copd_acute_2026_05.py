#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_copd_acute_2026_05.py
=========================
纯 COPD 队列：急性加重(AECOPD) vs 稳定期(单纯慢阻肺) 二分类。

 1) 剔除「全部支扩患者」610 例（主诊断以 支气管扩张/支气管扩张症/细支气管扩张症 开头）
    ——包括医生标注 COPD合并支扩(BCOS) 的 78 例也一并剔除
 2) 保留剩余 COPD 约 500 例
    Label=1 : 主诊断包含「急性加重」（慢性阻塞性肺病伴有急性加重 / 慢性阻塞性肺疾病急性加重）
    Label=0 : 单纯「慢性阻塞性肺疾病」（稳定期）
    其余 COPD 诊断（慢性支气管炎急性发作/伴急性下呼吸道感染/合并肺部感染等）不入模型
 3) 特征：pyRadiomics + MATLAB AirQuant(含 blur_*/TD_* 新列)
 4) 消融对比：radiomics | +airquant
 5) 输出图 + report_copd_acute_2026_05.md/.html
"""
import argparse
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
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_fusion_model import load_and_join, norm_id, select_features, univariate_summary
from plot_consistency_2026_05 import bootstrap_auc, forest_plot

SEG = "seg_results"
INFO = "info.csv"
RAD = os.path.join(SEG, "radiomics_2026_05_features.csv")
AQ = os.path.join(SEG, "airquant_2026_05_aggregated.csv")
LAB_OUT = os.path.join(SEG, "labels_copd_acute_2026_05.csv")
FIGDIR = os.path.join(SEG, "figs")
MD_OUT = os.path.join(SEG, "report_copd_acute_2026_05.md")
HTML_OUT = os.path.join(SEG, "report_copd_acute_2026_05.html")
SEED = 42
N_BOOT = 200
TOP_N = 15
LABEL_COL = "COPD_AE_Label"
ACUTE_KW = ["急性加重"]
STABLE_DIAG = "慢性阻塞性肺疾病"
BRONCH_PREFIX = ["支气管扩张", "支气管扩张症", "细支气管扩张症"]


def is_bronch(d):
    return any(d.startswith(b) for b in BRONCH_PREFIX)


def build_copd_labels():
    info = pd.read_csv(INFO, encoding="utf-8")
    info["_pid"] = info["患者id"].astype(str).str.strip().str.lstrip("0")

    rad = pd.read_csv(RAD, usecols=["Patient_ID", "PatientID"])
    rad["_pid"] = norm_id(rad["PatientID"].astype(str))
    pid2folder = rad.drop_duplicates("_pid").set_index("_pid")["Patient_ID"].to_dict()
    info["Patient_ID"] = info["_pid"].map(pid2folder)

    diag = info["主要诊断"].fillna("").astype(str)
    info["_diag"] = diag
    info["is_bronch"] = diag.apply(is_bronch)
    copd = info[~info["is_bronch"]].copy()          # 剔除全部支扩(610)

    d2 = copd["_diag"]
    copd["is_acute"] = d2.str.contains("急性加重")
    copd["is_stable"] = (d2 == STABLE_DIAG)
    copd["labeled"] = copd["is_acute"] | copd["is_stable"]
    copd["COPD_AE_Label"] = np.nan
    copd.loc[copd["is_acute"], "COPD_AE_Label"] = 1
    copd.loc[copd["is_stable"], "COPD_AE_Label"] = 0

    out = copd.rename(columns={"_pid": "patient_id", "_diag": "main_diagnosis"})
    out = out[["patient_id", "Patient_ID", "main_diagnosis",
               "COPD_AE_Label", "is_acute", "is_stable", "labeled"]]
    out.to_csv(LAB_OUT, index=False, encoding="utf-8-sig")

    n_br = int(info["is_bronch"].sum())
    n_ac = int(out["is_acute"].sum())
    n_st = int(out["is_stable"].sum())
    n_oth = int((~out["labeled"]).sum())
    print(f"[cohort] 剔除全部支扩 {n_br} | 保留 COPD {len(out)}")
    print(f"[label] 急性加重(1) {n_ac} | 单纯慢阻肺(0) {n_st} | 其他排除 {n_oth}")
    return out, n_br


class Args:
    radiomics = RAD
    airquant = AQ
    labels = LAB_OUT


def run_ablation(df, feats, tag, label):
    """给定特征子集跑分层 5 折 CV LR + 单变量，返回结果 dict"""
    y = label
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(
        df[feats].apply(pd.to_numeric, errors="coerce").median())
    Xv = X.values.astype(np.float64)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs, accs, sens, specs, coefs = [], [], [], [], []
    scaler = StandardScaler()
    for tr, te in skf.split(Xv, y):
        Xtr = scaler.fit_transform(Xv[tr]); Xte = scaler.transform(Xv[te])
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                 class_weight="balanced", max_iter=1000, random_state=SEED)
        clf.fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)[:, 1]
        pred = clf.predict(Xte)
        aucs.append(roc_auc_score(y[te], proba))
        accs.append(accuracy_score(y[te], pred))
        tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
        sens.append(tp / (tp + fn) if (tp + fn) else 0.0)
        specs.append(tn / (tn + fp) if (tn + fp) else 0.0)
        coefs.append(clf.coef_[0])
    res = {"tag": tag, "n_feat": len(feats), "auc": float(np.mean(aucs)),
           "auc_std": float(np.std(aucs)), "acc": float(np.mean(accs)),
           "sens": float(np.mean(sens)), "spec": float(np.mean(specs)),
           "aucs": aucs, "coefs": coefs, "feats": feats}
    print(f"[{tag}] n_feat={len(feats)} AUC={res['auc']:.3f}±{res['auc_std']:.3f} "
          f"Acc={res['acc']:.3f} Sens={res['sens']:.3f} Spec={res['spec']:.3f}")
    # 单变量
    uni = univariate_summary(df, feats, y, top=50)
    uni["auc_dev"] = (uni["auc_univ"] - 0.5).abs()
    uni.to_csv(os.path.join(SEG, f"fusion_copd_acute_{tag}_univariate_top.csv"),
               index=False, encoding="utf-8-sig")
    imp = pd.DataFrame({"feature": feats, "coef_mean": np.mean(coefs, axis=0),
                        "coef_std": np.std(coefs, axis=0)})
    imp["abs"] = imp["coef_mean"].abs()
    imp = imp.sort_values("abs", ascending=False)
    imp.to_csv(os.path.join(SEG, f"fusion_copd_acute_{tag}_lr_coefficients.csv"),
               index=False, encoding="utf-8-sig")
    res["uni"] = uni
    return res


def plot_roc(results, out_path):
    plt.figure(figsize=(6.5, 6.5))
    for r in results:
        y = r["_y"]; Xv = r["_Xv"]
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
        plt.plot(base_fpr, mean_tpr, lw=2,
                 label=f"{r['tag']} (AUC={r['auc']:.3f}±{r['auc_std']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"Pure COPD: AECOPD vs Stable (n={len(y)}, pos={int(y.sum())})")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print("ROC ->", out_path)


def shorten(name, n=44):
    return name if len(name) <= n else name[: n - 3] + "..."


def md_to_html(md_text, b64img):
    lines = md_text.splitlines()
    html = ["<h1>纯 COPD 队列：急性加重 vs 稳定期（Radiomics + AirQuant）</h1>"]
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


def write_report(results, cohort, oth_dist):
    L = []
    L.append("# 纯 COPD 队列：急性加重(AECOPD) vs 稳定期(SCOPD) 判别（Radiomics + AirQuant）\n")
    L.append(f"> 报告时间：2026-08-22　|　保留 COPD 队列 n={cohort['copd']}（剔除全部支扩 {cohort['excluded']}）　|　"
             f"建模样本 n={cohort['model_n']}（AECOPD {cohort['pos']} / 稳定 {cohort['neg']}）\n")
    L.append("## 1. 队列与标签\n")
    L.append("| 步骤 | 说明 |")
    L.append("|---|---|")
    L.append(f"| 剔除 | 删除全部支扩患者 {cohort['excluded']} 例（主诊断以 支气管扩张/支气管扩张症/细支气管扩张症 开头，含 BCOS 78 例） |")
    L.append(f"| 保留 | 剩余 COPD {cohort['copd']} 例 |")
    L.append("| Label=1 | 主诊断含「急性加重」→ AECOPD（慢性阻塞性肺病伴有急性加重 + 慢性阻塞性肺疾病急性加重） |")
    L.append("| Label=0 | 单纯「慢性阻塞性肺疾病」→ 稳定期 |")
    L.append("| 排除 | 其余 COPD 诊断（非“急性加重 vs 单纯稳定”二分类）不入模型 |")
    L.append("| 特征 | pyRadiomics + MATLAB AirQuant（含 FWHM 管壁密度/边界模糊/T-D 新列） |")
    L.append("")
    L.append("## 2. 队列诊断构成\n")
    L.append("| 类别 | 例数 |")
    L.append("|---|---|")
    L.append(f"| 急性加重（Label=1） | {cohort['pos_all']} |")
    L.append(f"| 单纯慢阻肺（Label=0） | {cohort['neg_all']} |")
    L.append(f"| 其他 COPD 诊断（排除） | {cohort['other']} |")
    L.append("")
    L.append("排除的诊断分布：")
    L.append("")
    for d, c in oth_dist.items():
        L.append(f"- {d}：{c} 例")
    L.append("")
    L.append("## 3. 消融对比（分层 5 折 CV LR）\n")
    L.append("| 特征集 | 特征数 | AUC | Acc | Sens | Spec |")
    L.append("|---|---|---|---|---|---|")
    for r in results:
        L.append(f"| {r['tag']} | {r['n_feat']} | {r['auc']:.3f} ± {r['auc_std']:.3f} | "
                 f"{r['acc']:.3f} | {r['sens']:.3f} | {r['spec']:.3f} |")
    L.append("")
    L.append("## 4. 单变量判别力 Top（全特征集）\n")
    full = [r for r in results if r["tag"] == "rad+aq"][0]
    L.append("| 特征 | AUC | Cohen's d | p(MWU) |")
    L.append("|---|---|---|---|")
    for _, row in full["uni"].sort_values("auc_dev", ascending=False).head(25).iterrows():
        L.append(f"| {row['feature']} | {row['auc_univ']:.3f} | {row['cohens_d']:+.2f} | {row['p_mwu']:.2g} |")
    L.append("")
    L.append("## 5. AirQuant（含 blur/T-D 新特征）单变量\n")
    aq_uni = full["uni"][full["uni"]["feature"].str.startswith("aq_")]
    L.append("| 特征 | AUC | Cohen's d | p(MWU) |")
    L.append("|---|---|---|---|")
    for _, row in aq_uni.sort_values("auc_dev", ascending=False).head(20).iterrows():
        L.append(f"| {row['feature']} | {row['auc_univ']:.3f} | {row['cohens_d']:+.2f} | {row['p_mwu']:.2g} |")
    L.append("")
    L.append("## 6. 图表\n")
    for fn, cap in [
        ("fig_roc_copd_acute_2026_05.png", "图 1. 两组消融 5 折 CV 平均 ROC"),
        ("fig_univariate_auc_copd_acute_2026_05.png", "图 2. 单变量 AUC Top 20（红=正向，蓝=负向）"),
        ("fig_univariate_auc_airquant_copd_acute_2026_05.png", "图 3. AirQuant 特征单变量 AUC Top 15"),
        ("fig_boxplot_top8_copd_acute_2026_05.png", "图 4. Top 8 特征按 稳定/加重 箱线图"),
        ("fig_consistency_radiomics_copd_acute_2026_05.png", "图 5. radiomics 一致性森林图"),
        ("fig_consistency_airquant_copd_acute_2026_05.png", "图 6. AirQuant 一致性森林图"),
    ]:
        L.append(f"![{cap}](figs/{fn})\n")
        L.append(f"*{cap}*\n")
    L.append("## 7. 结论\n")
    best = max(results, key=lambda r: r["auc"])
    L.append(f"- 纯 COPD 人群内，影像特征对「急性加重 vs 单纯稳定」的最佳判别 AUC = **{best['auc']:.3f} ± {best['auc_std']:.3f}**（{best['tag']}）。")
    L.append("- 目的：寻找容易导致急性加重的结构特征；AUC 越高说明影像结构特征对急性加重易感性越有可分辨信号。")
    L.append("- 注：「慢性支气管炎急性发作」(18) 等未含「急性加重」字样的 COPD 诊断已从建模中排除，可依临床需要并入 Label=1。")
    md = "\n".join(L) + "\n"
    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write(md)

    def b64img(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(md_to_html(md, b64img))
    print(f"[report] {MD_OUT}\n[report] {HTML_OUT}")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    labels, n_excluded = build_copd_labels()

    df = load_and_join(Args())
    if LABEL_COL not in df.columns:
        sys.exit(f"[err] 缺少 {LABEL_COL}")
    df = df[df[LABEL_COL].notna()].copy()
    y = df[LABEL_COL].values.astype(int)
    n = len(y); n_pos = int(y.sum())
    print(f"[model] 样本 {n}, AECOPD {n_pos} / 稳定 {n - n_pos}")

    feats_all, _ = select_features(df, LABEL_COL)
    feats_rad = [c for c in feats_all if not c.startswith("aq_")]

    tags = [("rad", feats_rad), ("rad+aq", feats_all)]
    results = []
    for tag, fset in tags:
        r = run_ablation(df, fset, tag, y)
        r["_y"] = y
        r["_Xv"] = df[fset].apply(pd.to_numeric, errors="coerce").fillna(
            df[fset].apply(pd.to_numeric, errors="coerce").median()).values.astype(np.float64)
        results.append(r)

    plot_roc(results, os.path.join(FIGDIR, "fig_roc_copd_acute_2026_05.png"))

    # 单变量/箱线/一致性 基于全特征集
    full = results[-1]
    uni = full["uni"]
    top = uni.sort_values("auc_dev", ascending=False).head(20)
    plt.figure(figsize=(9, 0.42 * len(top) + 1.5))
    colors = ["#d62728" if a >= 0.5 else "#1f77b4" for a in top["auc_univ"]]
    plt.barh(range(len(top)), top["auc_univ"], color=colors)
    plt.axvline(0.5, color="k", ls="--", lw=1)
    plt.yticks(range(len(top)), [shorten(f, 46) for f in top["feature"]], fontsize=8)
    plt.gca().invert_yaxis(); plt.xlabel("Univariate AUC"); plt.xlim(0, 1)
    plt.title("Pure COPD AECOPD vs Stable Top 20 by |AUC-0.5|")
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "fig_univariate_auc_copd_acute_2026_05.png"), dpi=150); plt.close()

    aq_top = uni[uni["feature"].str.startswith("aq_")].sort_values("auc_dev", ascending=False).head(15)
    if len(aq_top):
        colors = ["#d62728" if a >= 0.5 else "#1f77b4" for a in aq_top["auc_univ"]]
        plt.figure(figsize=(8, 0.42 * len(aq_top) + 1.5))
        plt.barh(range(len(aq_top)), aq_top["auc_univ"], color=colors)
        plt.axvline(0.5, color="k", ls="--", lw=1)
        plt.yticks(range(len(aq_top)), [shorten(f, 46) for f in aq_top["feature"]], fontsize=8)
        plt.gca().invert_yaxis(); plt.xlabel("Univariate AUC"); plt.xlim(0, 1)
        plt.title("Pure COPD AirQuant features Top 15")
        plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "fig_univariate_auc_airquant_copd_acute_2026_05.png"), dpi=150); plt.close()

    top8 = uni.sort_values("auc_dev", ascending=False).head(8)["feature"].tolist()
    ncol = 4; nrow = int(np.ceil(len(top8) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, top8):
        x = pd.to_numeric(df[c], errors="coerce")
        ax.boxplot([x[y == 0].dropna(), x[y == 1].dropna()], tick_labels=["Stable", "Acute"], widths=0.6)
        try:
            a = roc_auc_score(y, np.nan_to_num(x.values, nan=np.nanmedian(x.values)))
            ax.set_title(f"AUC={a:.3f}\n{shorten(c, 30)}", fontsize=8)
        except ValueError:
            ax.set_title(shorten(c, 30), fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(top8):]:
        ax.axis("off")
    fig.suptitle("Pure COPD Top 8 features by class", fontsize=12)
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "fig_boxplot_top8_copd_acute_2026_05.png"), dpi=150); plt.close()

    # 一致性
    def analyze(fl):
        out = []
        for c in fl:
            x = pd.to_numeric(df[c], errors="coerce").values
            if np.isnan(x).mean() > 0.3:
                continue
            try:
                point = roc_auc_score(y, np.nan_to_num(x, nan=np.nanmedian(x)))
            except ValueError:
                continue
            b = bootstrap_auc(x, y, n_iter=N_BOOT, seed=SEED)
            if len(b) < 50:
                continue
            lo, hi = np.percentile(b, [2.5, 97.5])
            stab = float(np.mean((b > 0.5) if point >= 0.5 else (b < 0.5)))
            out.append((c, point, lo, hi, stab, np.sign(point - 0.5)))
        out.sort(key=lambda t: -abs(t[1] - 0.5))
        return out[:TOP_N]

    rad_top = analyze(feats_rad)
    aq_topc = analyze([c for c in feats_all if c.startswith("aq_")])
    forest_plot(rad_top, "Pure COPD Consistency: radiomics features",
                os.path.join(FIGDIR, "fig_consistency_radiomics_copd_acute_2026_05.png"))
    forest_plot(aq_topc, "Pure COPD Consistency: AirQuant features",
                os.path.join(FIGDIR, "fig_consistency_airquant_copd_acute_2026_05.png"))

    oth = labels[~labels["labeled"]]
    oth_dist = oth["main_diagnosis"].value_counts().to_dict()
    cohort = {"copd": len(labels), "excluded": n_excluded,
              "pos_all": int(labels["is_acute"].sum()),
              "neg_all": int(labels["is_stable"].sum()),
              "other": int((~labels["labeled"]).sum()),
              "model_n": n, "pos": n_pos, "neg": n - n_pos}
    write_report(results, cohort, oth_dist)
    return 0


if __name__ == "__main__":
    sys.exit(main())
