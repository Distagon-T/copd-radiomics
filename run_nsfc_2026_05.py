#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_nsfc_2026_05.py
===================
泛气道疾病 "加重 vs 稳定"（NSFC_AE_Label）完整流水线：

  1) 从 info-2026-05.csv 的「主要诊断」按急性事件关键词池生成 NSFC_AE_Label
     （COPD 急性加重/下呼吸道感染/合并感染/肺部感染 + 支扩合并感染/咯血 -> 1；其余 -> 0）
  2) 若 MATLAB 新版特征已产出（E:\\DICOM\\2026-05-Airway_features\\airway_features_all.csv），
     先并入 airquant_2026_05_aggregated.csv（新增 blur_*/TD_* 等 aq_ 列）
  3) 融合 pyRadiomics + AirQuant + NSFC label：分层 5 折 LR CV + 单变量 + bootstrap 一致性
  4) 输出图 (figs/fig_*_nsfc_2026_05.png) 与报告 report_nsfc_2026_05.md / .html

用法：
  python run_nsfc_2026_05.py
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

SEG = r"E:\DICOM\2026-05-seg"
INFO_CSV = r"E:\DICOM\2026-05\info-2026-05.csv"
RAD = os.path.join(SEG, "radiomics_2026_05_features.csv")
AQ = os.path.join(SEG, "airquant_2026_05_aggregated.csv")
LAB_NSFC = os.path.join(SEG, "labels_nsfc_2026_05.csv")
MATLAB_FEATS = r"E:\DICOM\2026-05-Airway_features\airway_features_all.csv"
FIGDIR = os.path.join(SEG, "figs")
MD_OUT = os.path.join(SEG, "report_nsfc_2026_05.md")
HTML_OUT = os.path.join(SEG, "report_nsfc_2026_05.html")
LOG_OUT = os.path.join(SEG, "fusion_nsfc_2026_05.log")
TAG = "nsfc"
LABEL_COL = "NSFC_AE_Label"
SEED = 42
N_BOOT = 200
TOP_N = 15

ACUTE_KEYWORDS = ["急性加重", "急性发作", "合并感染", "肺部感染", "下呼吸道感染", "咯血"]


def generate_nsfc_label(diagnosis):
    """慢性气道疾病急性事件期 = 1；稳定期 = 0。"""
    diag = str(diagnosis)
    return 1 if any(kw in diag for kw in ACUTE_KEYWORDS) else 0


def shorten(name, n=44):
    return name if len(name) <= n else name[: n - 3] + "..."


# =========================================================================
# 1. 生成 NSFC 标签（全部临床行；保留 Patient_ID 便于与 AirQuant 对齐）
# =========================================================================
def build_nsfc_labels():
    info = pd.read_csv(INFO_CSV, encoding="utf-8")
    diag_col = "主要诊断" if "主要诊断" in info.columns else info.columns[1]
    id_col = "患者id" if "患者id" in info.columns else info.columns[0]
    info["_pid"] = norm_id(info[id_col].astype(str))

    rad = pd.read_csv(RAD, usecols=["Patient_ID", "PatientID"])
    rad["_pid"] = norm_id(rad["PatientID"].astype(str))
    pid2folder = (rad.drop_duplicates("_pid")
                  .set_index("_pid")["Patient_ID"].to_dict())

    info["Patient_ID"] = info["_pid"].map(pid2folder)
    info["NSFC_AE_Label"] = info[diag_col].fillna("").astype(str).apply(generate_nsfc_label)

    out = info.rename(columns={"_pid": "patient_id", diag_col: "main_diagnosis"})
    out = out[["patient_id", "Patient_ID", "main_diagnosis", "NSFC_AE_Label"]]
    out.to_csv(LAB_NSFC, index=False, encoding="utf-8-sig")

    n_all = len(out)
    dist_all = out["NSFC_AE_Label"].value_counts().to_dict()
    print(f"[labels] 全部临床行 {n_all} | NSFC 分布(1加重/0稳定) = {dist_all.get(1,0)}/{dist_all.get(0,0)}")
    return out


# =========================================================================
# 2. 若 MATLAB 新版特征已产出 -> 并入 airquant 聚合表
# =========================================================================
def maybe_merge_matlab_features():
    if not os.path.exists(MATLAB_FEATS):
        print("[warn] 未找到 MATLAB 新版特征 " + MATLAB_FEATS)
        print("       (blur_*/TD_* 等尚未并入；当前用基础 aq_* 特征。")
        print("       需先在 MATLAB 跑 compute_airway_features.m 后重跑本脚本)")
        return False
    sys.argv = ["merge_airway_features_2026_05.py",
                "--matlab-feats", MATLAB_FEATS,
                "--airquant", AQ, "--out", AQ]
    import merge_airway_features_2026_05 as mrg
    mrg.main()
    return True


# =========================================================================
# 3. 建模：融合 LR CV + 单变量 + 一致性
# =========================================================================
class Args:
    radiomics = RAD
    airquant = AQ
    labels = LAB_NSFC


def run_model():
    df = load_and_join(Args())
    if LABEL_COL not in df.columns:
        sys.exit(f"[err] 标签列 {LABEL_COL} 不在合并表中")
    y = df[LABEL_COL].values
    n_pos = int(y.sum())
    n = len(y)
    print(f"[model] 样本 {n}, 阳性(加重) {n_pos} ({n_pos/n:.1%})")
    if n_pos < 2:
        sys.exit("阳性数不足，无法建模")

    feats, dropped = select_features(df, LABEL_COL)
    n_aq = sum(1 for c in feats if c.startswith("aq_"))
    aq_in_df = [c for c in df.columns if c.startswith("aq_")]
    aq_kept = [c for c in feats if c.startswith("aq_")]
    # MATLAB 新版特征（blur/T-D/wall_hu/pca）：基础 aq_* 名称不含这些关键词
    new_kept = [c for c in aq_kept
                if ("blur" in c or c.startswith("aq_TD_") or "wall_hu" in c
                    or c.startswith("aq_pca"))]
    print(f"[model] 入选特征 {len(feats)} (aq_* 表内 {len(aq_in_df)} / 保留 {n_aq}，"
          f"其中 MATLAB 新特征 {len(new_kept)} 个参与)，剔除 {len(dropped)}")

    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(
        df[feats].apply(pd.to_numeric, errors="coerce").median())
    Xv = X.values.astype(np.float64)

    # ---- 单变量 ----
    uni = univariate_summary(df, feats, y, top=50)
    uni.to_csv(os.path.join(SEG, f"fusion_{TAG}_univariate_top.csv"),
               index=False, encoding="utf-8-sig")

    # ---- 分层 5 折 CV LR ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs, accs, sens, specs, coefs = [], [], [], [], []
    scaler = StandardScaler()
    log_lines = [f"label={LABEL_COL} n={n} pos={n_pos} feats={len(feats)}"]
    for fold, (tr, te) in enumerate(skf.split(Xv, y)):
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
        line = (f"fold{fold}: AUC={aucs[-1]:.3f} Acc={accs[-1]:.3f} "
                f"Sens={sens[-1]:.3f} Spec={specs[-1]:.3f} "
                f"(train {int(y[tr].sum())} pos / test {int(y[te].sum())} pos)")
        print("  " + line); log_lines.append(line)
    mean_line = (f"平均: AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f} "
                 f"Acc={np.mean(accs):.3f} Sens={np.mean(sens):.3f} Spec={np.mean(specs):.3f}")
    print(mean_line); log_lines.append(mean_line)
    with open(LOG_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    # ---- LR 系数 ----
    imp = pd.DataFrame({"feature": feats,
                        "coef_mean": np.mean(coefs, axis=0),
                        "coef_std": np.std(coefs, axis=0)})
    imp["abs"] = imp["coef_mean"].abs()
    imp = imp.sort_values("abs", ascending=False)
    imp.to_csv(os.path.join(SEG, f"fusion_{TAG}_lr_coefficients.csv"),
               index=False, encoding="utf-8-sig")

    # ---- ROC 图（5 折平均）----
    tprs, base_fpr = [], np.linspace(0, 1, 101)
    for tr, te in skf.split(Xv, y):
        Xtr = scaler.fit_transform(Xv[tr]); Xte = scaler.transform(Xv[te])
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                 class_weight="balanced", max_iter=1000, random_state=SEED)
        clf.fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(y[te], proba)
        tprs.append(np.interp(base_fpr, fpr, tpr)); tprs[-1][0] = 0.0
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    plt.figure(figsize=(6.5, 6.5))
    plt.plot(base_fpr, mean_tpr, "b-", lw=2,
             label=f"Mean ROC (AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f})")
    plt.fill_between(base_fpr, np.clip(mean_tpr - std_tpr, 0, 1),
                     np.clip(mean_tpr + std_tpr, 0, 1), color="b", alpha=0.15, label="±1 std")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"NSFC acute vs stable (n={n}, pos={n_pos})")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "fig_roc_nsfc_2026_05.png"), dpi=150); plt.close()

    # ---- 单变量 AUC 柱状图（整体 Top20 + AirQuant Top15）----
    uni["auc_dev"] = (uni["auc_univ"] - 0.5).abs()
    for col, out_name, ntop, title in [
        ("all", "fig_univariate_auc_nsfc_2026_05.png", 20, "NSFC Top 20 by |AUC-0.5|"),
        ("aq", "fig_univariate_auc_airquant_nsfc_2026_05.png", 15, "NSFC AirQuant features Top 15"),
    ]:
        sub = uni if col == "all" else uni[uni["feature"].str.startswith("aq_")]
        top = sub.sort_values("auc_dev", ascending=False).head(ntop)
        if len(top) == 0:
            continue
        colors = ["#d62728" if a >= 0.5 else "#1f77b4" for a in top["auc_univ"]]
        plt.figure(figsize=(9, 0.42 * len(top) + 1.5))
        plt.barh(range(len(top)), top["auc_univ"], color=colors)
        plt.axvline(0.5, color="k", ls="--", lw=1)
        plt.yticks(range(len(top)), [shorten(f, 46) for f in top["feature"]], fontsize=8)
        plt.gca().invert_yaxis(); plt.xlabel("Univariate AUC"); plt.xlim(0, 1)
        plt.title(title)
        plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, out_name), dpi=150); plt.close()

    # ---- Top8 箱线图 ----
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
    fig.suptitle("NSFC Top 8 features by class (Stable vs Acute)", fontsize=12)
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "fig_boxplot_top8_nsfc_2026_05.png"), dpi=150); plt.close()

    # ---- 一致性（bootstrap）：radiomics 与 AirQuant 分开 ----
    def point_auc(fl):
        out = []
        for c in fl:
            x = pd.to_numeric(df[c], errors="coerce").values
            if np.isnan(x).mean() > 0.3:
                continue
            try:
                a = roc_auc_score(y, np.nan_to_num(x, nan=np.nanmedian(x)))
            except ValueError:
                continue
            out.append((c, a))
        return out

    def analyze(fl):
        pts = sorted(point_auc(fl), key=lambda t: -abs(t[1] - 0.5))
        out = []
        for c, point in pts[:TOP_N]:
            x = pd.to_numeric(df[c], errors="coerce").values
            b = bootstrap_auc(x, y, n_iter=N_BOOT, seed=SEED)
            if len(b) < 50:
                continue
            lo, hi = np.percentile(b, [2.5, 97.5])
            stab = float(np.mean((b > 0.5) if point >= 0.5 else (b < 0.5)))
            out.append((c, point, lo, hi, stab, np.sign(point - 0.5)))
        return out

    rad_top = analyze([c for c in feats if not c.startswith("aq_")])
    aq_top = analyze([c for c in feats if c.startswith("aq_")])
    forest_plot(rad_top, "NSFC Consistency: significant radiomics features",
                os.path.join(FIGDIR, "fig_consistency_radiomics_nsfc_2026_05.png"))
    forest_plot(aq_top, "NSFC Consistency: AirQuant features",
                os.path.join(FIGDIR, "fig_consistency_airquant_nsfc_2026_05.png"))

    return {"n": n, "n_pos": n_pos, "n_feat": len(feats), "n_aq": n_aq,
            "n_aq_in_df": len(aq_in_df), "n_new_kept": len(new_kept),
            "auc": np.mean(aucs), "auc_std": np.std(aucs),
            "acc": np.mean(accs), "sens": np.mean(sens), "spec": np.mean(specs),
            "folds": list(zip(range(5), aucs, accs, sens, specs)),
            "uni": uni, "imp": imp, "rad_top": rad_top, "aq_top": aq_top}


# =========================================================================
# 4. 报告 md + html
# =========================================================================
def md_to_html(md_text, b64img):
    lines = md_text.splitlines()
    html = ["<h1>泛气道疾病 NSFC 加重 vs 稳定（Radiomics + AirQuant 融合）</h1>"]
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


def write_report(r, nsfc_all, has_matlab):
    L = []
    L.append("# 泛气道疾病 NSFC 急性加重 vs 稳定（Radiomics + AirQuant 融合）\n")
    L.append(f"> 报告时间：2026-08-22　|　临床样本 n={nsfc_all['n_all']}（加重 {nsfc_all['pos_all']} / 稳定 {nsfc_all['neg_all']}）　|　建模样本 n={r['n']}（加重 {r['n_pos']} / 稳定 {r['n'] - r['n_pos']}）\n")
    L.append("## 1. 数据与方法\n")
    L.append("| 组件 | 说明 |")
    L.append("|---|---|")
    L.append("| 影像特征 | pyRadiomics lite（shape+firstorder+自定义肺/气道/血管/主动脉指标） |")
    L.append(f"| AirQuant | branch 级指标聚合为患者级（表内 {r['n_aq_in_df']} 个 aq_*，建模保留 {r['n_aq']} 个）；"
              f"MATLAB FWHM 边界模糊/T-D 新特征参与建模：{r['n_new_kept']} 个（{'已产出并入' if has_matlab else '待跑 MATLAB 后并入'}） |")
    L.append("| Label | 主要诊断含「急性加重/急性发作/合并感染/肺部感染/下呼吸道感染/咯血」→1（加重期），否则 0（稳定期） |")
    L.append("| 模型 | Logistic Regression（L2, C=1.0, class_weight=balanced, liblinear），StandardScaler，分层 5 折 CV |")
    L.append(f"| 特征 | {r['n_feat']} 个入选（剔除高缺失/零方差），中位数填补 |")
    L.append("")
    L.append("## 2. 分类性能（5 折 CV）\n")
    L.append(f"**平均 AUC = {r['auc']:.3f} ± {r['auc_std']:.3f}**，Acc={r['acc']:.3f}，Sens={r['sens']:.3f}，Spec={r['spec']:.3f}\n")
    L.append("| Fold | AUC | Acc | Sens | Spec |")
    L.append("|---|---|---|---|---|")
    for fno, auc, acc, sn, sp in r["folds"]:
        L.append(f"| {fno} | {auc:.3f} | {acc:.3f} | {sn:.3f} | {sp:.3f} |")
    L.append("")
    L.append("## 3. 单变量判别力 Top 特征\n")
    L.append("| 特征 | AUC | Cohen's d | p(MWU) |")
    L.append("|---|---|---|---|")
    for _, row in r["uni"].sort_values("auc_dev", ascending=False).head(20).iterrows():
        L.append(f"| {row['feature']} | {row['auc_univ']:.3f} | {row['cohens_d']:+.2f} | {row['p_mwu']:.2g} |")
    L.append("")
    L.append("## 4. Logistic 回归系数 Top（相关性，非因果）\n")
    L.append("| 特征 | 平均系数 ± SD |")
    L.append("|---|---|")
    for _, row in r["imp"].head(15).iterrows():
        L.append(f"| {row['feature']} | {row['coef_mean']:+.3f} ± {row['coef_std']:.3f} |")
    L.append("")
    L.append("## 5. 一致性分析（bootstrap 200 次，单变量 AUC 均值 ± 95%CI + 同向稳定率）\n")
    L.append("### 5.1 radiomics 显著特征\n")
    L.append("| 特征 | AUC | 95%CI | 同向稳定 |")
    L.append("|---|---|---|---|")
    for t in r["rad_top"]:
        L.append(f"| {t[0]} | {t[1]:.3f} | [{t[2]:.3f}, {t[3]:.3f}] | {t[4]:.0%} |")
    L.append("\n### 5.2 AirQuant 特征\n")
    L.append("| 特征 | AUC | 95%CI | 同向稳定 |")
    L.append("|---|---|---|---|")
    for t in r["aq_top"]:
        L.append(f"| {t[0]} | {t[1]:.3f} | [{t[2]:.3f}, {t[3]:.3f}] | {t[4]:.0%} |")
    L.append("")
    L.append("## 6. 图表\n")
    for fn, cap in [
        ("fig_roc_nsfc_2026_05.png", "图 1. 5 折 CV 平均 ROC"),
        ("fig_univariate_auc_nsfc_2026_05.png", "图 2. 单变量 AUC Top 20（红=正向，蓝=负向）"),
        ("fig_univariate_auc_airquant_nsfc_2026_05.png", "图 3. AirQuant 特征单变量 AUC Top 15"),
        ("fig_boxplot_top8_nsfc_2026_05.png", "图 4. Top 8 特征按 稳定/加重 分组箱线图"),
        ("fig_consistency_radiomics_nsfc_2026_05.png", "图 5. radiomics 显著特征 bootstrap 一致性森林图"),
        ("fig_consistency_airquant_nsfc_2026_05.png", "图 6. AirQuant 特征 bootstrap 一致性森林图"),
    ]:
        L.append(f"![{cap}](figs/{fn})\n")
        L.append(f"*{cap}*\n")
    L.append("## 7. 结论与局限\n")
    L.append(f"- 泛气道「加重 vs 稳定」判别在该队列 5 折 CV AUC = **{r['auc']:.3f} ± {r['auc_std']:.3f}**（n={r['n']}）。")
    L.append("- 显著且一致的特征见第 3/5 节；需关注是否由 CT 采集/重建差异驱动。")
    L.append(f"- 局限：① 单中心回顾性，Label 由主要诊断文本关键词生成（阳性占比高，{r['n_pos']}/{r['n']}）；② 本次 AirQuant 参与建模 {r['n_aq']} 个（表内 {r['n_aq_in_df']} 个），其中 MATLAB 新特征 {r['n_new_kept']} 个；③ 若需 FWHM 边界模糊/T-D 变化特征全部参与，需先对全队列跑 MATLAB compute_airway_features.m 再重跑本脚本。")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-merge", action="store_true", help="跳过 MATLAB 特征并入")
    args = ap.parse_args()

    # 1. 标签
    nsfc_all = build_nsfc_labels()
    dist_all = nsfc_all["NSFC_AE_Label"].value_counts().to_dict()
    # 2. MATLAB 特征并入（若存在）
    has_matlab = False
    if not args.skip_merge:
        has_matlab = maybe_merge_matlab_features()
    # 3. 建模
    r = run_model()
    r["n_aq"] = int(r["n_aq"])
    # 4. 报告
    write_report(r, {"n_all": len(nsfc_all),
                     "pos_all": int(dist_all.get(1, 0)),
                     "neg_all": int(dist_all.get(0, 0))}, has_matlab)
    return 0


if __name__ == "__main__":
    sys.exit(main())
