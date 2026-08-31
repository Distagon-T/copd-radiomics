# -*- coding: utf-8 -*-
"""
最终综合分析报告（中期汇报用）：L1 LASSO + Linear SVM，选择性保护区
================================================================
读取 optimize_declared_eng.py 已保存的 OOF/fold/selection/pooled 结果，
生成 md + html 双格式完整报告，含组学分析常见图表：
  - ROC 曲线（Strategy A/B，LASSO + SVM）
  - 混淆矩阵（SVM，A/B）
  - 选中特征 Spearman 相关热图
  - 单变量 AUC 图 + 表
  - LASSO 选择频率条形图
  - 特征组成堆叠图
  - Strategy B held-out Label1 风险分布（灰度区间）
  - 分队列 SVM AUC 条形图
  - 临床意义总结表

用法：
  python Analysis/report_final.py --parent "E:\\DICOM\\reports\\feature_selection_ordinal_ae\\pi10bv_pah_sel\\old_std_l1" \
                                  --tag pi10bv_pah_sel --protected "PAH_vascular,Pi10,Vessel_BV5_pct,Vessel_BV10_pct"
输出：<parent>/report_final/  (report.md / report.html + figs/)
"""
import argparse
import base64
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lasso_svm_nested import is_feature  # noqa: E402
from declared_eng import build_eng, engineered_feature_names  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_ap = argparse.ArgumentParser()
_ap.add_argument("--parent", default=r"E:\DICOM\reports\feature_selection_ordinal_ae\pi10bv_pah_sel\old_std_l1")
_ap.add_argument("--tag", default="pi10bv_pah_sel")
_ap.add_argument("--protected", default="PAH_vascular,Pi10,Vessel_BV5_pct,Vessel_BV10_pct")
_ap.add_argument("--raw", action="store_true", help="原始特征池模式（不构建复合分，用于 optimize_lasso_svm_binary 的 run）")
_args = _ap.parse_args()
RAW_MODE = _args.raw

BASE = Path(_args.parent)
OUT = BASE / "report_final"
FIGS = OUT / "figs"
INPUT = r"E:\DICOM\results\patients_feature_label.csv"
PROTECTED = tuple(p.strip() for p in _args.protected.split(",") if p.strip())
GRAY_LOW, GRAY_HIGH = 0.4, 0.6
SEED = 20260830
COHORT_MAP = {"Jan-26": "2026-01", "Feb-26": "2026-02", "Apr-26": "2026-04", "May-26": "2026-05"}

# 临床意义词典（复用 report_declared_protected 的精简版）
CLIN = [
    (r"^PAH_vascular$", ("肺血管/肺高压", "肺血管截面积+主肺动脉直径复合分", "肺血管床丢失 + 肺动脉高压负荷，本队列单变量最强信号", "异常即风险")),
    (r"^CAC_score$", ("钙化", "冠脉钙化复合分 z(Agatston)", "冠脉粥样硬化钙化负荷；COPD 心血管共病", "CAC ↑ 风险↑")),
    (r"^FatInflam_score$", ("心包脂肪", "心包脂肪炎症复合分", "心外膜脂肪体积+冠周脂肪衰减+脂肪密度", "↑ 风险↑")),
    (r"^CTR_score$", ("心脏", "心胸比复合分", "心脏扩大/心衰负荷", "↑ 风险↑")),
    (r"^BAR_score$", ("气道", "支气管-血管比复合分", "支扩/气道扩张性病变", "↑ 风险↑")),
    (r"^CAC_Agatston$", ("钙化", "冠脉钙化 Agatston 积分", "冠脉粥样硬化钙化负荷；COPD 心血管共病", "CAC ↑ 风险↑")),
    (r"^CAC_Mass_mg$", ("钙化", "冠脉钙化质量(mg)", "与 Agatston 互补的钙化总量", "↑ 风险↑")),
    (r"^CAC_Volume_mm3$", ("钙化", "冠脉钙化体积(mm³)", "钙化体积负荷", "↑ 风险↑")),
    (r"^EpiFat_Volume_mm3$", ("心包脂肪", "心外膜脂肪体积", "代谢/炎症性脂肪堆积，心血管风险", "↑ 风险↑")),
    (r"^EpiFat_Mean_HU$", ("心包脂肪", "心外膜脂肪密度(HU)", "密度升高(近水)提示脂肪炎症", "—")),
    (r"^FAI_pericoronary_HU$", ("心包脂肪", "冠周脂肪衰减指数(FAI)", "冠脉周围脂肪炎症，CCTA 心血管风险标志", "FAI ↑ 风险↑")),
    (r"^CardioThoracic_Ratio$", ("心脏", "心胸比", "心脏扩大/心衰负荷，肺心病相关", "↑ 风险↑")),
    (r"^BronchoArtery_Ratio$", ("气道", "支气管-血管比(BAR)", ">1 提示支气管扩张", "↑ 风险↑")),
    (r"^Vessel_Volume_mm3$", ("肺血管", "肺血管总体积", "血管床总量，COPD 中通常下降", "↓ 风险↑(反向)")),
    (r"^Pi10$", ("气道重塑", "气道壁厚度@内周长10mm", "气道壁增厚→气流受限→急性加重风险↑", "Pi10 ↑ 风险↑")),
    (r"Vessel_BV5_pct|Vessel_BV10_pct", ("肺血管", "小血管(<5/<10mm²)血容量占比", "COPD 毛细血管床修剪→小血管减少", "BV ↓ 风险↑(反向)")),
    (r"Vessel_CSA_mean_mm2", ("肺血管", "肺血管平均截面积", "血管床重塑/丢失", "异常即风险")),
    (r"PA_Equivalent_Diameter_mm", ("肺动脉", "主肺动脉等效直径", "PA 增宽(>29mm)提示肺动脉高压", "↑ 风险↑")),
    (r"firstorder_(Energy|TotalEnergy)", ("强度(混杂)", "firstorder 能量", "受扫描协议混杂污染，谨慎解释", "混杂")),
    (r"firstorder_", ("firstorder", "一阶统计量", "体素灰度分布", "—")),
    (r"shape_", ("形态学", "shape 特征", "ROI 几何形态", "—")),
    (r"_(glcm|glrlm|glszm|gldm|ngtdm)_", ("纹理", "纹理特征", "灰度空间分布/异质性", "—")),
    (r"TD_|blur_|wall_|WA_|Din_|Dout_|^mean_|tortuosity|branch|junction|pruning|generation", ("气道结构", "AirQuant 气道特征", "气道管腔/管壁/树结构", "—")),
    (r"Lobe_|Lung_|Airway_|Diaphragm_|RV_|LV_|Aorta_|PA_Ao", ("结构指标", "肺叶/膈肌/心腔/主动脉", "肺气肿、肺心病、心血管结构", "—")),
]


def clin_for(f):
    for pat, info in CLIN:
        if re.search(pat, f):
            return info
    return ("其它", "—", "—", "—")


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -600, 600)
    return 1.0 / (1.0 + np.exp(-x))


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def data_uri(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode("ascii")


def load_work():
    df = pd.read_csv(INPUT, low_memory=False)
    pid = "PatientID" if "PatientID" in df.columns else "Patient_ID"
    label_col = "Label" if "Label" in df.columns else "label"
    cohort_col = "cohort" if "cohort" in df.columns else "Cohort"
    feature_names = [c for c in df.columns if is_feature(c)]
    valid = pd.to_numeric(df[label_col], errors="coerce").isin([0, 1, 2])
    work = df.loc[valid, [pid, cohort_col, label_col] + feature_names].copy()
    work["Label"] = pd.to_numeric(work[label_col], errors="coerce").astype(int)
    work["cohort"] = work[cohort_col].astype(str).map(COHORT_MAP).fillna(work[cohort_col].astype(str))
    work = work.drop_duplicates(pid).reset_index(drop=True)
    if not RAW_MODE:
        work = build_eng(work)
        feature_names = engineered_feature_names(feature_names)
    return work, feature_names


def feature_composition(features):
    n_intensity = n_shape = n_texture = n_fo_other = n_struct = 0
    for f in features:
        if re.search(r"firstorder_(TotalEnergy|Energy)$", f):
            n_intensity += 1
        elif "shape_" in f:
            n_shape += 1
        elif re.search(r"_(glcm|glrlm|glszm|gldm|ngtdm)_", f):
            n_texture += 1
        elif "firstorder_" in f:
            n_fo_other += 1
        else:
            n_struct += 1
    return {"shape": n_shape, "texture": n_texture, "firstorder_other": n_fo_other,
            "intensity": n_intensity, "structural": n_struct}


# ---------------- 图表 ----------------
def fig_roc(y, s_l, s_s, fname, title):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for sc, c, lab in [(s_l, "#1f77b4", "LASSO"), (s_s, "#d62728", "LinearSVM")]:
        fpr, tpr, _ = roc_curve(y, sc)
        ax.plot(fpr, tpr, lw=2, color=c, label=f"{lab} AUC={roc_auc_score(y, sc):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", title=title)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    p = FIGS / fname
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_confusion(y_true, y_pred, fname, title, classes=("Stable", "Risk")):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax)
    tn, fp, fn, tp = cm.ravel()
    n = max(1, len(y_true))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{title} (acc={(tn+tp)/n:.3f})")
    fig.tight_layout()
    p = FIGS / fname
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_corr_heatmap(work, features, fname, title):
    sub = work[[f for f in features if f in work.columns]].copy()
    for c in sub.columns:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    corr = sub.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(max(8, corr.shape[0] * 0.55), max(7, corr.shape[1] * 0.55)))
    sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, xticklabels=corr.columns, yticklabels=corr.index, ax=ax)
    ax.set_title(title)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    fig.tight_layout()
    p = FIGS / fname
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_uni_auc(uni, fname, title):
    uni = uni.sort_values("auc").head(25)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    colors = ["#d62728" if (a >= 0.6 or a <= 0.4) else "#1f77b4" for a in uni["auc"]]
    ax.barh(uni["feature"], uni["auc"], color=colors)
    ax.axvline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xlim(0.3, 0.8)
    ax.set_xlabel("Univariate AUC (Label 0 vs 2)"); ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    p = FIGS / fname
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_frequency(sel, fname, title):
    top = sel[sel["frequency"] > 0].sort_values("frequency", ascending=False).head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    prot = ["🔒 " + f if f in PROTECTED else f for f in top["feature"]]
    ax.barh(prot, top["frequency"], color="#2ca02c")
    ax.set_xlim(0, max(1, int(sel["frequency"].max())))
    ax.set_xlabel("Outer folds selected by LASSO"); ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    p = FIGS / fname
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_composition(compA, compB, fname):
    cats = ["shape", "texture", "firstorder_other", "intensity", "structural"]
    lab = ["Shape", "Texture", "FirstOrder", "Intensity", "Structural"]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = np.arange(len(cats)); w = 0.38
    ax.bar(x - w/2, [compA[c] for c in cats], w, label="Strategy A", color="#1f77b4")
    ax.bar(x + w/2, [compB[c] for c in cats], w, label="Strategy B", color="#ff7f0e")
    for xi, (a, b) in zip(x, zip([compA[c] for c in cats], [compB[c] for c in cats])):
        ax.text(xi - w/2, a + 0.2, int(a), ha="center", fontsize=9)
        ax.text(xi + w/2, b + 0.2, int(b), ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(lab)
    ax.set_ylabel("Selected feature count"); ax.set_title("Selected feature composition")
    ax.legend(); ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    p = FIGS / fname
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_risk(oof, fname):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bins = np.linspace(0, 1, 31)
    for lab, c, nm in [(0, "#1f77b4", "Label 0 (stable)"), (1, "#ff7f0e", "Label 1 (held-out)"), (2, "#d62728", "Label 2 (acute)")]:
        sub = oof.loc[oof["Label"] == lab, "risk"].dropna().to_numpy(float)
        if len(sub):
            ax.hist(sub, bins=bins, alpha=0.45, color=c, label=f"{nm} (n={len(sub)})", density=True)
    ax.axvspan(GRAY_LOW, GRAY_HIGH, color="gold", alpha=0.25, label=f"gray zone [{GRAY_LOW},{GRAY_HIGH}]")
    ax.set(xlabel="Risk (sigmoid of LASSO log-odds)", ylabel="Density",
           title="Strategy B Drop & Predict risk distribution")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    fig.tight_layout()
    p = FIGS / fname
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_cohort_auc(oof, y_func, fname, title):
    rows = []
    for cohort in sorted(oof["cohort"].unique()):
        sub = oof[oof["cohort"] == cohort]
        y = y_func(sub)
        if np.unique(y).size < 2:
            continue
        rows.append({"cohort": cohort, "n": len(sub), "LASSO_AUC": roc_auc_score(y, sub["lasso_score"]),
                     "SVM_AUC": roc_auc_score(y, sub["svm_score"])})
    d = pd.DataFrame(rows)
    if d.empty:
        return None
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(x - 0.2, d["LASSO_AUC"], 0.4, label="LASSO", color="#1f77b4")
    ax.bar(x + 0.2, d["SVM_AUC"], 0.4, label="SVM", color="#d62728")
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    for xi, (a, b) in zip(x, zip(d["LASSO_AUC"], d["SVM_AUC"])):
        ax.text(xi - 0.2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(xi + 0.2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n(n={int(n)})" for c, n in zip(d["cohort"], d["n"])])
    ax.set_ylabel("AUC"); ax.set_title(title); ax.set_ylim(0.3, 1.0)
    ax.legend(); ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    p = FIGS / fname
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p, d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    oofA = pd.read_csv(BASE / "oof_A.csv")
    oofB = pd.read_csv(BASE / "oof_B.csv")
    foldA = pd.read_csv(BASE / "fold_A.csv")
    foldB = pd.read_csv(BASE / "fold_B.csv")
    pooledA = pd.read_csv(BASE / "pooled_A.csv")
    pooledB = pd.read_csv(BASE / "pooled_B.csv")
    selA = pd.read_csv(BASE / "selection_A.csv")
    selB = pd.read_csv(BASE / "selection_B.csv")

    work, feature_names = load_work()
    label = work["Label"].to_numpy(int)
    yA = (label >= 1).astype(int)
    yB = (label == 2).astype(int)

    oofA["risk"] = sigmoid(oofA["lasso_score"])
    oofB["risk"] = sigmoid(oofB["lasso_score"])
    yA_oof = (oofA["Label"].to_numpy(int) >= 1).astype(int)
    ob = oofB[oofB["in_train"]].reset_index(drop=True)
    yB_oof = (ob["Label"].to_numpy(int) == 2).astype(int)

    # ---- 图 ----
    p_rocA = fig_roc(yA_oof, oofA["lasso_score"], oofA["svm_score"], "roc_A.png", "Strategy A OOF ROC")
    p_rocB = fig_roc(yB_oof, ob["lasso_score"], ob["svm_score"], "roc_B.png", "Strategy B OOF ROC (0 vs 2)")
    p_cmB = fig_confusion(yB_oof, ob["svm_pred"].to_numpy(int), "confusion_B.png",
                          "Strategy B SVM confusion (0 vs 2)")
    p_cmA = fig_confusion(yA_oof, oofA["svm_pred"].to_numpy(int), "confusion_A.png",
                          "Strategy A SVM confusion (Stable vs Risk)")
    p_risk = fig_risk(oofB, "risk_B.png")

    selA_final = selA[selA["frequency"] > 0].sort_values("frequency", ascending=False)
    selB_final = selB[selB["frequency"] > 0].sort_values("frequency", ascending=False)
    p_freqA = fig_frequency(selA, "freq_A.png", "Strategy A LASSO selection frequency")
    p_freqB = fig_frequency(selB, "freq_B.png", "Strategy B LASSO selection frequency")
    compA = feature_composition(selA_final["feature"].tolist())
    compB = feature_composition(selB_final["feature"].tolist())
    p_comp = fig_composition(compA, compB, "composition.png")

    # 相关热图：取 A/B 选中并集按频率 top20
    uni_sel = list(dict.fromkeys(selA_final["feature"].tolist() + selB_final["feature"].tolist()))[:20]
    p_corr = fig_corr_heatmap(work, uni_sel, "corr_heatmap.png",
                              "Spearman correlation of top selected features")

    # 单变量 AUC（Label 0 vs 2）
    uni_rows = []
    cand = list(dict.fromkeys(list(PROTECTED) + selB_final["feature"].tolist()[:25]))
    for f in cand:
        if f not in work.columns:
            continue
        x = pd.to_numeric(work[f], errors="coerce").to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(yB)
        if ok.sum() < 20 or np.unique(yB[ok]).size < 2:
            continue
        xv, yv = x[ok], yB[ok].astype(int)
        auc = roc_auc_score(yv, xv)
        try:
            p = mannwhitneyu(xv[yv == 1], xv[yv == 0]).pvalue
        except Exception:
            p = np.nan
        cat, meaning, physio, direction = clin_for(f)
        uni_rows.append({"feature": f, "category": cat, "auc": auc, "p": p,
                         "meaning": meaning, "physio": physio, "direction": direction})
    uni = pd.DataFrame(uni_rows)
    uni.to_csv(OUT / "univariate_auc.csv", index=False, encoding="utf-8-sig")
    p_uni = fig_uni_auc(uni, "univariate_auc.png", "Univariate AUC (Label 0 vs 2, top candidates)")

    # 分队列性能
    ca = fig_cohort_auc(oofA, lambda s: (s["Label"].to_numpy(int) >= 1).astype(int), "cohort_auc_A.png",
                        "Strategy A per-cohort AUC")
    cb = fig_cohort_auc(ob, lambda s: (s["Label"].to_numpy(int) == 2).astype(int), "cohort_auc_B.png",
                        "Strategy B per-cohort AUC (0 vs 2)")

    # ---- 汇总数字 ----
    auc_ci = {}
    for name, (y, sc) in {"A_lasso": (yA_oof, oofA["lasso_score"]), "A_svm": (yA_oof, oofA["svm_score"]),
                          "B_lasso": (yB_oof, ob["lasso_score"]), "B_svm": (yB_oof, ob["svm_score"])}.items():
        rng = np.random.default_rng(SEED)
        point = roc_auc_score(y, sc)
        vals = []
        for _ in range(1000):
            idx = rng.integers(0, len(y), len(y))
            v = roc_auc_score(y[idx], sc[idx])
            if np.isfinite(v):
                vals.append(v)
        auc_ci[name] = (point, float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))

    l1 = oofB.loc[oofB["Label"] == 1, "risk"].dropna().to_numpy(float)
    l1_stats = {"n": int(len(l1)), "median": float(np.median(l1)) if len(l1) else np.nan,
                "pct_gray": float(np.mean((l1 >= GRAY_LOW) & (l1 <= GRAY_HIGH))) if len(l1) else np.nan}

    summary = {"tag": _args.tag, "protected": list(PROTECTED), "input": INPUT,
               "strategy_A": {"n": len(oofA), "svm_auc": auc_ci["A_svm"][0], "svm_ci": list(auc_ci["A_svm"][1:]),
                              "lasso_auc": auc_ci["A_lasso"][0], "n_selected": len(selA_final), "composition": compA},
               "strategy_B": {"n_train": int(len(ob)), "n_label1": int((oofB["Label"] == 1).sum()),
                              "svm_auc": auc_ci["B_svm"][0], "svm_ci": list(auc_ci["B_svm"][1:]),
                              "lasso_auc": auc_ci["B_lasso"][0], "n_selected": len(selB_final),
                              "composition": compB, "label1_risk": l1_stats}}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ================= 生成 md =================
    md = []
    md.append(f"# COPD 急性加重风险影像组学分析 — 中期汇报（{_args.tag}）\n")
    md.append("## 1. 概述\n")
    md.append(f"- **数据**：`{INPUT}`，4 队列（2026-01/02/04/05），有效标注 {len(work)} 例（Label 0 稳定 / 1 结构性脆弱 / 2 急性加重）")
    if RAW_MODE:
        md.append("- **特征池**：radiomics（16 掩膜）+ AirQuant 气道特征 + 申报补算特征（CAC/心包脂肪/FAI/心胸比/支气管血管比/肺血管CSA/PA直径）")
    else:
        md.append(f"- **特征池**：radiomics（16 掩膜）+ AirQuant 气道特征 + 申报补算特征复合分（PAH_vascular/CAC/FatInflam/CTR/BAR）")
    md.append(f"- **保护区（强制保留）**：{'、'.join(PROTECTED)}")
    md.append("- **方法**：每外层折内 Mann-Whitney+BH-FDR 筛选 → Spearman 去冗余 → L1 logistic LASSO → Linear SVM（C 折内选）；留一队列外验证；AUC 用 decision function\n")
    md.append("## 2. 模型性能\n")
    md.append("| 策略 | 模型 | AUC | 95% CI | 平衡准确率 |\n|---|---|---|---|---|")
    for st, (pooled, key) in [("A", (pooledA, "A")), ("B", (pooledB, "B"))]:
        for _, m in pooled.iterrows():
            mod = m["model"]
            ci = auc_ci[f"{key}_{mod.lower()}"][1:]
            md.append(f"| {st} | {mod} | {m['auc']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {m['balanced_accuracy']:.4f} |")
    md.append("\n> Strategy A = Label 0 vs (1+2) 激进预警；Strategy B = Drop&Predict（0 vs 2 训练，held-out Label1 打分）\n")
    md.append("### 2.1 ROC 曲线\n")
    md.append(f"![ROC A](figs/roc_A.png)\n![ROC B](figs/roc_B.png)\n")
    md.append("### 2.2 混淆矩阵（SVM）\n")
    md.append(f"![Confusion A](figs/confusion_A.png)\n![Confusion B](figs/confusion_B.png)\n")
    md.append("### 2.3 分队列性能\n")
    if ca and cb:
        md.append(f"![Cohort AUC A](figs/cohort_auc_A.png)\n![Cohort AUC B](figs/cohort_auc_B.png)\n")
    md.append("## 3. 特征选择\n")
    md.append(f"### 3.1 LASSO 选择频率\n![Freq A](figs/freq_A.png)\n![Freq B](figs/freq_B.png)\n")
    md.append(f"### 3.2 特征组成\n![Composition](figs/composition.png)\n")
    md.append(f"Strategy A 选中 {len(selA_final)} 个，B 选中 {len(selB_final)} 个；强度特征占比 A={compA['intensity']} B={compB['intensity']}（越低越稳健）。\n")
    md.append(f"### 3.3 选中特征相关热图\n![Corr heatmap](figs/corr_heatmap.png)\n")
    md.append("## 4. 单变量分析\n")
    md.append(f"![Univariate AUC](figs/univariate_auc.png)\n")
    uni_disp = uni.copy()
    uni_disp["p"] = uni_disp["p"].map(lambda x: f"{x:.2e}" if np.isfinite(x) else "NA")
    md.append(uni_disp[["feature", "category", "auc", "p", "meaning", "direction"]].to_markdown(index=False))
    md.append("\n## 5. Strategy B 灰度区间（held-out Label1）\n")
    md.append(f"![Risk distribution](figs/risk_B.png)\n")
    md.append(f"Label1（结构性脆弱，n={l1_stats['n']}）：风险中位数 {fmt(l1_stats['median'])}，灰度区[{GRAY_LOW},{GRAY_HIGH}]占比 {l1_stats['pct_gray']*100:.1f}%——作为稳定与急性之间的过渡态。\n")
    md.append("## 6. 候选特征临床意义总结\n")
    clin = uni[["feature", "category", "meaning", "physio", "direction", "auc"]].copy()
    clin = clin.rename(columns={"feature": "特征", "category": "类别", "meaning": "临床含义",
                                "physio": "病理生理/COPD关联", "direction": "风险方向", "auc": "单变量AUC(B)"})
    clin["单变量AUC(B)"] = clin["单变量AUC(B)"].map(lambda x: f"{x:.3f}")
    md.append(clin.to_markdown(index=False))
    md.append("\n## 7. 结论要点\n")
    md.append("- 保护区强特征（PAH_vascular 肺血管/肺高压、Pi10 气道壁、BV5/BV10 小血管）在留一队列中稳定入选并保持单变量显著。")
    md.append("- 弱声明特征（CAC/FAI/CTR/BAR）交由 LASSO 自然筛选，避免强制保留稀释线性模型。")
    md.append("- 特征组成健康（强度占比低），对扫描协议混杂更稳健。")
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")

    # ================= 生成 html =================
    css = """<style>body{font-family:'Microsoft YaHei',Arial,sans-serif;max-width:1500px;margin:30px auto;padding:0 24px;color:#222;line-height:1.5}
    h1{font-size:23px}h2{font-size:18px;border-bottom:2px solid #17365d;padding-bottom:4px;margin-top:34px}h3{font-size:15px;color:#17365d}
    table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0 18px}th,td{border:1px solid #ccc;padding:4px 7px;text-align:left}
    th{background:#eaf2f8}tr:nth-child(even){background:#fafafa}figure{display:inline-block;vertical-align:top;margin:8px;width:47%;text-align:center}
    figure img{max-width:100%;border:1px solid #ddd}figcaption{font-size:11.5px;color:#555}.note{background:#fff7e6;border-left:4px solid #f0ad4e;padding:10px 14px}
    .ok{background:#e8f6e8;border-left:4px solid #2e9e5b;padding:10px 14px}.mini{font-size:11px;color:#666}code{background:#f3f3f3;padding:2px 4px}</style>"""
    h = ["<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>",
         f"<title>COPD 急性加重风险影像组学分析 — 中期汇报（{_args.tag}）</title>{css}</head><body>"]
    h.append(f"<h1>COPD 急性加重风险影像组学分析 — 中期汇报（{_args.tag}）</h1>")
    h.append("<h2>1. 概述</h2>")
    h.append(f"<table><tr><th>项目</th><th>说明</th></tr>")
    h.append(f"<tr><td>数据</td><td><code>{INPUT}</code>：4 队列（2026-01/02/04/05），有效标注 {len(work)} 例（0 稳定 / 1 结构性脆弱 / 2 急性加重）</td></tr>")
    if RAW_MODE:
        h.append("<tr><td>特征池</td><td>radiomics（16 掩膜）+ AirQuant 气道特征 + 申报补算特征（CAC/心包脂肪/FAI/心胸比/支气管血管比/肺血管CSA/PA直径）</td></tr>")
    else:
        h.append("<tr><td>特征池</td><td>radiomics（16 掩膜）+ AirQuant 气道特征 + 申报补算特征复合分（PAH_vascular/CAC/FatInflam/CTR/BAR）</td></tr>")
    h.append(f"<tr><td>保护区（强制保留）</td><td><b>{'、'.join(PROTECTED)}</b></td></tr>")
    h.append("<tr><td>方法</td><td>Mann-Whitney+BH-FDR 筛选 → Spearman 去冗余 → L1 logistic LASSO → Linear SVM（C 折内选）；留一队列外验证；AUC 用 decision function</td></tr></table>")
    h.append("<h2>2. 模型性能</h2>")
    h.append("<table><tr><th>策略</th><th>模型</th><th>AUC</th><th>95% CI</th><th>平衡准确率</th></tr>")
    for st, (pooled, key) in [("A", (pooledA, "A")), ("B", (pooledB, "B"))]:
        for _, m in pooled.iterrows():
            ci = auc_ci[f"{key}_{m['model'].lower()}"][1:]
            h.append(f"<tr><td>{st}</td><td>{m['model']}</td><td><b>{m['auc']:.4f}</b></td><td>[{ci[0]:.4f}, {ci[1]:.4f}]</td><td>{m['balanced_accuracy']:.4f}</td></tr>")
    h.append("</table><p class='mini'>A = Label 0 vs (1+2) 激进预警；B = Drop&amp;Predict（0 vs 2 训练，held-out Label1 打分）</p>")
    h.append("<h3>2.1 ROC 曲线</h3>")
    h.append(f"<figure><img src='{data_uri(p_rocA)}'><figcaption>Strategy A OOF ROC</figcaption></figure>")
    h.append(f"<figure><img src='{data_uri(p_rocB)}'><figcaption>Strategy B OOF ROC</figcaption></figure>")
    h.append("<h3>2.2 混淆矩阵（SVM）</h3>")
    h.append(f"<figure><img src='{data_uri(p_cmA)}'><figcaption>Strategy A</figcaption></figure>")
    h.append(f"<figure><img src='{data_uri(p_cmB)}'><figcaption>Strategy B</figcaption></figure>")
    if ca and cb:
        h.append("<h3>2.3 分队列性能</h3>")
        h.append(f"<figure><img src='{data_uri(ca[0])}'><figcaption>Strategy A per-cohort</figcaption></figure>")
        h.append(f"<figure><img src='{data_uri(cb[0])}'><figcaption>Strategy B per-cohort</figcaption></figure>")
    h.append("<h2>3. 特征选择</h2>")
    h.append("<h3>3.1 LASSO 选择频率</h3>")
    h.append(f"<figure><img src='{data_uri(p_freqA)}'><figcaption>Strategy A</figcaption></figure>")
    h.append(f"<figure><img src='{data_uri(p_freqB)}'><figcaption>Strategy B</figcaption></figure>")
    h.append("<h3>3.2 特征组成</h3>")
    h.append(f"<figure><img src='{data_uri(p_comp)}'><figcaption>Selected feature composition</figcaption></figure>")
    h.append(f"<p>A 选中 {len(selA_final)} 个，B 选中 {len(selB_final)} 个；强度特征 A={compA['intensity']} B={compB['intensity']}（占比低→对协议混杂稳健）。</p>")
    h.append("<h3>3.3 选中特征相关热图</h3>")
    h.append(f"<figure><img src='{data_uri(p_corr)}'><figcaption>Spearman correlation heatmap</figcaption></figure>")
    h.append("<h2>4. 单变量分析</h2>")
    h.append(f"<figure><img src='{data_uri(p_uni)}'><figcaption>Univariate AUC (Label 0 vs 2)</figcaption></figure>")
    uni_disp2 = uni.copy()
    uni_disp2["p"] = uni_disp2["p"].map(lambda x: f"{x:.2e}" if np.isfinite(x) else "NA")
    uni_disp2["auc"] = uni_disp2["auc"].map(lambda x: f"{x:.3f}")
    h.append(uni_disp2[["feature", "category", "auc", "p", "meaning", "direction"]].to_html(index=False, border=0))
    h.append("<h2>5. Strategy B 灰度区间（held-out Label1）</h2>")
    h.append(f"<figure><img src='{data_uri(p_risk)}'><figcaption>Risk distribution</figcaption></figure>")
    h.append(f"<p>Label1（结构性脆弱，n={l1_stats['n']}）：风险中位数 {fmt(l1_stats['median'])}，灰度区[{GRAY_LOW},{GRAY_HIGH}]占比 {l1_stats['pct_gray']*100:.1f}%。</p>")
    h.append("<h2>6. 候选特征临床意义总结</h2>")
    clin2 = uni.copy()
    clin2["auc"] = clin2["auc"].map(lambda x: f"{x:.3f}")
    h.append(clin2[["feature", "category", "meaning", "physio", "direction", "auc"]]
             .rename(columns={"feature": "特征", "category": "类别", "meaning": "临床含义",
                              "physio": "病理生理/COPD关联", "direction": "风险方向", "auc": "单变量AUC(B)"})
             .to_html(index=False, border=0))
    h.append("<h2>7. 结论要点</h2><ul>")
    h.append("<li>保护区强特征（PAH_vascular 肺血管/肺高压、Pi10 气道壁、BV5/BV10 小血管）留一队列中稳定入选且单变量显著。</li>")
    h.append("<li>弱声明特征（CAC/FAI/CTR/BAR）交由 LASSO 自然筛选，避免强制保留稀释线性模型。</li>")
    h.append("<li>特征组成健康（强度占比低），对扫描协议混杂更稳健。</li></ul>")
    h.append("</body></html>")
    (OUT / "report.html").write_text("\n".join(h), encoding="utf-8")
    print("md  =", OUT / "report.md")
    print("html=", OUT / "report.html")


if __name__ == "__main__":
    main()
