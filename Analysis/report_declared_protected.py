# -*- coding: utf-8 -*-
"""
综合报告：申报清单补算特征 + Pi10/BV 保护区 结合 radiomics + AQ 特征
==================================================================
L1 LASSO + Linear SVM 嵌套特征选择（留一队列）完整分析报告，含三部分：
  1) 单变量 AUC 分析：保护区 12 项 + 各折实际选中特征，在 Strategy A（0 vs 1+2）
     与 B（0 vs 2）下分别算 AUC / Mann-Whitney U / 方向
  2) 组学分析：pooled OOF 指标 + ROC + LASSO 选择频率 + 特征组成 + Strategy B
     灰度区间（held-out Label1 风险分布）
  3) 临床意义总结：每个候选/选中特征的类别、临床含义、COPD/急性加重相关病理生理、
     预期方向与本队列单变量 AUC 汇总

依赖 optimize_lasso_svm_binary.py 已保存的 OOF/fold/selection 结果。

用法：
  python Analysis/report_declared_protected.py --parent "E:\\DICOM\\reports\\feature_selection_ordinal_ae\\pi10bv_declared9" --tag pi10bv_declared9
输出：
  E:\\DICOM\\reports\\feature_selection_ordinal_ae\\pi10bv_declared9\\report_declared_protected\\report.html
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
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lasso_svm_nested import is_feature  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_ap = argparse.ArgumentParser()
_ap.add_argument("--parent", default=r"E:\DICOM\reports\feature_selection_ordinal_ae\pi10bv_declared9")
_ap.add_argument("--tag", default="pi10bv_declared9")
_args = _ap.parse_args()

BASE = Path(_args.parent)
OUT = BASE / "report_declared_protected"
INPUT = r"E:\DICOM\results\patients_feature_label.csv"
GRAY_LOW, GRAY_HIGH = 0.4, 0.6
SEED = 20260830
COHORT_MAP = {"Jan-26": "2026-01", "Feb-26": "2026-02", "Apr-26": "2026-04", "May-26": "2026-05"}

# 保护区 12 项（申报 9 项 + Pi10 + BV5 + BV10；无 Fractal）
PROTECTED = ["Pi10", "Vessel_BV5_pct", "Vessel_BV10_pct",
             "CAC_Agatston", "CAC_Mass_mg", "EpiFat_Volume_mm3", "EpiFat_Mean_HU",
             "FAI_pericoronary_HU", "CardioThoracic_Ratio", "BronchoArtery_Ratio",
             "Vessel_CSA_mean_mm2", "PA_Equivalent_Diameter_mm"]

# 临床意义词典：key 为特征名前缀/正则（按顺序匹配）；value = (类别, 临床含义, 病理生理与COPD/急性加重关联, 风险升高方向)
CLIN = [
    (r"^Pi10$", ("气道重塑", "气道壁厚度@内周长10mm", "气道壁增厚是 COPD 气道重塑/炎症的核心标志，与气流受限(FEV1↓)和急性加重风险↑独立相关；Pi10↑→风险↑", "Pi10 ↑ 风险↑")),
    (r"Vessel_BV5_pct|Vessel_BV10_pct", ("肺血管", "小血管(<5/<10mm²)血容量占比", "COPD 肺泡毛细血管床破坏/修剪→小血管减少→BV5/BV10 下降；低值提示肺血管床丢失与肺气肿", "BV ↓ 风险↑(反向)")),
    (r"Vessel_Fractal_Dim", ("肺血管", "肺血管网分形维度", "血管网空间复杂度：远端修剪→维度↓；畸形增生→维度↑", "双向")),
    (r"Vessel_CSA_mean_mm2", ("肺血管", "肺血管平均截面积", "肺血管重塑/破坏的总体量度；CSA 异常反映血管床丢失或淤血", "异常即风险")),
    (r"Vessel_Volume_mm3", ("肺血管", "肺血管总体积", "肺血管床总量，COPD 中通常下降", "↓ 风险↑(反向)")),
    (r"Vessel_Tortuosity", ("肺血管", "血管迂曲度", "缺氧/高压下血管迂曲增加", "↑ 风险↑")),
    (r"Vessel_", ("肺血管", "肺血管其它指标", "血管网形态/分支/骨架特征", "—")),
    (r"^CAC_Agatston$", ("钙化", "冠脉钙化 Agatston 积分", "冠脉粥样硬化钙化负荷；COPD 患者心血管共病与急性事件风险↑；高积分→心血管风险↑", "CAC ↑ 风险↑")),
    (r"^CAC_Mass_mg$", ("钙化", "冠脉钙化质量(mg)", "与 Agatston 互补的钙化总量度量", "↑ 风险↑")),
    (r"^CAC_Volume_mm3$", ("钙化", "冠脉钙化体积(mm³)", "钙化体积负荷", "↑ 风险↑")),
    (r"^EpiFat_Volume_mm3$", ("心包脂肪", "心外膜脂肪体积", "代谢/肥胖相关炎症性脂肪堆积，与心血管风险及全身炎症相关", "↑ 风险↑")),
    (r"^EpiFat_Mean_HU$", ("心包脂肪", "心外膜脂肪平均密度(HU)", "密度升高(接近水)提示脂肪炎症浸润", "↑(更致密) 风险↑")),
    (r"^FAI_pericoronary_HU$", ("心包脂肪", "冠周脂肪衰减指数(FAI)", "冠脉周围脂肪炎症的影像学标志，CCTA 心血管风险预测指标", "FAI ↑ 风险↑")),
    (r"^CardioThoracic_Ratio$", ("心脏", "心胸比", ">0.5 提示心脏扩大，反映心衰/容量负荷，COPD 肺心病相关", "↑ 风险↑")),
    (r"^BronchoArtery_Ratio$", ("气道", "支气管-血管比(BAR)", ">1 提示支气管扩张(柱状)；比值升高提示气道扩张性病变", "↑ 风险↑")),
    (r"^PA_Equivalent_Diameter_mm$", ("肺动脉", "主肺动脉等效直径", "PA 增宽(>29mm)提示肺动脉高压；COPD 低氧→肺高压→PA↑", "↑ 风险↑")),
    (r"PA_Ao_", ("肺动脉", "PA/Ao 直径比", "肺动脉高压影像标志(>1 提示 PAH)", "↑ 风险↑")),
    (r"^Aorta_", ("主动脉", "主动脉直径/壁厚", "主动脉粥样硬化/壁增厚，心血管共病", "↑ 风险↑")),
    (r"^RV_", ("心脏", "右心室容积", "肺心病/右心负荷", "↑ 风险↑")),
    (r"^LV_", ("心脏", "左心室容积", "左心结构", "—")),
    (r"TD_", ("气道", "气管腔/壁 T-D 比", "气道扩张度(T/D)，支扩相关，跨树异质性反映炎性重塑", "TD 异质/扩张 风险↑")),
    (r"blur_", ("气道", "气道模糊/边界特征", "气道壁边界模糊(炎症)", "↑ 风险↑")),
    (r"wall_|WA_", ("气道", "气道壁面积", "气道壁增厚(重塑)", "↑ 风险↑")),
    (r"Din_|Dout_", ("气道", "气道内/外径", "气道口径：内径缩小提示狭窄/重塑", "口径异常 风险↑")),
    (r"^mean_", ("气道", "气道均值指标", "AirQuant 气道整体均值", "—")),
    (r"Pi10|Pi_", ("气道", "气道壁厚度(Pi)", "气道重塑核心指标", "Pi10 ↑ 风险↑")),
    (r"Lobe_.*LAA|Lung_.*LAA", ("肺气肿", "肺叶/全肺 LAA-950%", "肺气肿程度；低衰减区比例↑→肺气肿↑、加重风险↑", "↑ 风险↑")),
    (r"Perc15", ("肺气肿", "Perc15(第15百分位HU)", "肺气肿密度指标；越低肺气肿越重", "↓ 风险↑(反向)")),
    (r"Lobe_|Lung_", ("肺叶", "肺叶体积/占比", "肺叶容积与分布", "—")),
    (r"Airway_", ("气道", "气道体积/肺叶耦合", "气道总体积与肺叶分布", "—")),
    (r"Diaphragm_", ("膈肌", "膈肌形态(肺底填充比)", "膈肌变平(肺过度充气)", "↓填充比 风险↑")),
    (r"tortuosity|branch|junction|pruning|generation", ("气道结构", "气道树图论特征", "气道树分支/末梢修剪/迂曲", "—")),
    (r"firstorder_(Energy|TotalEnergy)", ("强度(混杂)", "firstorder 能量强度", "受 CT 扫描协议/管电流混杂污染，判别力在队内标准化后塌陷——谨慎解释", "混杂")),
    (r"firstorder_", ("firstorder", "一阶统计量", "体素灰度分布统计", "—")),
    (r"shape_", ("形态学", "shape 形态学特征", "ROI 几何形态(体积/表面积/球形度等)", "—")),
    (r"_(glcm|glrlm|glszm|gldm|ngtdm)_", ("纹理", "纹理特征", "灰度空间分布/异质性", "—")),
]


def clin_for(feature):
    for pat, info in CLIN:
        if re.search(pat, feature):
            return info
    return ("其它", "—", "—", "—")


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -600, 600)
    return 1.0 / (1.0 + np.exp(-x))


def data_uri(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


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


def univariate_auc(work, feature, y):
    """单变量 AUC（NaN 行剔除）+ Mann-Whitney U p + 方向（风险组 vs 稳定组均值差）。"""
    x = pd.to_numeric(work[feature], errors="coerce").to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10 or np.unique(y[m]).size < 2:
        return {"auc": np.nan, "p": np.nan, "mean_risk": np.nan, "mean_stable": np.nan}
    xv, yv = x[m], y[m].astype(int)
    try:
        auc = float(roc_auc_score(yv, xv))
    except Exception:
        auc = np.nan
    if np.unique(yv).size == 2:
        try:
            p = float(mannwhitneyu(xv[yv == 1], xv[yv == 0], alternative="two-sided").pvalue)
        except Exception:
            p = np.nan
    else:
        p = np.nan
    return {"auc": auc, "p": p,
            "mean_risk": float(xv[yv == 1].mean()), "mean_stable": float(xv[yv == 0].mean())}


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
    return work, feature_names


def main():
    OUT.mkdir(parents=True, exist_ok=True)
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
    yB_all = (label == 2).astype(int)

    # ---- 单变量 AUC（对候选特征：保护区 12 + A/B 折内实际选中特征并集）----
    selA_final = selA[selA["frequency"] > 0]["feature"].tolist()
    selB_final = selB[selB["frequency"] > 0]["feature"].tolist()
    cand = list(dict.fromkeys(PROTECTED + selA_final + selB_final))
    uni = []
    for f in cand:
        if f not in work.columns:
            continue
        ua = univariate_auc(work, f, yA)
        ub = univariate_auc(work, f, yB_all)
        cat, meaning, physio, direction = clin_for(f)
        freqA = int(selA[selA["feature"] == f]["frequency"].sum()) if f in selA["feature"].values else 0
        freqB = int(selB[selB["feature"] == f]["frequency"].sum()) if f in selB["feature"].values else 0
        uni.append({
            "feature": f, "category": cat, "clinical_meaning": meaning,
            "physio": physio, "direction": direction,
            "AUC_A": ua["auc"], "p_A": ua["p"], "mean_risk_A": ua["mean_risk"], "mean_stable_A": ua["mean_stable"],
            "AUC_B": ub["auc"], "p_B": ub["p"], "mean_risk_B": ub["mean_risk"], "mean_stable_B": ub["mean_stable"],
            "freq_A": freqA, "freq_B": freqB,
        })
    uni_df = pd.DataFrame(uni).sort_values(["freq_A", "AUC_A"], ascending=[False, False]).reset_index(drop=True)
    uni_df.to_csv(OUT / "univariate_auc.csv", index=False, encoding="utf-8-sig")

    # ---- 组学：pooled 指标 + ROC ----
    oofA["risk"] = sigmoid(oofA["lasso_score"])
    oofB["risk"] = sigmoid(oofB["lasso_score"])
    yA_oof = (oofA["Label"].to_numpy(int) >= 1).astype(int)
    ob = oofB[oofB["in_train"]].reset_index(drop=True)
    yB_oof = (ob["Label"].to_numpy(int) == 2).astype(int)

    def fig_roc(y, s_l, s_s, fname, title):
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        for sc, c, lab in [(s_l, "#1f77b4", "LASSO"), (s_s, "#d62728", "LinearSVM")]:
            fpr, tpr, _ = roc_curve(y, sc)
            ax.plot(fpr, tpr, lw=2, color=c, label=f"{lab} AUC={roc_auc_score(y, sc):.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", title=title)
        ax.grid(alpha=0.2); ax.legend(fontsize=9, loc="lower right")
        fig.tight_layout(); p = OUT / fname
        fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
        return p

    p_rocA = fig_roc(yA_oof, oofA["lasso_score"].to_numpy(float), oofA["svm_score"].to_numpy(float),
                     "roc_A.png", "Strategy A OOF ROC (stable vs risk)")
    p_rocB = fig_roc(yB_oof, ob["lasso_score"].to_numpy(float), ob["svm_score"].to_numpy(float),
                     "roc_B.png", "Strategy B OOF ROC (Label 0 vs 2)")

    def fig_freq(sel, fname, title):
        top = sel[sel["frequency"] > 0].sort_values("frequency", ascending=False).head(25).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        ax.barh(top["feature"], top["frequency"], color="#2ca02c")
        ax.set_xlim(0, max(1, int(sel["frequency"].max())))
        ax.set_xlabel("Outer folds selected by LASSO"); ax.set_title(title)
        ax.grid(axis="x", alpha=0.2); fig.tight_layout()
        p = OUT / fname; fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
        return p

    p_freqA = fig_freq(selA, "freq_A.png", "Strategy A LASSO selection frequency")
    p_freqB = fig_freq(selB, "freq_B.png", "Strategy B LASSO selection frequency")

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
        ax.legend(fontsize=9); ax.grid(alpha=0.2); fig.tight_layout()
        p = OUT / fname; fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
        return p

    p_risk = fig_risk(oofB, "risk_B.png")

    # Label1 灰度区间（策略 B held-out）
    l1 = oofB.loc[oofB["Label"] == 1, "risk"].dropna().to_numpy(float)
    l1_stats = {"n": int(len(l1)),
                "median": float(np.median(l1)) if len(l1) else np.nan,
                "pct_gray": float(np.mean((l1 >= GRAY_LOW) & (l1 <= GRAY_HIGH))) if len(l1) else np.nan,
                "pct_below": float(np.mean(l1 < GRAY_LOW)) if len(l1) else np.nan,
                "pct_above": float(np.mean(l1 > GRAY_HIGH)) if len(l1) else np.nan}

    compA = feature_composition(selA_final)
    compB = feature_composition(selB_final)

    # ---- 汇总 JSON ----
    summary = {
        "tag": _args.tag, "protected": PROTECTED, "input": INPUT,
        "strategy_A": {"n": len(oofA), "n_selected": len(selA_final), "composition": compA},
        "strategy_B": {"n_train": int(len(ob)), "n_label1": int((oofB["Label"] == 1).sum()),
                       "n_selected": len(selB_final), "composition": compB,
                       "label1_risk": l1_stats},
        "univariate_candidates": len(uni_df),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ================= HTML =================
    css = """<style>body{font-family:'Microsoft YaHei',Arial,sans-serif;max-width:1600px;margin:30px auto;padding:0 24px;color:#222;line-height:1.5}
    h1{font-size:22px}h2{font-size:18px;border-bottom:2px solid #17365d;padding-bottom:4px;margin-top:34px}h3{font-size:15px;color:#17365d}
    table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0 18px}th,td{border:1px solid #ccc;padding:4px 7px;text-align:left;vertical-align:top}
    th{background:#eaf2f8}tr:nth-child(even){background:#fafafa}figure{display:inline-block;vertical-align:top;margin:8px;width:46%;text-align:center}
    figure img{max-width:100%;border:1px solid #ddd}figcaption{font-size:11.5px;color:#555}.note{background:#fff7e6;border-left:4px solid #f0ad4e;padding:10px 14px}
    .prot{background:#eef7ff;font-weight:600}.ok{background:#e8f6e8}.warn{background:#fff3cd}code{background:#f3f3f3;padding:2px 4px}
    .sig{background:#d4edda}.nonsig{background:#fff}.mini{font-size:11px;color:#666}</style>"""
    h = ["<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>",
         f"<title>申报补算特征保护区 + LASSO/SVM 综合分析报告 ({_args.tag})</title>{css}</head><body>"]
    h.append(f"<h1>申报清单补算特征保护区 + 旧版 LASSO 选择 + Linear SVM 综合分析</h1>")
    h.append(f"<p class='note'>保护区 12 项：<b>{'、'.join(PROTECTED[:6])}、…</b>（申报 9 项 + Pi10 + BV5/BV10，去掉 Fractal），"
             "结合已算好的 radiomics 特征（16 掩膜）+ AirQuant 气道特征（AQ），每外层折内做 "
             "Mann-Whitney+BH-FDR 单变量筛选 → Spearman 去冗余 → <b>L1 logistic LASSO</b> → <b>Linear SVM</b>（C∈{0.01..100} 折内选 C）。"
             "保护区特征无论单变量/LASSO 结果如何都强制进入候选并保留在最终模型。留一队列外验证（2026-01/02/04/05）。AUC 用 decision function。</p>")

    # ---- 1. 单变量 AUC ----
    h.append("<h2>1. 单变量 AUC 分析（候选特征）</h2>")
    h.append("<p class='mini'>对保护区 12 项 + 各折实际选中特征并集，分别算 Strategy A（Label 0 vs 1+2）与 "
             "Strategy B（Label 0 vs 2）的单变量 AUC、Mann-Whitney U p 值、风险组均值与稳定组均值。"
             "绿色=单变量 AUC≥0.60 或 p<0.05；保护区行蓝色加粗。freq_A/B = 被 LASSO 在 4 个外层折中选中的次数。</p>")
    uni_disp = uni_df.copy()
    # 标记保护区行（first column 加粗高亮）
    prot_set = set(PROTECTED)
    for idx, r in uni_disp.iterrows():
        if r["feature"] in prot_set:
            uni_disp.at[idx, "feature"] = "🔒 " + str(r["feature"])
    h.append(uni_disp.drop(columns=["physio", "clinical_meaning"]).to_html(
        index=False, border=0, float_format=lambda x: f"{x:.3f}", classes="uni"))
    h.append("<style>.uni tr:has(td) td:first-child{font-weight:600}</style>")

    # ---- 2. 组学分析 ----
    h.append("<h2>2. 组学分析（嵌套 LASSO + Linear SVM，留一队列）</h2>")
    h.append("<h3>2.1 Strategy A：Label 0 vs (1+2) — pooled OOF</h3>")
    h.append(pooledA.drop(columns=["config", "strategy"], errors="ignore").to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}"))
    h.append(f"<h3>2.2 Strategy B：Drop &amp; Predict（0 vs 2 训练，held-out Label1 打分）— pooled OOF（0/2 群体）</h3>")
    h.append(pooledB.drop(columns=["config", "strategy"], errors="ignore").to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}"))
    h.append(f"<p>held-out Label1（结构性脆弱，n={l1_stats['n']}）：风险中位数 {fmt(l1_stats['median'])}，"
             f"灰度区[{GRAY_LOW},{GRAY_HIGH}]占比 {l1_stats['pct_gray']*100:.1f}%，"
             f"低于 {l1_stats['pct_below']*100:.1f}%，高于 {l1_stats['pct_above']*100:.1f}%。"
             "（Label1 作为介于稳定与急性之间的过渡，期望其风险分落在灰度区。）</p>")
    h.append(f"<figure><img src='{data_uri(p_rocA)}'><figcaption>Strategy A OOF ROC</figcaption></figure>")
    h.append(f"<figure><img src='{data_uri(p_rocB)}'><figcaption>Strategy B OOF ROC</figcaption></figure>")
    h.append(f"<figure><img src='{data_uri(p_risk)}'><figcaption>Strategy B held-out Label1 风险分布</figcaption></figure>")
    h.append(f"<h3>2.3 LASSO 选择频率（外折）</h3>")
    h.append(f"<figure><img src='{data_uri(p_freqA)}'><figcaption>Strategy A 选择频率</figcaption></figure>")
    h.append(f"<figure><img src='{data_uri(p_freqB)}'><figcaption>Strategy B 选择频率</figcaption></figure>")
    h.append("<h3>2.4 特征组成</h3>")
    h.append(f"<p>Strategy A 选中 {len(selA_final)} 个：形态 {compA['shape']} / 纹理 {compA['texture']} / firstorder其他 {compA['firstorder_other']} / "
             f"<b>强度 {compA['intensity']}</b> / 结构 {compA['structural']}。"
             f"Strategy B 选中 {len(selB_final)} 个：形态 {compB['shape']} / 纹理 {compB['texture']} / firstorder其他 {compB['firstorder_other']} / "
             f"<b>强度 {compB['intensity']}</b> / 结构 {compB['structural']}。</p>")
    h.append("<p class='note'>强度特征（firstorder Energy/TotalEnergy）受 CT 扫描协议混杂污染，队内标准化后判别力塌陷——"
             "选中特征中强度占比应尽量低；结构特征（气道/血管/肺叶/钙化/脂肪等）越均衡越稳健。</p>")
    h.append("<h3>2.5 外层折叠明细</h3>")
    h.append("<h4>Strategy A</h4>" + foldA.drop(columns=["config", "strategy", "comp"], errors="ignore").to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}"))
    h.append("<h4>Strategy B</h4>" + foldB.drop(columns=["config", "strategy", "comp"], errors="ignore").to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}"))

    # ---- 3. 临床意义总结 ----
    h.append("<h2>3. 候选特征临床意义总结</h2>")
    h.append("<p>每个候选/选中特征的类别、临床含义、COPD/急性加重相关病理生理与预期风险方向，结合本队列单变量 AUC（A/B）与 LASSO 选中频率。</p>")
    clin = uni_df[["feature", "category", "clinical_meaning", "physio", "direction",
                   "AUC_A", "p_A", "AUC_B", "p_B", "freq_A", "freq_B"]].copy()
    clin["AUC_A(95%方向)"] = clin["AUC_A"].map(lambda x: fmt(x, 3))
    clin = clin.rename(columns={"feature": "特征", "category": "类别", "clinical_meaning": "临床含义",
                                "physio": "病理生理 / COPD-急性加重关联", "direction": "风险升高方向",
                                "AUC_A": "单变量AUC(A)", "p_A": "p(A)", "AUC_B": "单变量AUC(B)", "p_B": "p(B)",
                                "freq_A": "LASSO选中(A)", "freq_B": "LASSO选中(B)"})
    h.append(clin.to_html(index=False, border=0, float_format=lambda x: f"{x:.3f}"))

    # ---- 结语 ----
    h.append("<h2>4. 结论要点</h2><ul>")
    h.append("<li>保护区（申报 9 项 + Pi10/BV5/BV10）为临床/病理生理明确的影像学标志：钙化(CAC)、心包脂肪+FAI、心胸比、支气管-血管比、肺血管 CSA、PA 直径、气道壁(Pi10)、小血管容量(BV)。</li>")
    h.append("<li>若保护区特征在留一队列中反复入选且单变量方向符合预期（如 CAC↑/FAI↑/CTR↑→风险↑、BV5↓→风险↑），则说明它们独立于扫描协议、对急性加重有增量判别力。</li>")
    h.append("<li>强度特征占比低（<30%）时模型特征组成健康、对协议混杂更稳健。</li>")
    h.append("</ul></body></html>")
    (OUT / "report.html").write_text("\n".join(h), encoding="utf-8")
    print("report =", OUT / "report.html")
    print("univariate candidates:", len(uni_df))


if __name__ == "__main__":
    main()
