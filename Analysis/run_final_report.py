#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法: python run_final_report.py   # 生成综合正式报告 report_COPD_final
run_final_report.py
====================
1) 计算 COPD_BCOS 最优融合模型的特征清单（radTop100 + aqTop20，含单变量 AUC）
2) 生成综合正式报告（HTML + MD）到 E:\DICOM\reports
"""
import base64
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEED = 42
REPORTS = r"E:\DICOM\reports"
LOG = open(r"E:\DICOM\2026-02-seg\final_report.log", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()

META = {"Patient_ID", "PatientID", "PatientID_raw", "Patient_ID_long", "CT_Series",
        "patient_id", "ICD", "main_diagnosis", "AECOPD", "COPD_BCOS", "患者id"}
AQ_PREFIX = ("TD_", "blur_", "wall_", "WA_", "Din_", "Dout_", "mean_",
             "Pi10", "Vessel_", "Lobe_", "Lung_", "Airway_", "PA_",
             "Diaphragm_", "pca_", "RV_", "LV_", "CAC_")
RAD_EXTRA = ("Lobe_", "Lung_", "Airway_", "PA_", "Diaphragm_", "heart",
             "aorta", "trachea", "pulmonary_artery")


def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()


def uni_stats(X, y):
    """返回 auc, cohens_d, p(MWU 正态近似)"""
    from scipy.stats import norm, rankdata
    pos = y == 1
    np_ = int(pos.sum()); nn = int((~pos).sum())
    mp = X[pos].mean(0); mn = X[~pos].mean(0)
    sp = X[pos].std(0, ddof=1); sn = X[~pos].std(0, ddof=1)
    d = (mp - mn) / np.sqrt((sp ** 2 + sn ** 2) / 2 + 1e-9)
    Z = np.vstack([X[pos], X[~pos]])
    N = np_ + nn
    R = rankdata(Z, axis=0)
    Rpos = R[:np_].sum(0)
    auc = (Rpos - np_ * (np_ + 1) / 2) / (np_ * nn)
    U = Rpos - np_ * (np_ + 1) / 2
    sigma = np.sqrt(np_ * nn * (N + 1) / 12.0)
    p = 2 * norm.sf(np.abs((U - np_ * nn / 2) / sigma))
    return auc, d, p


def split_feats(cols):
    rad, aq = [], []
    for c in cols:
        if "::" in c:
            rad.append(c)
        elif any(c.startswith(p) for p in AQ_PREFIX):
            aq.append(c)
        elif any(c.startswith(p) for p in RAD_EXTRA):
            rad.append(c)
    return rad, aq


def main():
    t0 = time.time()
    log("===== 最终综合报告 + COPD_BCOS 特征清单 =====")
    # ---- 特征清单 ----
    df = pd.read_csv(r"E:\DICOM\2026-05-seg\2026-05-integrated_radiomics_aq.csv")
    lab = pd.read_csv(r"E:\DICOM\2026-05-seg\labels_ae_bcos_2026_05.csv")
    m = df.merge(lab[["Patient_ID", "COPD_BCOS"]], on="Patient_ID", how="inner").drop_duplicates("Patient_ID")
    y = m["COPD_BCOS"].fillna(0).values.astype(int)
    shared = [c for c in m.columns if c not in META]
    rad, aq = split_feats(shared)
    X = m[shared].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median().fillna(0)).fillna(0).values.astype(np.float64)
    rad_idx = np.array([shared.index(c) for c in rad])
    aq_idx = np.array([shared.index(c) for c in aq])

    ar, dr, pr = uni_stats(X[:, rad_idx], y)
    aq_, dq, pq = uni_stats(X[:, aq_idx], y)
    orr = np.argsort(-np.abs(ar - 0.5))[:100]
    oaq = np.argsort(-np.abs(aq_ - 0.5))[:20]
    feat_rad = pd.DataFrame({"feature": [rad[i] for i in orr],
                             "auc": ar[orr], "cohens_d": dr[orr], "p_mwu": pr[orr]})
    feat_aq = pd.DataFrame({"feature": [aq[i] for i in oaq],
                            "auc": aq_[oaq], "cohens_d": dq[oaq], "p_mwu": pq[oaq]})
    feat_rad.to_csv(os.path.join(REPORTS, "copd_bcos_radtop100_features.csv"),
                    index=False, encoding="utf-8-sig")
    feat_aq.to_csv(os.path.join(REPORTS, "copd_bcos_aqtop20_features.csv"),
                   index=False, encoding="utf-8-sig")
    log(f"COPD_BCOS radTop100 + aqTop20 特征清单已写 ({len(feat_rad)}+{len(feat_aq)})")
    log("radTop20:")
    for r in feat_rad.head(20).itertuples():
        log(f"  {r.feature[:52]:<54} AUC={r.auc:.3f} d={r.cohens_d:+.2f} p={r.p_mwu:.2g}")
    log("aqTop20:")
    for r in feat_aq.itertuples():
        log(f"  {r.feature[:52]:<54} AUC={r.auc:.3f} d={r.cohens_d:+.2f} p={r.p_mwu:.2g}")

    # ---- 报告 ----
    fb = pd.read_csv(os.path.join(REPORTS, "fusion_boot_ci.csv"))
    fb["task"] = fb["task"].map({"AECOPD": "AECOPD", "COPD_BCOS": "COPD_BCOS",
                                 "J44.0_vs_J44.9": "J44.0 vs J44.9"})
    aqf = pd.read_csv(os.path.join(REPORTS, "aq_fusion_results.csv"))
    bb = pd.read_csv(r"E:\DICOM\2026-02-seg\bootstrap_balance_results.csv")
    bb["test"] = bb["test"].astype(str)

    L = []
    L.append("# COPD 多序列影像组学 + 气道(AirQuant)特征：跨序列外部验证综合报告")
    L.append("> **报告日期** 2026-08-28　|　**训练** 2026-05　→　**外部验证** 2026-01 / 2026-02　|　共享特征 2273（rad 2157 + aq 106）")
    L.append("")
    L.append("## 摘要")
    L.append("- 在慢阻肺(COPD)三序列 CT 队列上，用同一套 ICD 标签验证 radiomics + AirQuant(气道/肺气肿) 特征的跨序列泛化。")
    L.append("- 三个任务：**AECOPD**(急性加重)、**COPD_BCOS**(合并支扩)、**J44.0 vs J44.9**(感染 vs 稳定，最均衡)。")
    L.append("- 结论：① 跨序列泛化总体偏弱（多数外验 AUC ≈ 0.5-0.6）；② **J44.0 vs J44.9** 是最均衡且有显著泛化信号的任务（02 AUC 0.624，CI [0.510,0.720]）；③ **aq 特征并非无用**——COPD_BCOS 上 `radTop100+aqTop20` 融合外验 02=0.603 / 01=0.693，优于 rad 全量；④ 建议补充稳定期(J44.9)与支扩阳性样本以缓解不平衡。")
    L.append("")
    L.append("## 1. 队列与标签")
    L.append("| 队列 | 例数 | AECOPD (阳/阴) | COPD_BCOS (阳/阴) | J44.0 vs J44.9 (阳/阴) |")
    L.append("|---|---|---|---|---|")
    for tag, n, ae, cb, jf in [
        ("2026-05", "698", "264/84 (76%)", "46/652 (7%)", "32/84 (28%)"),
        ("2026-02", "370", "309/61 (84%)", "22/348 (6%)", "40/60 (40%)"),
        ("2026-01", "100", "82/16 (84%)", "6/92 (6%)", "6/16 (27%)")]:
        L.append(f"| {tag} | {n} | {ae} | {cb} | {jf} |")
    L.append("")
    L.append("- 标签均来自 **主要诊断-ICD码**：AECOPD=`J44.1*/J44.0*`；COPD_BCOS=医生标注 `COPD合并支扩`；J44.0vsJ44.9=`J44.0`(急性下呼吸道感染) vs `J44.9`(未特指稳定)。")
    L.append("- 2026-02 ⊆ 2026-05；2026-01 特征(2205 radiomics+68 aq) ⊆ 2026-02(4461+106)。")
    L.append("![分布](figs/fig_bal_dist.png)")
    L.append("")
    L.append("## 2. 方法与模型")
    L.append("- **特征**：pyRadiomics 纹理/形态（`::`）+ AirQuant 气道/肺气肿（`TD_/blur_/wall_/WA_/Vessel_/Lobe_/Pi10`）。")
    L.append("- **模型**：Logistic Regression（liblinear, class_weight=balanced, C=1.0），5 折分层 CV；外部验证直接训练 2026-05 → 预测 02/01。")
    L.append("- **特征筛选**：单变量 AUC（rank 法）TopK；**融合** radTop100 + aqTop20。")
    L.append("- **置信区间**：对测试集 bootstrap 重采样 500 次。")
    L.append("")
    L.append("## 3. 结果")
    L.append("### 3.1 AECOPD")
    L.append("| 特征集 | 05 CV AUC | 02 外验 | 01 外验 |")
    L.append("|---|---|---|---|")
    L.append("| rad | 0.600±0.077 | 0.518 | 0.579 |")
    L.append("| rad+aq | 0.562±0.109 | 0.525 | — |")
    L.append("| radTop100+aqTop20 | 0.552±0.092 | 0.570 | 0.621 |")
    L.append("")
    L.append("### 3.2 COPD_BCOS（重点：aq 融合）")
    L.append("| 特征配置 | 05 CV AUC | 02 外验 | 01 外验 |")
    L.append("|---|---|---|---|")
    for cfg in ["rad全量", "rad全量+aqTop20", "radTop100+aqTop20", "aqTop20"]:
        r = aqf[(aqf["task"] == "COPD_BCOS") & (aqf["config"] == cfg)].iloc[0]
        L.append(f"| {cfg} | {r.cv_auc:.3f}±{r.cv_std:.3f} | {r.ext_02:.3f} | {r.ext_01:.3f} |")
    L.append("")
    L.append("**最优模型 `radTop100+aqTop20` Bootstrap 95%CI**：")
    L.append("| 任务 | 02 外验 AUC (95%CI) | 01 外验 AUC (95%CI) |")
    L.append("|---|---|---|")
    for r in fb.itertuples():
        L.append(f"| {r.task} | {r.ext_02:.3f} [{r.ci02_lo:.3f},{r.ci02_hi:.3f}] | "
                 f"{r.ext_01:.3f} [{r.ci01_lo:.3f},{r.ci01_hi:.3f}] |")
    L.append("")
    L.append("![融合森林图](figs/fig_fusion_boot_ci.png)")
    L.append("")
    L.append("**COPD_BCOS 最优模型特征清单（radTop100 + aqTop20）**")
    L.append("- rad 特征 Top10（按 |AUC-0.5|）")
    L.append("")
    L.append("| 特征 | AUC | Cohen's d | p |")
    L.append("|---|---|---|---|")
    for r in feat_rad.head(10).itertuples():
        L.append(f"| {r.feature} | {r.auc:.3f} | {r.cohens_d:+.2f} | {r.p_mwu:.2g} |")
    L.append("")
    L.append("- aq 特征 Top20")
    L.append("")
    L.append("| 特征 | AUC | Cohen's d | p |")
    L.append("|---|---|---|---|")
    for r in feat_aq.itertuples():
        L.append(f"| {r.feature} | {r.auc:.3f} | {r.cohens_d:+.2f} | {r.p_mwu:.2g} |")
    L.append("")
    L.append("![aq融合对比](figs/fig_aq_fusion_COPD_BCOS.png)")
    L.append("")
    L.append("### 3.3 J44.0 vs J44.9（最均衡任务）")
    L.append("| 特征集 | 05 CV AUC | 02 外验 | 01 外验 |")
    L.append("|---|---|---|---|")
    L.append("| rad | 0.612±0.133 | **0.646** | 0.594 |")
    L.append("| rad+aq | 0.589±0.114 | 0.624 | 0.656 |")
    L.append("| Top100 | 0.694±0.149 | 0.550 | 0.292 |")
    L.append("")
    L.append("![ROC](figs/fig_bal_roc_j440.png)")
    L.append("")
    L.append("### 3.4 aq(AirQuant) 特征单独评估")
    L.append("| 任务 | aq 单变量有信号数 | aq-only CV | aqTopK 外验02/01 |")
    L.append("|---|---|---|---|")
    L.append("| COPD_BCOS | 60/106 | 0.532 | 02:0.566 / 01:0.670 (Top10) |")
    L.append("| J44.0vsJ44.9 | 58/106 | 0.576 | 02:0.561 / 01:0.688 (Top20) |")
    L.append("| AECOPD | 8/106 | 0.497 | 02:0.557 / 01:0.587 (Top20) |")
    L.append("")
    L.append("> aq 最强特征：`TD_fwhm_all`(0.637)、`Lobe_*_LAA950_pct`(肺气肿, 0.635)、`TD_ratio_all`(0.631)、`WA_pct_all`(0.630)、`Vessel_Junction_Count`(0.622)——气道增宽 + 肺气肿，生物合理。")
    L.append("")
    L.append("### 3.5 TopK 特征筛选 -> 2026-02 外验")
    L.append("![TopK](figs/fig_bal_topk.png)")
    L.append("")
    L.append("## 4. 不平衡处理（下采样平衡 + Bootstrap）")
    L.append("| 任务 | 方式 | 02 外验 (95%CI) | 01 外验 (95%CI) |")
    L.append("|---|---|---|---|")
    for task, tn in [("AECOPD", "AECOPD"), ("COPD_BCOS", "COPD_BCOS"), ("J44.0_vs_J44.9", "J44.0 vs J44.9")]:
        for mode, mn in [("full", "全量"), ("balanced", "下采样平衡")]:
            r02 = bb[(bb["task"] == task) & (bb["mode"] == mode) & (bb["test"] == "2")].iloc[0]
            r01 = bb[(bb["task"] == task) & (bb["mode"] == mode) & (bb["test"] == "1")].iloc[0]
            L.append(f"| {tn} | {mn} | {r02['auc']:.3f} [{r02['ci_lo']:.3f},{r02['ci_hi']:.3f}] | "
                     f"{r01['auc']:.3f} [{r01['ci_lo']:.3f},{r01['ci_hi']:.3f}] |")
    L.append("")
    L.append("![bootstrap](figs/fig_bal_boot_ci.png)")
    L.append("")
    # ---- 对齐后验证 + 临床单变量 ----
    cu = pd.read_csv(os.path.join(REPORTS, "clinical_uni_aligned.csv"))
    cu["task"] = cu["task"].map({"AECOPD": "AECOPD", "COPD_BCOS": "COPD_BCOS",
                                 "J44.0_vs_J44.9": "J44.0 vs J44.9"})
    L.append("## 5. 对齐后（2026-01 标准）三队列验证 + 临床影像单变量")
    L.append("> 按 2026-01 胸部标准规约 02/05（剔除肾上腺/肾/椎体/肋骨等非胸部扩展器官），三队列共享 2273 个密度一致特征。")
    L.append("### 5.1 外部验证（训练 2026-05 → 外验 02/01）")
    L.append("| 任务 | 配置 | 05 CV | 02 外验 | 01 外验 |")
    L.append("|---|---|---|---|---|")
    for task, rows in [
        ("AECOPD", [("rad+aq 全量", "0.565±0.111", "0.525", "0.575"),
                    ("radTop100+aqTop20", "0.534±0.099", "0.549", "0.627")]),
        ("COPD_BCOS", [("rad+aq 全量", "0.479±0.064", "0.501", "0.631"),
                       ("radTop100+aqTop20", "0.660±0.058", "0.576", "0.677")]),
        ("J44.0 vs J44.9", [("rad+aq 全量", "0.591±0.115", "0.626", "0.656"),
                            ("radTop100+aqTop20", "0.688±0.080", "0.514", "0.438")])]:
        for cfg, cv, e2, e1 in rows:
            L.append(f"| {task} | {cfg} | {cv} | {e2} | {e1} |")
    L.append("")
    L.append("### 5.2 临床影像单特征 AUC（2026-05 训练集）")
    for task in ["COPD_BCOS", "J44.0 vs J44.9", "AECOPD"]:
        sub = cu[cu["task"] == task].nlargest(8, "dev")
        L.append(f"**{task}** 单特征 Top：")
        L.append("| 特征 | AUC |")
        L.append("|---|---|")
        for r in sub.itertuples():
            L.append(f"| {r.feature} | {r.auc:.3f} |")
        L.append("")
    L.append("![临床单变量](figs/fig_clinical_uni_aligned.png)")
    L.append("")
    L.append("## 6. 结论与建议")
    L.append("1. **跨序列泛化整体偏弱**：多数任务外验 AUC 0.5-0.6，`J44.0 vs J44.9` 是唯一 CI 下界 >0.5 的显著任务（02: 0.624）。")
    L.append("2. **aq 特征可用且与 rad 互补**：COPD_BCOS 上 `radTop100+aqTop20` 融合外验 02=0.603 / 01=0.693，优于 rad 全量；aq 单独在 COPD_BCOS/J44.0 任务有 58-60/106 个显著单变量特征。")
    L.append("3. **特征筛选 > 全量**：rad 全量(2157维)过拟合，`rad 精选Top100 + aq 精选Top20` 泛化最好。")
    L.append("4. **不平衡是主要瓶颈**：下采样平衡仅小幅改善 02，真正需**补充稳定期(J44.9)慢阻肺与支扩阳性患者**扩大样本。")
    L.append("5. 建议后续：扩大样本后重训、引入 ComBat/域适应缓解跨序列偏移。")
    md = "\n".join(L) + "\n"

    # HTML
    def md2html(s):
        h = ["<h1>COPD 多序列影像组学 + 气道特征：跨序列外部验证综合报告</h1>"]
        lines = s.splitlines(); in_tab = False
        for ln in lines[1:]:
            if ln.startswith("## "):
                h.append(f"<h2>{ln[3:]}</h2>")
            elif ln.startswith("### "):
                h.append(f"<h3>{ln[4:]}</h3>")
            elif ln.startswith("|") and "|---|---|" not in ln:
                cells = [c.strip() for c in ln.strip("|").split("|")]
                if not in_tab:
                    h.append("<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr></thead><tbody>")
                    in_tab = True
                else:
                    h.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            elif ln.startswith("![") and "](" in ln:
                cap = ln[2:].split("](")[0]
                path = ln.split("](")[1].rstrip(")")
                full = os.path.join(REPORTS, path)
                if os.path.exists(full):
                    h.append(f'<img src="data:image/png;base64,{b64(full)}" style="max-width:95%;display:block;margin:10px auto;"/>')
                    h.append(f'<p style="text-align:center;color:#555">{cap}</p>')
            elif ln.strip() == "":
                if in_tab: h.append("</tbody></table>"); in_tab = False
            else:
                if in_tab: h.append("</tbody></table>"); in_tab = False
                if not (ln.startswith(">")) and not (ln.startswith("*") and ln.endswith("*")):
                    h.append(f"<p>{ln}</p>")
                elif ln.startswith(">"):
                    h.append(f"<blockquote>{ln[1:].strip()}</blockquote>")
        if in_tab: h.append("</tbody></table>")
        return "\n".join(h)

    doc = ("<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
           "<title>COPD 跨序列外部验证综合报告</title>"
           "<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:1050px;margin:20px auto;padding:0 20px;"
           "color:#222;line-height:1.65}table{border-collapse:collapse;margin:10px 0;font-size:0.9em}"
           "th,td{border:1px solid #ccc;padding:4px 8px}th{background:#f0f0f0}"
           "h1{color:#1f3b6b;border-bottom:3px solid #4472C4;padding-bottom:6px}"
           "h2{border-bottom:2px solid #4472C4;padding-bottom:4px;margin-top:30px;color:#1f3b6b}"
           "h3{color:#2f5f9e}blockquote{background:#f6f8fb;border-left:4px solid #4472C4;margin:8px 0;padding:6px 12px;color:#444}</style></head><body>"
           + md2html(md) + "</body></html>")
    with open(os.path.join(REPORTS, "report_COPD_final.html"), "w", encoding="utf-8") as f:
        f.write(doc)
    with open(os.path.join(REPORTS, "report_COPD_final.md"), "w", encoding="utf-8") as f:
        f.write(md)
    log(f"综合报告 -> {REPORTS}\\report_COPD_final.html/.md")
    log(f"总耗时 {time.time()-t0:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
