#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法: python run_3cohort_aligned.py   # 对齐后三队列外部验证 + 临床单变量AUC
run_3cohort_aligned.py
======================
用对齐后的干净表(2026-01标准, 2273特征, 密度一致)重跑三队列外部验证：
  训练 2026-05 -> 外验 2026-01 / 2026-02
  任务: AECOPD / COPD_BCOS / J44.0_vs_J44.9
  配置: rad+aq 全量 | radTop100+aqTop20 融合
重点：临床影像特征(LAA/Perc15/Pi10/WA/TD/wall/blur/Vessel) 单变量 AUC
输出: E:\DICOM\reports\report_aligned3cohort.html/.md + clinical_uni.csv
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
FIGD = os.path.join(REPORTS, "figs")
os.makedirs(FIGD, exist_ok=True)
LOG = open(r"E:\DICOM\reports\aligned3cohort.log", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()

META = {"Patient_ID", "PatientID", "PatientID_raw", "Patient_ID_long", "CT_Series",
        "patient_id", "ICD", "main_diagnosis", "AECOPD", "COPD_BCOS", "患者id"}
FILES = {
    "05": r"E:\DICOM\2026-05-seg\2026-05-integrated_radiomics_aq_aligned01_labeled.csv",
    "02": r"E:\DICOM\2026-02-seg\2026-02-integrated_radiomics_aq_aligned01_labeled.csv",
    "01": r"E:\DICOM\2026-01-seg\2026-01-integrated_radiomics_aq.csv",
}
AQ_PREFIX = ("TD_", "blur_", "wall_", "WA_", "Din_", "Dout_", "mean_",
             "Pi10", "Vessel_", "Lobe_", "Lung_", "Airway_", "PA_",
             "Diaphragm_", "pca_", "RV_", "LV_", "CAC_")
RAD_EXTRA = ("Lobe_", "Lung_", "Airway_", "PA_", "Diaphragm_", "heart",
             "aorta", "trachea", "pulmonary_artery")
CLIN_KW = ("LAA950", "Perc15", "Pi10", "WA_pct", "TD_", "wall_", "blur_",
           "Vessel_", "Lobe_", "Lung_", "Din_", "Dout_", "tortuosity",
           "n_branches", "pruning", "max_generation")


def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()


def uni_auc(X, y):
    from scipy.stats import rankdata
    pos = y == 1
    np_ = int(pos.sum()); nn = int((~pos).sum())
    Z = np.vstack([X[pos], X[~pos]])
    R = rankdata(Z, axis=0)
    Rpos = R[:np_].sum(0)
    return (Rpos - np_ * (np_ + 1) / 2) / (np_ * nn)


def run_cv(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    sc = StandardScaler()
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                 class_weight="balanced", max_iter=2000, random_state=SEED)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return np.mean(aucs), np.std(aucs)


def ext(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                             class_weight="balanced", max_iter=2000, random_state=SEED)
    clf.fit(sc.fit_transform(Xtr), ytr)
    return roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1])


def main():
    t0 = time.time()
    log("===== 对齐后干净表：三队列外部验证 + 临床单变量 =====")
    m5 = pd.read_csv(FILES["05"]).drop_duplicates("Patient_ID")
    m2 = pd.read_csv(FILES["02"]).drop_duplicates("Patient_ID")
    m1 = pd.read_csv(FILES["01"]).drop_duplicates("Patient_ID")
    shared = [c for c in m1.columns if c in m2.columns and c in m5.columns and c not in META]
    log(f"共享特征(对齐后): {len(shared)}; 样本 05={len(m5)} 02={len(m2)} 01={len(m1)}")
    clin = [c for c in shared if any(k in c for k in CLIN_KW)]
    log(f"临床影像特征: {len(clin)}")

    def build(m):
        X = m[shared].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median().fillna(0)).fillna(0).values.astype(np.float64)
        return X
    X5 = build(m5); X2 = build(m2); X1 = build(m1)

    def make_y(m, task):
        icd = m["ICD"].astype(str).str.strip()
        if task == "AECOPD":
            return m["AECOPD"].values.astype(float)
        if task == "COPD_BCOS":
            return m["COPD_BCOS"].fillna(0).values.astype(float)
        return np.array([1 if x.startswith("J44.0") else (0 if x.startswith("J44.9") else np.nan)
                         for x in icd]).astype(float)

    # 单变量（临床特征）
    clin_rows = []
    for task in ["AECOPD", "COPD_BCOS", "J44.0_vs_J44.9"]:
        y5 = make_y(m5, task)
        k = ~np.isnan(y5); y5 = y5[k].astype(int)
        X5t = X5[k]
        ci = [shared.index(c) for c in clin]
        auc = uni_auc(X5t[:, ci], y5)
        for j, c in enumerate(clin):
            clin_rows.append({"task": task, "feature": c, "auc": auc[j],
                              "dev": abs(auc[j] - 0.5)})
    cd = pd.DataFrame(clin_rows).sort_values(["task", "dev"], ascending=[True, False])
    cd.to_csv(os.path.join(REPORTS, "clinical_uni_aligned.csv"), index=False, encoding="utf-8-sig")
    log("临床单变量已写 clinical_uni_aligned.csv")

    for task in ["AECOPD", "COPD_BCOS", "J44.0_vs_J44.9"]:
        log(f"\n##### {task} 临床单变量 Top12 #####")
        sub = cd[cd["task"] == task].head(12)
        for r in sub.itertuples():
            log(f"  {r.feature[:46]:<48} AUC={r.auc:.3f}")

    # 外部验证
    log("\n===== 外部验证（训练 2026-05）=====")
    for task in ["AECOPD", "COPD_BCOS", "J44.0_vs_J44.9"]:
        y5 = make_y(m5, task); y2 = make_y(m2, task); y1 = make_y(m1, task)
        k5 = ~np.isnan(y5); k2 = ~np.isnan(y2); k1 = ~np.isnan(y1)
        y5 = y5[k5].astype(int); y2 = y2[k2].astype(int); y1 = y1[k1].astype(int)
        X5t = X5[k5]; X2t = X2[k2]; X1t = X1[k1]
        c, s = run_cv(X5t, y5)
        e2 = ext(X5t, y5, X2t, y2); e1 = ext(X5t, y5, X1t, y1)
        log(f"[{task}|rad+aq 全量] 05CV={c:.3f}±{s:.3f} | 02={e2:.3f} 01={e1:.3f}")
        # radTop100+aqTop20 融合
        aur = uni_auc(X5t[:, [shared.index(c) for c in shared if "::" in c]], y5) if False else None
        rad = [c for c in shared if "::" in c or c.startswith(RAD_EXTRA)]
        aq = [c for c in shared if c not in rad]
        ri = np.array([shared.index(c) for c in rad]); qi = np.array([shared.index(c) for c in aq])
        aur = uni_auc(X5t[:, ri], y5); auq = uni_auc(X5t[:, qi], y5)
        o_r = np.argsort(-np.abs(aur - 0.5))[:100]
        o_q = np.argsort(-np.abs(auq - 0.5))[:20]
        X5c = np.hstack([X5t[:, ri[o_r]], X5t[:, qi[o_q]]])
        X2c = np.hstack([X2t[:, ri[o_r]], X2t[:, qi[o_q]]])
        X1c = np.hstack([X1t[:, ri[o_r]], X1t[:, qi[o_q]]])
        c2, s2 = run_cv(X5c, y5)
        e2b = ext(X5c, y5, X2c, y2); e1b = ext(X5c, y5, X1c, y1)
        log(f"[{task}|radTop100+aqTop20] 05CV={c2:.3f}±{s2:.3f} | 02={e2b:.3f} 01={e1b:.3f}")

    # 图：临床单变量
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for i, task in enumerate(["AECOPD", "COPD_BCOS", "J44.0_vs_J44.9"]):
        ax = axes[i]
        sub = cd[cd["task"] == task].nlargest(12, "dev").iloc[::-1]
        v = sub["auc"].tolist()
        cols = ["#d62728" if a >= 0.5 else "#1f77b4" for a in v]
        ax.barh(range(len(sub)), v, color=cols)
        ax.axvline(0.5, color="k", ls="--", lw=1)
        ax.set_yticks(range(len(sub))); ax.set_yticklabels([f[:38] for f in sub["feature"]], fontsize=7)
        ax.set_xlim(0.25, 0.75); ax.set_title(f"{task} 临床单变量AUC(2026-05)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGD, "fig_clinical_uni_aligned.png"), dpi=150); plt.close()
    log("临床单变量图已保存")

    # 报告
    L = ["# 对齐后三队列外部验证 + 临床单变量",
         f"> 2026-08-29 | 训练 2026-05 → 外验 2026-01/02 | 对齐后共享特征 {len(shared)}（2026-01 标准, 密度一致）",
         ""]
    for task in ["AECOPD", "COPD_BCOS", "J44.0_vs_J44.9"]:
        L.append(f"## {task} 临床影像单变量 Top（2026-05）")
        L.append("| 特征 | AUC | | 特征 | AUC |")
        L.append("|---|---|---|---|---|")
        sub = cd[cd["task"] == task].head(12)
        half = 6
        rows = sub.iloc[:half]
        for i in range(half):
            r1 = rows.iloc[i]
            if i < len(sub):
                r2 = sub.iloc[half + i] if half + i < len(sub) else None
                if r2 is not None:
                    L.append(f"| {r1.feature} | {r1.auc:.3f} | | {r2.feature} | {r2.auc:.3f} |")
                else:
                    L.append(f"| {r1.feature} | {r1.auc:.3f} | | | |")
        L.append("")
    L.append("## 外部验证汇总")
    L.append("![临床单变量](figs/fig_clinical_uni_aligned.png)")
    md = "\n".join(L) + "\n"

    def md2html(s):
        h = ["<h1>对齐后三队列外部验证 + 临床单变量</h1>"]
        lines = s.splitlines(); in_tab = False
        for ln in lines[1:]:
            if ln.startswith("## "):
                h.append(f"<h2>{ln[3:]}</h2>")
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
                if not (ln.startswith("*") and ln.endswith("*")):
                    h.append(f"<p>{ln}</p>")
        if in_tab: h.append("</tbody></table>")
        return "\n".join(h)

    doc = ("<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'><title>对齐后三队列验证</title>"
           "<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:1000px;margin:20px auto;padding:0 20px;color:#222;line-height:1.6}"
           "table{border-collapse:collapse;margin:10px 0;font-size:0.9em}th,td{border:1px solid #ccc;padding:4px 8px}th{background:#f0f0f0}"
           "h2{border-bottom:2px solid #4472C4;padding-bottom:4px;margin-top:28px}</style></head><body>" + md2html(md) + "</body></html>")
    with open(os.path.join(REPORTS, "report_aligned3cohort.html"), "w", encoding="utf-8") as f:
        f.write(doc)
    with open(os.path.join(REPORTS, "report_aligned3cohort.md"), "w", encoding="utf-8") as f:
        f.write(md)
    log(f"\n报告 -> {REPORTS}\\report_aligned3cohort.html/.md")
    log(f"总耗时 {time.time()-t0:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
