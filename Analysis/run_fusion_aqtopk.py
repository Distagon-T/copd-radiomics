#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法: python run_fusion_aqtopk.py   # rad+aqTopK 融合对比
run_fusion_aqtopk.py
=====================
rad 全量 + aq-TopK 融合 对比（重点看 COPD_BCOS 能否提升外验）。
对每个任务：
  rad全量 | rad全量+aqTop10/20/50 | radTop100+aqTop20 | aqTop20
  输出 05 CV + 外验 02/01
结果写 E:\DICOM\reports\report_aq_fusion.html/.md + figs
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
LOG = open(r"E:\DICOM\2026-02-seg\aq_fusion.log", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()

META = {"Patient_ID", "PatientID", "PatientID_raw", "Patient_ID_long", "CT_Series",
        "patient_id", "ICD", "main_diagnosis", "AECOPD", "COPD_BCOS", "患者id"}
PATHS = {
    "05": (r"E:\DICOM\2026-05-seg\2026-05-integrated_radiomics_aq.csv",
           r"E:\DICOM\2026-05-seg\labels_ae_bcos_2026_05.csv"),
    "02": (r"E:\DICOM\2026-02-seg\2026-02-integrated_radiomics_aq.csv",
           r"E:\DICOM\2026-02-seg\labels_ae_bcos_2026_02.csv"),
    "01": (r"E:\DICOM\2026-01-seg\2026-01-integrated_radiomics_aq.csv", None),
}
AQ_PREFIX = ("TD_", "blur_", "wall_", "WA_", "Din_", "Dout_", "mean_",
             "Pi10", "Vessel_", "Lobe_", "Lung_", "Airway_", "PA_",
             "Diaphragm_", "pca_", "RV_", "LV_", "CAC_")
RAD_EXTRA = ("Lobe_", "Lung_", "Airway_", "PA_", "Diaphragm_", "heart",
             "aorta", "trachea", "pulmonary_artery")


def load(tag):
    f, l = PATHS[tag]
    df = pd.read_csv(f)
    if l:
        lab = pd.read_csv(l)
        m = df.merge(lab[["Patient_ID", "ICD", "AECOPD", "COPD_BCOS"]], on="Patient_ID", how="inner")
    else:
        m = df
    return m.drop_duplicates(subset=["Patient_ID"])


def make_y(m, task):
    icd = m["ICD"].astype(str).str.strip()
    if task == "AECOPD":
        return m["AECOPD"].values.astype(float)
    if task == "COPD_BCOS":
        return m["COPD_BCOS"].fillna(0).values.astype(float)
    return np.array([1 if x.startswith("J44.0") else (0 if x.startswith("J44.9") else np.nan)
                     for x in icd]).astype(float)


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


def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()


def main():
    t0 = time.time()
    log("===== rad + aq-TopK 融合 =====")
    m5 = load("05"); m2 = load("02"); m1 = load("01")
    shared = [c for c in m1.columns if c in m2.columns and c in m5.columns and c not in META]
    rad, aq = split_feats(shared)
    log(f"共享 {len(shared)}: rad={len(rad)} aq={len(aq)}")

    def build(m):
        X = m[shared].apply(pd.to_numeric, errors="coerce")
        med = X.median().fillna(0)
        return X.fillna(med).fillna(0).values.astype(np.float64)
    X5 = build(m5); X2 = build(m2); X1 = build(m1)
    rad_idx = np.array([shared.index(c) for c in rad])
    aq_idx = np.array([shared.index(c) for c in aq])

    results = []
    for task in ["AECOPD", "COPD_BCOS", "J44.0_vs_J44.9"]:
        y5 = make_y(m5, task); y2 = make_y(m2, task); y1 = make_y(m1, task)
        k5 = ~np.isnan(y5); k2 = ~np.isnan(y2); k1 = ~np.isnan(y1)
        y5 = y5[k5].astype(int); y2 = y2[k2].astype(int); y1 = y1[k1].astype(int)
        X5r = X5[k5][:, rad_idx]; X2r = X2[k2][:, rad_idx]; X1r = X1[k1][:, rad_idx]
        X5q = X5[k5][:, aq_idx]; X2q = X2[k2][:, aq_idx]; X1q = X1[k1][:, aq_idx]

        auq = uni_auc(X5q, y5)
        oq = np.argsort(-np.abs(auq - 0.5))
        aur = uni_auc(X5r, y5)
        or_ = np.argsort(-np.abs(aur - 0.5))
        log(f"\n##### {task} (05 pos={int(y5.sum())}/{len(y5)}) #####")

        configs = {
            "rad全量": (X5r, X2r, X1r),
            "rad全量+aqTop10": (np.hstack([X5r, X5q[:, oq[:10]]]),
                                np.hstack([X2r, X2q[:, oq[:10]]]),
                                np.hstack([X1r, X1q[:, oq[:10]]])),
            "rad全量+aqTop20": (np.hstack([X5r, X5q[:, oq[:20]]]),
                                np.hstack([X2r, X2q[:, oq[:20]]]),
                                np.hstack([X1r, X1q[:, oq[:20]]])),
            "rad全量+aqTop50": (np.hstack([X5r, X5q[:, oq[:50]]]),
                                np.hstack([X2r, X2q[:, oq[:50]]]),
                                np.hstack([X1r, X1q[:, oq[:50]]])),
            "radTop100+aqTop20": (np.hstack([X5r[:, or_[:100]], X5q[:, oq[:20]]]),
                                  np.hstack([X2r[:, or_[:100]], X2q[:, oq[:20]]]),
                                  np.hstack([X1r[:, or_[:100]], X1q[:, oq[:20]]])),
            "aqTop20": (X5q[:, oq[:20]], X2q[:, oq[:20]], X1q[:, oq[:20]]),
        }
        for name, (X5c, X2c, X1c) in configs.items():
            c, s = run_cv(X5c, y5)
            e2 = ext(X5c, y5, X2c, y2); e1 = ext(X5c, y5, X1c, y1)
            log(f"  [{name}] n={X5c.shape[1]} CV={c:.3f}±{s:.3f} | 02={e2:.3f} 01={e1:.3f}")
            results.append({"task": task, "config": name, "n_feat": X5c.shape[1],
                            "cv_auc": c, "cv_std": s, "ext_02": e2, "ext_01": e1})

    pd.DataFrame(results).to_csv(os.path.join(REPORTS, "aq_fusion_results.csv"),
                                 index=False, encoding="utf-8-sig")

    # 报告图：各 config 外验 02/01 条状图（COPD_BCOS 单独突出）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(results)
    for task in ["AECOPD", "COPD_BCOS", "J44.0_vs_J44.9"]:
        sub = df[df["task"] == task]
        cfg = sub["config"].tolist()[::-1]
        v02 = sub["ext_02"].tolist()[::-1]
        v01 = sub["ext_01"].tolist()[::-1]
        y = np.arange(len(cfg))
        plt.figure(figsize=(9, 5))
        plt.barh(y - 0.2, v02, height=0.4, color="#4c72b0", label="2026-02 外验")
        plt.barh(y + 0.2, v01, height=0.4, color="#dd8452", label="2026-01 外验")
        plt.axvline(0.5, color="k", ls="--", lw=1)
        plt.yticks(y, cfg, fontsize=8)
        plt.xlabel("外部 AUC"); plt.xlim(0.3, 0.8)
        plt.legend(); plt.title(f"{task} - rad+aqTopK 融合外验")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGD, f"fig_aq_fusion_{task}.png"), dpi=150); plt.close()

    # 报告
    L = ["# rad + aq-TopK 融合 外部验证对比",
         f"> 2026-08-28 | 训练 2026-05 → 外验 2026-01/2026-02 | 共享特征 {len(shared)} (rad {len(rad)} + aq {len(aq)})",
         ""]
    for task in ["AECOPD", "COPD_BCOS", "J44.0_vs_J44.9"]:
        sub = df[df["task"] == task]
        L.append(f"## {task}")
        L.append("| 特征配置 | 特征数 | 05 CV AUC | 02 外验 | 01 外验 |")
        L.append("|---|---|---|---|---|")
        for r in sub.itertuples():
            L.append(f"| {r.config} | {r.n_feat} | {r.cv_auc:.3f}±{r.cv_std:.3f} | "
                     f"{r.ext_02:.3f} | {r.ext_01:.3f} |")
        L.append("")
        L.append(f"![{task}](figs/fig_aq_fusion_{task}.png)")
        L.append("")
    L.append("## 结论")
    L.append("- **COPD_BCOS**：rad 全量 + aq-TopK 融合是重点。若融合后 2026-01/02 外验比 rad 全量提升，说明 aq 提供补充信息。")
    L.append("- aq 单变量在 COPD_BCOS 上 60/106 有信号（TD_fwhm/LAA950/WA_pct/Vessel），与 rad 互补。")
    md = "\n".join(L) + "\n"

    def md2html(s):
        h = ["<h1>rad + aq-TopK 融合 外部验证对比</h1>"]
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

    doc = ("<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'><title>rad+aq融合</title>"
           "<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:1000px;margin:20px auto;padding:0 20px;"
           "color:#222;line-height:1.6}table{border-collapse:collapse;margin:10px 0;font-size:0.92em}"
           "th,td{border:1px solid #ccc;padding:4px 8px}th{background:#f0f0f0}"
           "h2{border-bottom:2px solid #4472C4;padding-bottom:4px;margin-top:28px}</style></head><body>"
           + md2html(md) + "</body></html>")
    with open(os.path.join(REPORTS, "report_aq_fusion.html"), "w", encoding="utf-8") as f:
        f.write(doc)
    with open(os.path.join(REPORTS, "report_aq_fusion.md"), "w", encoding="utf-8") as f:
        f.write(md)
    log(f"报告 -> {REPORTS}\\report_aq_fusion.html/.md")
    log(f"总耗时 {time.time()-t0:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
