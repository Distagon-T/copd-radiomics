#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BCOS 研究收尾: 特征族分类 + bootstrap一致性 + 图"""
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

SEG = r"E:\DICOM\2026-05-seg"
FIG = os.path.join(SEG, "figs")
os.makedirs(FIG, exist_ok=True)

uni = pd.read_csv(os.path.join(SEG, "bcos_screening_univariate_top.csv"))

# 1) 特征族分类
def fam(c):
    low = c.lower()
    if "::" in c:
        roi, rest = c.split("::", 1)
        filt = "original" if "original" in rest else ("wavelet" if "wavelet" in rest
                                                      else ("log-sigma" if "log-sigma" in rest else "other"))
        cls = "glcm" if "glcm" in low else ("glrlm" if "glrlm" in low else
                                            ("glszm" if "glszm" in low else ("ngtdm" if "ngtdm" in low else
                                            ("firstorder" if "firstorder" in low else "shape"))))
        return f"{roi.split('_')[0] if roi not in ('lung_vessels','heart_myocardium','pulmonary_artery','aorta','trachea') else roi}::{filt}::{cls}"
    low2 = low
    for k in ["vessel", "fractal", "bv", "pi10", "tortuosity", "td", "blur", "wall", "wa_", "lobe"]:
        if k in low2:
            return k
    return "other"

uni["family"] = uni["feature"].map(fam)
top100 = uni.head(100)
c = Counter(top100["family"])
print("=== Top100 特征族分布 ===")
for k, v in c.most_common(20):
    print(f"  {k}: {v}")
print("\n=== Top100 里 'lung_vessels/vessel/其他新特征' ===")
for _, r in uni.head(100).iterrows():
    if any(k in r["feature"].lower() for k in ["vessel", "fractal", "bv", "pi10",
                                               "tortuosity", "td", "blur", "wall", "wa_", "lobe", "airway"]):
        print(f"  {r['feature']}  AUC={r['auc']:.3f} p={r['p_mwu']:.2g}")

# 2) bootstrap 一致性 top10
tr = pd.read_csv(os.path.join(SEG, "2026-05-integrated_radiomics_aq.csv"))
lab = pd.read_csv(os.path.join(SEG, "labels_bcos_2026_05.csv"))
tr["_nid"] = tr["PatientID"].astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")
lab["_nid"] = lab["patient_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")
m = tr.merge(lab[["_nid", "COPD_BCOS"]], on="_nid", how="inner").drop_duplicates(subset=["_nid"])
y = m["COPD_BCOS"].astype(int).values
rng = np.random.default_rng(42)
ip = np.where(y == 1)[0]; ig = np.where(y == 0)[0]
cons = []
for c in uni.head(10)["feature"]:
    x = pd.to_numeric(m[c], errors="coerce").fillna(pd.to_numeric(m[c], errors="coerce").median()).values
    bs = []
    for _ in range(200):
        si = np.concatenate([rng.choice(ip, len(ip), True), rng.choice(ig, len(ig), True)])
        bs.append(roc_auc_score(y[si], x[si]))
    pt = uni.loc[uni["feature"] == c, "auc"].values[0]
    bs = np.array(bs)
    cons.append((c, np.mean(bs), *np.percentile(bs, [2.5, 97.5]),
                 np.mean(bs > 0.5) if pt >= 0.5 else np.mean(bs < 0.5)))
print("\n=== top10 bootstrap 一致性 ===")
for c, mu, lo, hi, stab in cons:
    print(f"  {c[:52]:54s} AUC={mu:.3f} CI[{lo:.3f},{hi:.3f}] 稳定={stab:.0%}")

# 3) 图
# ROC 需要重新训，这里只画单变量AUC柱状图 + 一致性森林图
top = uni.head(20).iloc[::-1]
plt.figure(figsize=(9, 7))
colors = ["#d62728" if a >= 0.5 else "#1f77b4" for a in top["auc"]]
plt.barh(range(len(top)), top["auc"], color=colors)
plt.axvline(0.5, color="k", ls="--", lw=1)
plt.yticks(range(len(top)), [f[:46] for f in top["feature"]], fontsize=8)
plt.xlabel("Univariate AUC (train 2026-05)"); plt.xlim(0, 1)
plt.title("COPD+BCOS feature screening top20")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_bcos_univariate_auc.png"), dpi=150); plt.close()

cons = sorted(cons, key=lambda t: -abs(t[1] - 0.5))
ys = np.arange(len(cons))[::-1]
plt.figure(figsize=(8, 4.5))
for i, (c, mu, lo, hi, stab) in enumerate(cons):
    col = "#d62728" if mu >= 0.5 else "#1f77b4"
    plt.errorbar([mu], [ys[i]], xerr=[[mu - lo], [hi - mu]], fmt="o",
                 color=col, ecolor=col, capsize=3, ms=5)
    plt.text(0.5, ys[i] + 0.2, f"{stab:.0%}", ha="center", fontsize=7, color="gray")
plt.axvline(0.5, color="k", ls="--", lw=1)
plt.yticks(ys, [t[0][:44] for t in cons], fontsize=8)
plt.xlabel("Bootstrap univariate AUC (95%CI)"); plt.xlim(0, 1)
plt.title("COPD+BCOS top10 consistency (bootstrap n=200)")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_bcos_consistency.png"), dpi=150); plt.close()
print("\n图已保存: fig_bcos_univariate_auc.png, fig_bcos_consistency.png")
