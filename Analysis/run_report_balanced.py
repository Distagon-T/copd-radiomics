#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法: python run_report_balanced.py   # 均衡任务整合报告
run_report_balanced.py
=======================
整合 HTML 报告：
  1) 三队列(05/02/01)各任务阳性/阴性分布
  2) J44.0 vs J44.9 均衡任务：消融(rad/rad+aq/Top100) CV + 外验 + ROC/单变量/一致性图
  3) TopK 特征筛选泛化结果(AECOPD/COPD_BCOS/J44.0vsJ44.9)
输出: E:\DICOM\2026-02-seg\report_balanced_3cohort.html
"""
import base64
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOG = open(r"E:\DICOM\2026-02-seg\report_balanced.log", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()

SEED = 42
SEG2 = r"E:\DICOM\2026-02-seg"
FIGD = os.path.join(SEG2, "figs")
os.makedirs(FIGD, exist_ok=True)
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


def load(tag):
    f, l = PATHS[tag]
    df = pd.read_csv(f)
    if l:
        lab = pd.read_csv(l)
        m = df.merge(lab[["Patient_ID", "ICD", "AECOPD", "COPD_BCOS"]], on="Patient_ID", how="inner")
    else:
        m = df
    return m.drop_duplicates(subset=["Patient_ID"])


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
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs, accs, sens, specs = [], [], [], []
    sc = StandardScaler()
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                 class_weight="balanced", max_iter=2000, random_state=SEED)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        Xte = sc.transform(X[te])
        p = clf.predict_proba(Xte)[:, 1]; pred = clf.predict(Xte)
        aucs.append(roc_auc_score(y[te], p))
        accs.append(accuracy_score(y[te], pred))
        tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
        sens.append(tp / (tp + fn) if tp + fn else 0)
        specs.append(tn / (tn + fp) if tn + fp else 0)
    return (float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(accs)),
            float(np.mean(sens)), float(np.mean(specs)))


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
    log("===== 整合报告生成 =====")
    m5 = load("05"); m2 = load("02"); m1 = load("01")
    shared = [c for c in m1.columns if c in m2.columns and c in m5.columns and c not in META]
    rad, aq = split_feats(shared)
    log(f"共享特征 {len(shared)} (rad={len(rad)}, aq={len(aq)})")

    def build(m):
        X = m[shared].apply(pd.to_numeric, errors="coerce")
        med = X.median().fillna(0)
        return X.fillna(med).fillna(0).values.astype(np.float64)
    X5 = build(m5); X2 = build(m2); X1 = build(m1)
    rad_idx = [shared.index(c) for c in rad]

    def j440(m):
        icd = m["ICD"].astype(str).str.strip()
        y = pd.Series([1 if x.startswith("J44.0") else (0 if x.startswith("J44.9") else np.nan)
                       for x in icd]).values.astype(float)
        return y

    # ---- 1) 分布 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = ["AECOPD", "COPD_BCOS", "J44.0_vs_J44.9"]
    dist = {}
    for tag, m in [("05", m5), ("02", m2), ("01", m1)]:
        dist[tag] = {}
        dist[tag]["AECOPD"] = [int((m["AECOPD"] == 1).sum()), int((m["AECOPD"] == 0).sum())]
        dist[tag]["COPD_BCOS"] = [int((m["COPD_BCOS"] == 1).sum()), int((m["COPD_BCOS"] == 0).sum())]
        yj = j440(m)
        dist[tag]["J44.0_vs_J44.9"] = [int((yj == 1).sum()), int((yj == 0).sum())]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (task, _) in zip(axes, [(t, None) for t in tasks]):
        tags = ["05", "02", "01"]
        pos = [dist[t][task][0] for t in tags]
        neg = [dist[t][task][1] for t in tags]
        ax.bar(tags, neg, color="#4c72b0", label="阴性")
        ax.bar(tags, pos, bottom=neg, color="#dd8452", label="阳性")
        for i, (p, n) in enumerate(zip(pos, neg)):
            ax.text(i, p + n + 2, f"{p}/({p+n})\n{p/(p+n):.0%}", ha="center", fontsize=8)
        ax.set_title(task)
        ax.set_ylabel("例数")
    axes[0].legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGD, "fig_bal_dist.png"), dpi=150); plt.close()
    log("分布图已保存")

    # ---- 2) J44.0 vs J44.9 分析 ----
    y5 = j440(m5); y2 = j440(m2); y1 = j440(m1)
    k5 = ~np.isnan(y5); k2 = ~np.isnan(y2); k1 = ~np.isnan(y1)
    y5 = y5[k5].astype(int); y2 = y2[k2].astype(int); y1 = y1[k1].astype(int)
    X5t = X5[k5]; X2t = X2[k2]; X1t = X1[k1]
    log(f"J44.0vsJ44.9: 05 n={len(y5)} pos={int(y5.sum())} | 02 n={len(y2)} pos={int(y2.sum())} | 01 n={len(y1)} pos={int(y1.sum())}")

    # 单变量 top
    auc = uni_auc(X5t, y5)
    order = np.argsort(-np.abs(auc - 0.5))
    top20 = [shared[i] for i in order[:20]]

    abl = []
    for tag, Xs, ys, Xe2, ye2, Xe1, ye1 in [
        ("rad", X5t[:, rad_idx], y5, X2t[:, rad_idx], y2, X1t[:, rad_idx], y1),
        ("rad+aq", X5t, y5, X2t, y2, X1t, y1)]:
        c, s, acc, se, sp = run_cv(Xs, ys)
        e2 = ext(Xs, ys, Xe2, ye2); e1 = ext(Xs, ys, Xe1, ye1)
        log(f"[J44.0vsJ44.9|{tag}] CV={c:.3f}±{s:.3f} | 02={e2:.3f} 01={e1:.3f}")
        abl.append((tag, Xs.shape[1], c, s, acc, se, sp, e2, e1))
    # Top100
    idx100 = order[:100]
    c, s, acc, se, sp = run_cv(X5t[:, idx100], y5)
    e2 = ext(X5t[:, idx100], y5, X2t[:, idx100], y2)
    e1 = ext(X5t[:, idx100], y5, X1t[:, idx100], y1)
    log(f"[J44.0vsJ44.9|Top100] CV={c:.3f}±{s:.3f} | 02={e2:.3f} 01={e1:.3f}")
    abl.append(("Top100", 100, c, s, acc, se, sp, e2, e1))

    # ROC
    from sklearn.metrics import roc_curve
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    plt.figure(figsize=(6.5, 6.5))
    base = np.linspace(0, 1, 101); tprs = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    sc = StandardScaler()
    for tr, te in skf.split(X5t, y5):
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                 class_weight="balanced", max_iter=2000, random_state=SEED)
        clf.fit(sc.fit_transform(X5t[tr]), y5[tr])
        fpr, tpr, _ = roc_curve(y5[te], clf.predict_proba(sc.transform(X5t[te]))[:, 1])
        tprs.append(np.interp(base, fpr, tpr)); tprs[-1][0] = 0
    mt = np.mean(tprs, axis=0); mt[-1] = 1
    plt.plot(base, mt, "b-", lw=2, label=f"05 CV rad+aq ({c:.3f}±{s:.3f})")
    for tag, mtx, ytx, col in [("02", X2t, y2, "#dd8452"), ("01", X1t, y1, "#55a868")]:
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                 class_weight="balanced", max_iter=2000, random_state=SEED)
        clf.fit(sc.fit_transform(X5t), y5)
        fpr, tpr, _ = roc_curve(ytx, clf.predict_proba(sc.transform(mtx))[:, 1])
        plt.plot(fpr, tpr, "-", lw=2, color=col,
                 label=f"ext 2026-{tag} ({roc_auc_score(ytx, clf.predict_proba(sc.transform(mtx))[:, 1]):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(loc="lower right")
    plt.title("J44.0 (急性下呼吸道感染) vs J44.9 (稳定)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGD, "fig_bal_roc_j440.png"), dpi=150); plt.close()

    # 单变量 top20 图
    tp = top20[::-1]
    v = [auc[shared.index(c)] for c in tp]
    plt.figure(figsize=(9, 7))
    colors = ["#d62728" if a >= 0.5 else "#1f77b4" for a in v]
    plt.barh(range(len(tp)), v, color=colors)
    plt.axvline(0.5, color="k", ls="--", lw=1)
    plt.yticks(range(len(tp)), [f[:46] for f in tp], fontsize=8)
    plt.xlabel("单变量 AUC (2026-05, J44.0 vs J44.9)"); plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGD, "fig_bal_uni_j440.png"), dpi=150); plt.close()

    # 一致性 top8
    rng = np.random.default_rng(SEED)
    ip = np.where(y5 == 1)[0]; ig = np.where(y5 == 0)[0]
    cons = []
    for c in top20[:8]:
        j = shared.index(c)
        x = X5t[:, j]
        bs = np.array([roc_auc_score(y5[np.concatenate([rng.choice(ip, len(ip), True), rng.choice(ig, len(ig), True)])],
                                     x[np.concatenate([rng.choice(ip, len(ip), True), rng.choice(ig, len(ig), True)])])
                       for _ in range(200)])
        cons.append((c, np.mean(bs), *np.percentile(bs, [2.5, 97.5])))
    cons.sort(key=lambda t: -abs(t[1] - 0.5))
    ys = np.arange(len(cons))[::-1]
    plt.figure(figsize=(8, 4.5))
    for i, (c, mu, lo, hi) in enumerate(cons):
        col = "#d62728" if mu >= 0.5 else "#1f77b4"
        plt.errorbar([mu], [ys[i]], xerr=[[mu - lo], [hi - mu]], fmt="o", color=col,
                     ecolor=col, capsize=3, ms=5)
    plt.axvline(0.5, color="k", ls="--", lw=1)
    plt.yticks(ys, [t[0][:44] for t in cons], fontsize=8)
    plt.xlabel("Bootstrap 单变量 AUC (95%CI)"); plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGD, "fig_bal_cons_j440.png"), dpi=150); plt.close()

    # ---- 2026-05 内部效果 ROC（3 任务）----
    def cv_mean_roc(X, y):
        skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        sc3 = StandardScaler()
        tprs = []; aucs = []
        for tr, te in skf2.split(X, y):
            clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                     class_weight="balanced", max_iter=2000, random_state=SEED)
            clf.fit(sc3.fit_transform(X[tr]), y[tr])
            p = clf.predict_proba(sc3.transform(X[te]))[:, 1]
            aucs.append(roc_auc_score(y[te], p))
            fpr, tpr, _ = roc_curve(y[te], p)
            tprs.append(np.interp(base, fpr, tpr)); tprs[-1][0] = 0
        mt = np.mean(tprs, axis=0); mt[-1] = 1
        return base, mt, float(np.mean(aucs))

    def roc_05_task(task):
        y = make_y05(task)
        k = ~np.isnan(y)
        return cv_mean_roc(X5[k], y[k].astype(int))

    def make_y05(task):
        icd = m5["ICD"].astype(str).str.strip()
        if task == "AECOPD":
            return m5["AECOPD"].values.astype(float)
        if task == "COPD_BCOS":
            return m5["COPD_BCOS"].fillna(0).values.astype(float)
        return np.array([1 if x.startswith("J44.0") else (0 if x.startswith("J44.9") else np.nan)
                         for x in icd]).astype(float)

    plt.figure(figsize=(15, 5))
    for i, task in enumerate(["AECOPD", "COPD_BCOS", "J44.0_vs_J44.9"]):
        ax = plt.subplot(1, 3, i + 1)
        b, mt, a = roc_05_task(task)
        ax.plot(b, mt, "b-", lw=2, label=f"2026-05 内部 CV (AUC={a:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(loc="lower right"); ax.set_title(f"2026-05 内部效果 - {task}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGD, "fig_roc_05_tasks.png"), dpi=150); plt.close()
    log("2026-05 内部 ROC 图已保存")

    # ---- 3) TopK 泛化图（复用 topk_gen 结果, 简化在此重算 AECOPD 与 J44.0）----
    def make_y(m, task):
        icd = m["ICD"].astype(str).str.strip()
        if task == "AECOPD":
            return m["AECOPD"].values.astype(float)
        if task == "COPD_BCOS":
            return m["COPD_BCOS"].fillna(0).values.astype(float)
        return np.array([1 if x.startswith("J44.0") else (0 if x.startswith("J44.9") else np.nan)
                         for x in icd]).astype(float)

    plt.figure(figsize=(8, 5))
    for task, col in [("AECOPD", "#4c72b0"), ("COPD_BCOS", "#dd8452"), ("J44.0_vs_J44.9", "#55a868")]:
        y5t = make_y(m5, task); y2t = make_y(m2, task)
        k5t = ~np.isnan(y5t); k2t = ~np.isnan(y2t)
        y5t = y5t[k5t].astype(int); y2t = y2t[k2t].astype(int)
        X5q = X5[k5t]; X2q = X2[k2t]
        au = uni_auc(X5q, y5t)
        o = np.argsort(-np.abs(au - 0.5))
        extv = []
        for K in [20, 50, 100, 200]:
            idx = o[:K]
            sc2 = StandardScaler()
            clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                     class_weight="balanced", max_iter=2000, random_state=SEED)
            clf.fit(sc2.fit_transform(X5q[:, idx]), y5t)
            extv.append(roc_auc_score(y2t, clf.predict_proba(sc2.transform(X2q[:, idx]))[:, 1]))
        plt.plot([20, 50, 100, 200], extv, "o-", color=col, label=f"{task} -> 02")
    plt.axhline(0.5, color="k", ls="--", lw=0.8)
    plt.xlabel("TopK 特征数"); plt.ylabel("2026-02 外部 AUC")
    plt.ylim(0.3, 0.7); plt.legend()
    plt.title("TopK 特征筛选 -> 2026-02 外验 AUC")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGD, "fig_bal_topk.png"), dpi=150); plt.close()
    log("TopK 图已保存")

    # ---- HTML ----
    L = []
    L.append("# 三队列均衡任务 + 泛化报告")
    L.append("> 2026-08-28 | 训练 2026-05 → 外验 2026-01/2026-02 | 共享特征 %d" % len(shared))
    L.append("")
    L.append("## 1. 三队列阳性/阴性分布")
    L.append("| 任务 | 2026-05 | 2026-02 | 2026-01 |")
    L.append("|---|---|---|---|")
    for task, label in [("AECOPD", "AECOPD (急性加重)"),
                        ("COPD_BCOS", "COPD_BCOS (合并支扩)"),
                        ("J44.0_vs_J44.9", "J44.0 vs J44.9 (感染 vs 稳定)")]:
        cells = []
        for tag in ["05", "02", "01"]:
            p, n = dist[tag][task]
            cells.append(f"{p}/{n} ({p/(p+n):.0%})")
        L.append(f"| {label} | {' | '.join(cells)} |")
    L.append("")
    L.append("![分布](figs/fig_bal_dist.png)")
    L.append("")
    L.append("## 2. J44.0 vs J44.9（急性下呼吸道感染 vs 稳定未特指）")
    L.append("| 特征集 | 特征数 | 05 CV AUC | Acc | Sens | Spec | 02 外验 | 01 外验 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for tag, n, c, s, acc, se, sp, e2, e1 in abl:
        L.append(f"| {tag} | {n} | {c:.3f} ± {s:.3f} | {acc:.3f} | {se:.3f} | {sp:.3f} | {e2:.3f} | {e1:.3f} |")
    L.append("")
    L.append("![ROC](figs/fig_bal_roc_j440.png)")
    L.append("")
    L.append("### 单变量 Top20（2026-05）")
    L.append("| 特征 | AUC |")
    L.append("|---|---|")
    for c, v in zip(top20, [auc[shared.index(x)] for x in top20]):
        L.append(f"| {c} | {v:.3f} |")
    L.append("")
    L.append("![单变量](figs/fig_bal_uni_j440.png)")
    L.append("")
    L.append("![一致性](figs/fig_bal_cons_j440.png)")
    L.append("")
    L.append("### 2026-05 内部效果 ROC（三个任务，5 折 CV）")
    L.append("![2026-05 内部ROC](figs/fig_roc_05_tasks.png)")
    L.append("")
    L.append("## 3. TopK 特征筛选 -> 2026-02 外验泛化")
    L.append("![TopK](figs/fig_bal_topk.png)")
    L.append("")

    # Bootstrap + 平衡 结果（读取已生成 CSV）
    bc = pd.read_csv(os.path.join(SEG2, "bootstrap_balance_results.csv"))
    bc["test"] = bc["test"].astype(str).str.zfill(2)
    L.append("## 4. Bootstrap 95%CI + 下采样平衡")
    L.append("| 任务 | 训练方式 | 2026-02 外验 AUC (95%CI) | 2026-01 外验 AUC (95%CI) |")
    L.append("|---|---|---|---|")
    for task, tn in [("AECOPD", "AECOPD"), ("COPD_BCOS", "COPD_BCOS"), ("J44.0_vs_J44.9", "J44.0 vs J44.9")]:
        for mode, mn in [("full", "全量"), ("balanced", "下采样平衡")]:
            r02 = bc[(bc["task"] == task) & (bc["mode"] == mode) & (bc["test"] == "02")].iloc[0]
            r01 = bc[(bc["task"] == task) & (bc["mode"] == mode) & (bc["test"] == "01")].iloc[0]
            L.append(f"| {tn} | {mn} (n={int(r02['train_n'])}) | "
                     f"{r02['auc']:.3f} [{r02['ci_lo']:.3f},{r02['ci_hi']:.3f}] | "
                     f"{r01['auc']:.3f} [{r01['ci_lo']:.3f},{r01['ci_hi']:.3f}] |")
    L.append("")
    L.append("![Bootstrap森林图](figs/fig_bal_boot_ci.png)")
    L.append("")
    L.append("## 5. 结论")
    L.append("- 三队列中 **J44.0 vs J44.9** 是阳性率最接近均衡的任务（2026-02 达 40%），但 2026-05/01 阳性仍偏少(27%)。")
    L.append("- J44.0 vs J44.9 在 2026-05 内部 CV 可达 ~0.7-0.78（Top100/Top20），但外验 02/01 仍 ~0.5-0.58，跨序列不泛化。")
    L.append("- AECOPD/COPD_BCOS 的 TopK 筛选对 **2026-01 外验有改善**（0.68-0.70），但对 **2026-02 基本无效**（~0.5）。")
    L.append("- 若要做均衡的 AECOPD 二分类，建议补充稳定期(J44.9)慢阻肺病人，使阳性率降至 50% 附近。")
    L.append("")
    L.append("### Bootstrap/平衡补充解读")
    L.append("- **J44.0 vs J44.9 → 2026-02**：外验 AUC 0.624，95%CI **[0.510, 0.720] 下界 > 0.5**，是唯一统计上显著优于随机的跨序列任务（虽较弱）。")
    L.append("- COPD_BCOS/AECOPD 的 2026-02 外验 CI 均包含 0.5（不显著）；2026-01 CI 极宽（阳性太少）。")
    L.append("- 下采样平衡对 2026-02 略有帮助（AECOPD 0.525→0.550、COPD_BCOS 0.554→0.590），对 2026-01 反而变差，且 CI 都很宽，需谨慎解读。")
    L.append("")

    # ---- 5) aq 融合最优模型 ----
    import shutil
    fb = pd.read_csv(os.path.join(SEG2, "..", "reports", "fusion_boot_ci.csv"))
    fb["task"] = fb["task"].map({"AECOPD": "AECOPD", "COPD_BCOS": "COPD_BCOS",
                                 "J44.0_vs_J44.9": "J44.0 vs J44.9"})
    src_fig = os.path.join(SEG2, "..", "reports", "figs", "fig_fusion_boot_ci.png")
    if os.path.exists(src_fig):
        shutil.copy(src_fig, os.path.join(FIGD, "fig_fusion_boot_ci.png"))
    L.append("## 5. aq 融合最优模型（radTop100 + aqTop20，bootstrap 95%CI）")
    L.append("| 任务 | 2026-02 外验 AUC (95%CI) | 2026-01 外验 AUC (95%CI) |")
    L.append("|---|---|---|")
    for r in fb.itertuples():
        L.append(f"| {r.task} | {r.ext_02:.3f} [{r.ci02_lo:.3f},{r.ci02_hi:.3f}] | "
                 f"{r.ext_01:.3f} [{r.ci01_lo:.3f},{r.ci01_hi:.3f}] |")
    L.append("")
    L.append("![aq融合bootstrap](figs/fig_fusion_boot_ci.png)")
    L.append("")
    L.append("### aq 融合解读")
    L.append("- **COPD_BCOS**：`radTop100 + aqTop20` 融合外验 02=0.603 / 01=0.693，优于 rad 全量(0.541/0.456)——aq 精选特征(TD_fwhm/LAA950/WA_pct/Vessel)与 rad 互补，**aq 不是没用，是需要筛选**。")
    L.append("- 但 02 的 CI [0.481,0.722] 仍含 0.5；01 CI [0.537,0.855] 下界 >0.5（但 01 仅 6 阳，样本小）。")
    L.append("- AECOPD：aq 融合帮助有限；J44.0 vs J44.9：rad 全量仍最佳（02=0.646）。")
    md = "\n".join(L) + "\n"

    def md2html(s):
        h = ["<h1>三队列均衡任务 + 泛化报告</h1>"]
        lines = s.splitlines()
        in_tab = False
        for ln in lines[2:]:
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
                full = os.path.join(SEG2, path)
                if os.path.exists(full):
                    h.append(f'<img src="data:image/png;base64,{b64(full)}" '
                             f'style="max-width:95%;height:auto;display:block;margin:10px auto;"/>')
                    h.append(f'<p style="text-align:center;color:#555">{cap}</p>')
            elif ln.strip() == "":
                if in_tab:
                    h.append("</tbody></table>"); in_tab = False
            else:
                if in_tab:
                    h.append("</tbody></table>"); in_tab = False
                if not (ln.startswith("*") and ln.endswith("*")):
                    h.append(f"<p>{ln}</p>")
        if in_tab:
            h.append("</tbody></table>")
        return "\n".join(h)

    body = md2html(md)
    doc = ("<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
           "<title>三队列均衡任务 + 泛化报告</title>"
           "<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:1000px;margin:20px auto;"
           "padding:0 20px;color:#222;line-height:1.6}table{border-collapse:collapse;margin:10px 0;font-size:0.92em}"
           "th,td{border:1px solid #ccc;padding:4px 8px}th{background:#f0f0f0}"
           "h2{border-bottom:2px solid #4472C4;padding-bottom:4px;margin-top:28px}</style></head><body>"
           + body + "</body></html>")
    out = os.path.join(SEG2, "report_balanced_3cohort.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    log(f"HTML -> {out}")
    with open(os.path.join(SEG2, "report_balanced_3cohort.md"), "w", encoding="utf-8") as f:
        f.write(md)
    log("MD -> report_balanced_3cohort.md")
    log(f"总耗时 {time.time()-t0:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
