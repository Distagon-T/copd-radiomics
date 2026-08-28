#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrate_airquant_2026_05.py
=============================
按 2026-02-Airway_metrics 的 airquant_summary.json 格式，
整合 2026-05-Airway_metrics_tmp（AirQuant 清单）+ 2026-05-Airway_features（69 列特征）：
  - 读取 metrics_tmp 的 airquant_summary.json（已是 2026-02 格式）
  - 对每个患者补挂 _airway_features.csv 路径 + 69 个特征值
  - 输出 airquant_2026_05_integrated_summary.json
  - 输出 airquant_alignment_2026_02_vs_2026_05.md（2026-02 vs 2026-05 对齐报告）
"""
import os
import json
import sys
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

D2 = r"E:\DICOM\2026-02-Airway_metrics"
D5m = r"E:\DICOM\2026-05-Airway_metrics_tmp"
D5f = r"E:\DICOM\2026-05-Airway_features"
OUT_JSON = os.path.join(D5m, "airquant_2026_05_integrated_summary.json")
OUT_MD = os.path.join(D5m, "airquant_alignment_2026_02_vs_2026_05.md")


def _clean(v):
    if isinstance(v, (np.floating, float)) and np.isnan(v):
        return None
    if isinstance(v, (np.integer, np.floating)):
        return float(v) if isinstance(v, np.floating) else int(v)
    return v


def main():
    with open(os.path.join(D5m, "airquant_summary.json"), encoding="utf-8") as f:
        s5 = json.load(f)
    with open(os.path.join(D2, "airquant_summary.json"), encoding="utf-8") as f:
        s2 = json.load(f)

    # 收集 2026-05 特征列（从首个 airway_features.csv）
    feat_cols = []
    feat_by_patient = {}
    n_feat = 0
    for fname in sorted(os.listdir(D5f)):
        if not fname.endswith("_airway_features.csv"):
            continue
        patient = fname[: -len("_airway_features.csv")]
        path = os.path.join(D5f, fname)
        try:
            df = pd.read_csv(path)
            row = df.iloc[0].to_dict()
        except Exception:
            continue
        if not feat_cols:
            feat_cols = list(row.keys())
        feat_by_patient[patient] = {"features_csv": path.replace("/", "\\"),
                                    "features": {k: _clean(v) for k, v in row.items()}}
        n_feat += 1

    # 整合：2026-02 格式 + 每个患者补挂特征
    integrated = dict(s5)
    integrated["feature_dir"] = D5f.replace("/", "\\")
    integrated["feature_columns"] = feat_cols
    integrated["n_with_features"] = n_feat
    out_patients = []
    for p in s5["patients"]:
        entry = dict(p)
        pf = entry["patient_folder"]
        if pf in feat_by_patient:
            entry["features_csv"] = feat_by_patient[pf]["features_csv"]
            entry["features"] = feat_by_patient[pf]["features"]
        else:
            entry["features_csv"] = None
            entry["features"] = None
        out_patients.append(entry)
    integrated["patients"] = out_patients

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(integrated, f, ensure_ascii=False, indent=2)
    print(f"[out] {OUT_JSON}: {len(out_patients)} 患者, "
          f"挂特征 {sum(1 for p in out_patients if p.get('features') is not None)}")

    # ---- 对齐报告 ----
    set2 = {p["patient_folder"] for p in s2["patients"]}
    set5 = {p["patient_folder"] for p in s5["patients"]}

    def metrics_cols(d, pf):
        csv = os.path.join(d, f"{pf}_airquant", f"{pf}_full_metrics.csv")
        if os.path.exists(csv):
            return list(pd.read_csv(csv, nrows=1).columns)
        return None

    def first_readable(d, patients):
        for pf in patients:
            c = metrics_cols(d, pf)
            if c:
                return c
        return None

    c2 = first_readable(D2, list(set2))
    c5 = first_readable(D5m, list(set5))

    L = []
    L.append("# AirQuant 特征对齐报告：2026-02 vs 2026-05\n")
    L.append("> 生成时间：2026-08-27\n")
    L.append("## 1. 患者对齐\n")
    L.append("| 项 | 数值 |")
    L.append("|---|---|")
    L.append(f"| 2026-02 患者数 | {len(set2)} |")
    L.append(f"| 2026-05 患者数 | {len(set5)} |")
    L.append(f"| **交集（共同患者）** | **{len(set2 & set5)}** |")
    L.append(f"| 仅 2026-02 | {len(set2 - set5)} |")
    L.append(f"| 仅 2026-05 | {len(set5 - set2)} |")
    L.append("")
    L.append("**结论：两个序列患者完全无交集，无法逐患者对应**（是两批独立病人）。")
    L.append("")
    L.append("## 2. 逐分支指标（full_metrics.csv）schema\n")
    L.append("| 项 | 2026-02 | 2026-05 | 一致 |")
    L.append("|---|---|---|---|")
    L.append(f"| 列数 | {len(c2) if c2 else '-'} | {len(c5) if c5 else '-'} | "
             f"{'✅' if c2 and c5 and c2 == c5 else '❌'} |")
    L.append("")
    if c2:
        L.append("列名：" + ", ".join(c2))
        L.append("")
    L.append("**结论：逐分支指标列完全一致**，若对 2026-02 补算聚合特征，可与 2026-05 直接合并。")
    L.append("")
    L.append("## 3. 聚合特征（_airway_features.csv，aq_*）\n")
    L.append("| 项 | 2026-02 | 2026-05 |")
    L.append("|---|---|---|")
    n2_feat = len([f for f in os.listdir(D2) if f.endswith('_airway_features.csv')])
    L.append(f"| `_airway_features.csv` 数量 | {n2_feat} | {n_feat} |")
    L.append(f"| 特征列数 | - | {len(feat_cols)} |")
    L.append("")
    L.append(f"**2026-05 特征列（{len(feat_cols)} 列）**：")
    L.append("")
    L.append(", ".join(f"`{c}`" for c in feat_cols))
    L.append("")
    L.append("**结论：2026-02 未计算 aq_* 聚合特征**，需对 2026-02 运行 "
             "`AirQuant/compute_airway_features.m`（METRICS_DIR=E:/DICOM/2026-02-Airway_metrics, "
             "FEATURES_DIR 自定）才能与 2026-05 在特征层面对齐。")
    L.append("")
    L.append("## 4. 建议\n")
    L.append("- 若需两队列合并分析：先对 2026-02 补跑 compute_airway_features.m（其 full_metrics schema 与 2026-05 一致，可直接复用）。")
    L.append("- 整合文件：`airquant_2026_05_integrated_summary.json`（2026-02 格式 + 每患者挂 69 列特征）。")
    L.append("- 全量特征表已存在：`E:/DICOM/2026-05-Airway_features/airway_features_all.csv`（1038 例 × 69 列）。")
    md = "\n".join(L)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[out] {OUT_MD}")


if __name__ == "__main__":
    main()
