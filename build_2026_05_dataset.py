#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_2026_05_dataset.py
========================
2026-05 队列：读取 E:/DICOM/2026-05-seg/ 下每个患者 radiomics json，
用 E:/DICOM/2026-05/info-2026-05.csv 匹配临床。
Label = 主要诊断（第二列）含 "急性加重" → 1，否则 0。

输出:
  E:/DICOM/2026-05-seg/radiomics_2026_05_features.csv  (纯特征表，供 LR)
  E:/DICOM/2026-05-seg/labels_2026_05.csv              (patient_id + label)
  E:/DICOM/2026-05-seg/label_diagnostics_2026_05.txt
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEG = r"E:\DICOM\2026-05-seg"
INFO = r"E:\DICOM\2026-05\info-2026-05.csv"
FEAT_OUT = os.path.join(SEG, "radiomics_2026_05_features.csv")
LAB_OUT = os.path.join(SEG, "labels_2026_05.csv")
REP = os.path.join(SEG, "label_diagnostics_2026_05.txt")

# 复用 ID 归一化
sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_clinical_radiomics import normalize_id


def load_radiomics():
    rows = []
    files = sorted(glob.glob(os.path.join(SEG, "*_radiomics.json")))
    n_pid = 0
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                r = json.load(fh)
        except Exception:
            continue
        if not isinstance(r, dict) or "Patient_ID" not in r:
            continue
        if r.get("PatientID"):
            n_pid += 1
        rows.append(r)
    df = pd.DataFrame(rows)
    print(f"radiomics json: {len(files)} 个, 有效 {len(df)}, 含 PatientID {n_pid}")
    return df


def main():
    radi = load_radiomics()
    radi["PatientID"] = normalize_id(radi["PatientID"])

    # ---- 读临床（info-2026-05.csv, UTF-8）----
    clin = pd.read_csv(INFO, encoding="utf-8")
    print(f"info-2026-05 读取: {len(clin)} 行, 列: {list(clin.columns)}")
    clin_id = "患者id"
    if clin_id not in clin.columns:
        sys.exit("info-2026-05 缺 患者id 列")
    clin[clin_id] = normalize_id(clin[clin_id])

    # ---- 匹配 ----
    m = radi.merge(clin.rename(columns={clin_id: "PatientID"}),
                   on="PatientID", how="inner")
    print(f"匹配到临床的患者: {len(m)} / {len(radi)}")
    if len(m) == 0:
        sys.exit("无匹配，停止")
    m = m.drop_duplicates(subset=["PatientID"], keep="first")
    print(f"去重后: {len(m)}")

    # ---- 生成 label：主要诊断含"急性加重" ----
    diag_col = "主要诊断"
    rows = []
    for _, r in m.iterrows():
        txt = str(r.get(diag_col, ""))
        label = 1 if "急性加重" in txt else 0
        rows.append({
            "PatientID": r["PatientID"],
            "Patient_ID": r["Patient_ID"],
            "cvd_exacerbation_label": label,
            "main_diagnosis": txt,
        })
    lab = pd.DataFrame(rows)

    # ---- 保存：特征表（纯 radiomics，无临床列）----
    id_cols = ["Patient_ID", "PatientID", "CT_Series"]
    feat_cols = [c for c in radi.columns if c not in id_cols]
    feats = m[id_cols + feat_cols].copy()
    feats.to_csv(FEAT_OUT, index=False, encoding="utf-8-sig")
    # 保存 label 表
    lab_out = lab.rename(columns={"PatientID": "patient_id"})
    lab_out.to_csv(LAB_OUT, index=False, encoding="utf-8-sig")

    # ---- 报告 ----
    L = []
    L.append(f"=== 2026-05 队列 急性加重 Label 诊断报告 ===")
    L.append(f"总患者数: {len(m)}")
    L.append(f"阳性(急性加重): {int(lab['cvd_exacerbation_label'].sum())}")
    L.append(f"阴性: {int((lab['cvd_exacerbation_label']==0).sum())}")
    L.append("")
    L.append("--- 阳性主要诊断样例 ---")
    for _, r in lab[lab["cvd_exacerbation_label"] == 1].head(10).iterrows():
        L.append(f"  {r['PatientID']} | {str(r['main_diagnosis'])[:50]}")
    txt = "\n".join(L)
    with open(REP, "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    print(f"\n特征表: {FEAT_OUT}")
    print(f"Label表: {LAB_OUT}")


if __name__ == "__main__":
    main()
