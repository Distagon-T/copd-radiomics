#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法: python build_bcos_labels_consistent.py   # 生成一致 ICD 标签(05/02)
build_bcos_labels_consistent.py
================================
用同一套 ICD 规则，给 2026-05 / 2026-02 两队列整理出可对齐的分类标签：
  AECOPD    (COPD 急性加重): 主要诊断-ICD码 以 J44.1*(急性加重) 或 J44.0*(急性下呼吸道感染) 开头 -> 1
                              J44.9*(未特指)/J44.8*(其他) -> 0；J47*(支扩) 排除(置 NaN)
  COPD_BCOS (COPD 合并支扩): 医生标注 COPD合并支扩 == 1 -> 1，否则 0
输出:
  E:\DICOM\2026-05-seg\labels_ae_bcos_2026_05.csv
  E:\DICOM\2026-02-seg\labels_ae_bcos_2026_02.csv
并打印两队列对齐后的分布对比。
"""
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def norm(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip().str.lstrip("0")


def ae_label(icd):
    s = icd.astype(str).str.strip()
    out = pd.Series(np.nan, index=icd.index)
    out[(s.str.startswith("J44.1")) | (s.str.startswith("J44.0"))] = 1
    out[(s.str.startswith("J44.9")) | (s.str.startswith("J44.8"))] = 0
    return out.astype("Int64")


def main():
    # ---------- 2026-05 ----------
    ov5 = pd.read_excel(r"E:\DICOM\2026-05\2026-5-9-overlap.xlsx")
    ov5["ICD"] = ov5["主要诊断-ICD码"].astype(str).str.strip()
    ov5["AECOPD"] = ae_label(ov5["ICD"])
    ov5["COPD_BCOS"] = (ov5["COPD合并支扩"] == 1).astype(int)
    # 患者id -> Patient_ID(long) 映射（经 radiomics_2026_05_features.csv）
    rad5 = pd.read_csv(r"E:\DICOM\2026-05-seg\radiomics_2026_05_features.csv",
                       usecols=["Patient_ID", "PatientID"])
    rad5["n"] = norm(rad5["PatientID"])
    pid2long = rad5.drop_duplicates("n").set_index("n")["Patient_ID"].to_dict()
    ov5["n"] = norm(ov5["患者id"])
    ov5["Patient_ID"] = ov5["n"].map(pid2long)
    out5 = ov5[["患者id", "Patient_ID", "ICD", "主要诊断", "AECOPD", "COPD_BCOS"]].copy()
    out5 = out5.rename(columns={"患者id": "patient_id", "主要诊断": "main_diagnosis"})
    out5.to_csv(r"E:\DICOM\2026-05-seg\labels_ae_bcos_2026_05.csv",
                index=False, encoding="utf-8-sig")

    # ---------- 2026-02 ----------
    x2 = pd.read_excel(r"E:\DICOM\2026-02-seg\2026-2提取.xlsx")
    x2["ICD"] = x2["主要诊断-ICD码"].astype(str).str.strip()
    x2["AECOPD"] = ae_label(x2["ICD"])
    x2["COPD_BCOS"] = (x2["COPD合并支扩"] == 1).astype(int)
    csv2 = pd.read_csv(r"E:\DICOM\2026-02-seg\2026-02-integrated_radiomics_aq.csv",
                       usecols=["Patient_ID", "PatientID"])
    csv2["n"] = norm(csv2["PatientID"])
    pid2long2 = csv2.drop_duplicates("n").set_index("n")["Patient_ID"].to_dict()
    x2["n"] = norm(x2["患者id"])
    x2["Patient_ID"] = x2["n"].map(pid2long2)
    out2 = x2[["患者id", "Patient_ID", "ICD", "AECOPD", "COPD_BCOS"]].copy()
    out2 = out2.rename(columns={"患者id": "patient_id"})
    out2.to_csv(r"E:\DICOM\2026-02-seg\labels_ae_bcos_2026_02.csv",
                index=False, encoding="utf-8-sig")

    # ---------- 对比 ----------
    def summ(df, tag):
        j44 = df[df["ICD"].str.startswith("J44")]
        j47 = df[df["ICD"].str.startswith("J47")]
        ae = j44["AECOPD"]
        print(f"== 2026-{tag} ==  总 {len(df)} 行")
        print(f"  J44(COPD)={len(j44)}  J47(支扩)={len(j47)}  其他={len(df)-len(j44)-len(j47)}")
        print(f"  AECOPD: 1={int((ae==1).sum())} 0={int((ae==0).sum())} "
              f"(阳性率 {int((ae==1).sum())/len(j44):.1%})")
        print(f"  COPD_BCOS(J44内): 1={int((j44['COPD_BCOS']==1).sum())} "
              f"0={int((j44['COPD_BCOS']==0).sum())}")
        print(f"  Patient_ID 可映射特征表: {int(df['Patient_ID'].notna().sum())}/{len(df)}")

    print("=" * 60)
    summ(out5, "05")
    print()
    summ(out2, "02")
    print("=" * 60)
    print("标签文件已输出: labels_ae_bcos_2026_05.csv / labels_ae_bcos_2026_02.csv")


if __name__ == "__main__":
    main()
