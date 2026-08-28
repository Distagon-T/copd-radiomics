#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法: python integrate_2026_01.py   # 整合 2026-01 特征大表(radiomics+airway)
integrate_2026_01.py
=====================
整合 2026-01 特征为一张大表（与 2026-02/05 同逻辑）：
  - radiomics : E:\DICOM\2026-01-seg\<PatientFolder>_radiomics.json  (100 个, 2205 特征)
  - airway(aq): E:\DICOM\2026-01-Airway_features\<PatientFolder>_airway_features.csv (87 个)
  - PatientID : 数字（来自 2026-01-nifti 的 dicom_info.json）
  - labels    : E:\DICOM\2026-01\2026-1标注信息.xlsx -> AECOPD(J44.1|J44.0) / COPD_BCOS(慢阻肺合并支扩)
输出:
  2026-01-integrated_radiomics_aq.csv  (长串 Patient_ID 版)
  2026-01-integrated_radiomics_aq_numid.csv (数字 PatientID 版)
  01_pid_map.csv
"""
import os
import sys
import glob
import json

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOG = open(r"E:\DICOM\2026-01-seg\integrate_01.log", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()


def norm(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip().str.lstrip("0")


def build_pid_map():
    m = {}
    for jf in glob.glob(r"E:\DICOM\2026-01-nifti\*\*_dicom_info.json"):
        folder = os.path.basename(os.path.dirname(jf))
        try:
            with open(jf, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        pid = None
        for s in (d.get("Series") or []):
            if s and s.get("Patient") and s["Patient"].get("PatientID"):
                pid = str(s["Patient"]["PatientID"]).strip()
                break
        if pid:
            m[folder] = pid
    return m


def main():
    log("===== 整合 2026-01 特征 =====")
    # 1) radiomics
    rads = []
    for jf in sorted(glob.glob(r"E:\DICOM\2026-01-seg\*_radiomics.json")):
        with open(jf, encoding="utf-8") as f:
            d = json.load(f)
        rads.append(d)
    df_rad = pd.DataFrame(rads)
    log(f"radiomics: {len(df_rad)} 行 x {len(df_rad.columns)} 列")

    # 2) airway (跳过 airway_features_all.csv 汇总表, 只读逐患者文件)
    aq_list = []
    for cf in sorted(glob.glob(r"E:\DICOM\2026-01-Airway_features\*.csv")):
        if os.path.basename(cf).lower().endswith("all.csv"):
            continue
        s = pd.read_csv(cf).rename(columns={"patient_folder": "Patient_ID"})
        aq_list.append(s)
    df_aq = pd.concat(aq_list, ignore_index=True)
    df_aq = df_aq.drop_duplicates(subset=["Patient_ID"], keep="first")
    log(f"airway: {len(df_aq)} 行 x {len(df_aq.columns)} 列")

    # 3) merge
    m = df_rad.merge(df_aq, on="Patient_ID", how="left")
    log(f"merge: {len(m)} 行 x {len(m.columns)} 列")

    # 4) numeric PatientID
    pm = build_pid_map()
    m["_raw"] = m["Patient_ID"].map(pm)
    m["PatientID"] = pd.to_numeric(m["_raw"], errors="coerce").astype("Int64")
    m["PatientID_raw"] = m["_raw"].fillna("")
    log(f"数字 PatientID 映射: {int(m['PatientID'].notna().sum())}/{len(m)}")

    # 5) labels
    lab = pd.read_excel(r"E:\DICOM\2026-01\2026-1标注信息.xlsx")
    lab["_nid"] = norm(lab["患者id"])
    icd = lab["主要诊断-ICD码"].astype(str).str.strip()
    lab["AECOPD"] = pd.Series(np.nan, index=lab.index)
    lab.loc[icd.str.startswith(("J44.1", "J44.0")), "AECOPD"] = 1
    lab.loc[icd.str.startswith(("J44.9", "J44.8")), "AECOPD"] = 0
    lab["AECOPD"] = lab["AECOPD"].astype("Int64")
    bcol = [c for c in lab.columns if "支扩" in c][0]
    lab["COPD_BCOS"] = (lab[bcol] == 1).astype(int)
    m["_nid"] = norm(m["PatientID"])
    m = m.merge(lab[["_nid", "ICD" if "ICD" in lab.columns else "主要诊断-ICD码", "AECOPD", "COPD_BCOS"]]
                .rename(columns={"主要诊断-ICD码": "ICD"}), on="_nid", how="left")
    log(f"join 标签后: {len(m)} 行; AECOPD 1={int((m['AECOPD']==1).sum())} 0={int((m['AECOPD']==0).sum())}; "
        f"COPD_BCOS 1={int((m['COPD_BCOS']==1).sum())}")
    m = m.drop(columns=["_nid", "_raw"])

    # 6) 输出
    out_cols = ["Patient_ID", "PatientID", "PatientID_raw", "ICD", "AECOPD", "COPD_BCOS"] + \
               [c for c in m.columns if c not in ("Patient_ID", "PatientID", "PatientID_raw", "ICD", "AECOPD", "COPD_BCOS")]
    m = m[out_cols]
    m.to_csv(r"E:\DICOM\2026-01-seg\2026-01-integrated_radiomics_aq.csv",
             index=False, encoding="utf-8-sig")
    log(f"输出 -> E:\\DICOM\\2026-01-seg\\2026-01-integrated_radiomics_aq.csv ({len(m.columns)} 列)")

    # numid 版 + pid map
    m2 = m.copy()
    if "Patient_ID_long" not in m2.columns:
        m2 = m2.rename(columns={"Patient_ID": "Patient_ID_long"})
    m2.to_csv(r"E:\DICOM\2026-01-seg\2026-01-integrated_radiomics_aq_numid.csv",
              index=False, encoding="utf-8-sig")
    pm_df = pd.DataFrame({"Patient_ID_long": m2["Patient_ID_long"],
                          "PatientID": m2["PatientID"].astype(str),
                          "PatientID_raw": m2["PatientID_raw"]}).drop_duplicates()
    pm_df.to_csv(r"E:\DICOM\2026-01-seg\01_pid_map.csv", index=False, encoding="utf-8-sig")
    log("numid 版 + 01_pid_map.csv 已写")
    LOG.close()


if __name__ == "__main__":
    main()
