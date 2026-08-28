#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法: python convert_pid_to_numeric_v2.py <02|05>   # PatientID 转数字
convert_pid_to_numeric_v2.py  <02|05>   (分块版，避免宽表整写被静默 kill)
======================================================================
在 integrated CSV 原表基础上：
  - PatientID     -> 整数数字 ID（来自 nifti json 的 Patient.PatientID，去前导零；
                     无 json 时回退原 CSV 数值）
  - PatientID_raw -> json 原始字符串（保留前导零，如 0000846230）
  - Patient_ID    -> 保留原长字符串（目录名）
分块读取/写入，输出 <tag>-integrated_radiomics_aq_numid.csv
另写 <tag>_pid_map.csv 映射表。
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

tag = sys.argv[1] if len(sys.argv) > 1 else "02"
SEG = {"02": r"E:\DICOM\2026-02-seg", "05": r"E:\DICOM\2026-05-seg"}[tag]
NIFTI = {"02": r"E:\DICOM\2026-02-nifti", "05": r"E:\DICOM\2026-05-nifti"}[tag]
CSV = os.path.join(SEG, f"2026-{tag}-integrated_radiomics_aq.csv")
OUT = os.path.join(SEG, f"2026-{tag}-integrated_radiomics_aq_numid.csv")
LOGP = os.path.join(SEG, f"pid_convert_{tag}.log")

LOG = open(LOGP, "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()


def build_map(nifti_dir):
    m = {}
    jsons = sorted(glob.glob(os.path.join(nifti_dir, "*", "*_dicom_info.json")))
    for jf in jsons:
        folder = os.path.basename(os.path.dirname(jf))
        try:
            with open(jf, "r", encoding="utf-8") as f:
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


def to_int_str(x):
    if pd.isna(x):
        return ""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x).strip()


def main():
    log(f"===== 2026-{tag} PatientID -> 数字 (分块) =====")
    log(f"CSV: {CSV}")
    m = build_map(NIFTI)
    log(f"json 映射数: {len(m)}")
    long_col = None
    hdr = pd.read_csv(CSV, nrows=0).columns
    if "Patient_ID" in hdr:
        long_col = "Patient_ID"
    pid_col = "PatientID" if "PatientID" in hdr else long_col
    log(f"长ID列={long_col}, 数字ID列={pid_col}")

    total = miss = json_hit = 0
    first = True
    chunks = pd.read_csv(CSV, chunksize=250)
    for ch in chunks:
        total += len(ch)
        ch["_long"] = ch[long_col].astype(str)
        ch["_raw"] = ch["_long"].map(m)              # NaN 表示无 json
        json_hit += int(ch["_raw"].notna().sum())
        ch["_csv"] = ch[pid_col].map(to_int_str).fillna("") if pid_col in ch else ""
        ch["_pid"] = ch["_raw"].where(ch["_raw"].notna(), ch["_csv"])
        ch["PatientID"] = pd.to_numeric(ch["_pid"], errors="coerce").astype("Int64")
        ch["PatientID_raw"] = ch["_raw"].fillna("")
        miss += int(ch["PatientID"].isna().sum())
        ch = ch.drop(columns=["_long", "_raw", "_csv", "_pid"])
        ch.to_csv(OUT, mode="w" if first else "a", header=first, index=False,
                  encoding="utf-8-sig")
        first = False
        log(f"  已写 {total} 行")
    log(f"完成: 共 {total} 行, json 命中 {json_hit}, 完全缺失(无数字ID) {miss}")
    log(f"输出 -> {OUT}")
    LOG.close()

    # 映射表（小文件单独写）
    d = pd.read_csv(OUT, usecols=["PatientID", "PatientID_raw", long_col])
    d = d.rename(columns={long_col: "Patient_ID_long"})
    d = d.drop_duplicates()
    d.to_csv(os.path.join(SEG, f"{tag}_pid_map.csv"), index=False, encoding="utf-8-sig")
    print(f"映射表 -> {SEG}\\{tag}_pid_map.csv ({len(d)} 行)")


if __name__ == "__main__":
    main()
