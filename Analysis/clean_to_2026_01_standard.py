#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法: python clean_to_2026_01_standard.py   # 02/05 规约到 2026-01 胸部标准特征
clean_to_2026_01_standard.py
============================
以 2026-01 的特征列为标准（胸部CT分割覆盖的特征集，剔除肾上腺/肾/椎体/肋骨/髋等
非胸部扩展器官），规约 2026-02 和 2026-05：
  - 只保留 2026-01 标准中存在的特征列 + ID/标签列
  - 分块读写，输出 *_aligned01.csv
输出: E:\DICOM\2026-02-seg\2026-02-integrated_radiomics_aq_aligned01.csv
      E:\DICOM\2026-05-seg\2026-05-integrated_radiomics_aq_aligned01.csv
"""
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOG = open(r"E:\DICOM\reports\clean_aligned01.log", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()

META = {"Patient_ID", "PatientID", "PatientID_raw", "Patient_ID_long", "CT_Series",
        "patient_id", "ICD", "main_diagnosis", "AECOPD", "COPD_BCOS", "患者id"}


def main():
    log("===== 按 2026-01 标准规约 2026-02 / 2026-05 =====")
    # 1) 2026-01 标准特征列
    d1 = pd.read_csv(r"E:\DICOM\2026-01-seg\2026-01-integrated_radiomics_aq.csv", nrows=0)
    std_cols = [c for c in d1.columns if c not in META]
    log(f"2026-01 标准特征列: {len(std_cols)}")

    for tag, src, idcols in [
        ("02", r"E:\DICOM\2026-02-seg\2026-02-integrated_radiomics_aq.csv",
         ["Patient_ID", "PatientID", "CT_Series"]),
        ("05", r"E:\DICOM\2026-05-seg\2026-05-integrated_radiomics_aq.csv",
         ["Patient_ID", "PatientID", "CT_Series"])]:
        hdr = list(pd.read_csv(src, nrows=0).columns)
        keep = idcols + [c for c in hdr if c in std_cols and c not in idcols]
        dropped = [c for c in hdr if c not in keep]
        # 按块统计丢弃
        from collections import Counter
        blk = Counter()
        for c in dropped:
            if "::" in c:
                blk[c.split("::")[0]] += 1
            else:
                blk[c] += 1
        top = blk.most_common(15)
        log(f"\n== 2026-{tag} == 原特征 {len(hdr)-len(idcols)} -> 保留 {len(keep)-len(idcols)} "
            f"(丢弃 {len(dropped)})")
        log("  丢弃块 Top: " + "; ".join(f"{k}({v})" for k, v in top))
        # 确认丢弃的都是非胸部
        chest_kw = ("lung", "heart", "aorta", "pulmonary", "trachea", "airway",
                    "bronchus", "diaphragm", "mediastinum", "thymus", "esophagus",
                    "aortic", "coronary", "rib_", "spine", "vertebra", "sternum",
                    "clavicle", "scapula", "humerus", "lobe", "vessel")
        non_chest_drop = [c for c in dropped if not any(k in c for k in chest_kw)]
        log(f"  丢弃中明显非胸部(无 lung/heart/aorta 等关键词)约: {len(non_chest_drop)}/{len(dropped)}")
        for c in dropped[:6]:
            log(f"    例: {c[:70]}")

        # 分块写
        out = src.replace(".csv", "_aligned01.csv")
        first = True
        tot = 0
        for ch in pd.read_csv(src, usecols=keep, chunksize=250):
            ch.to_csv(out, mode="w" if first else "a", header=first, index=False,
                      encoding="utf-8-sig")
            first = False
            tot += len(ch)
        log(f"  输出 -> {out} ({tot} 行 x {len(keep)} 列)")

    LOG.close()


if __name__ == "__main__":
    main()
