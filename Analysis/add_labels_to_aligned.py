#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法: python add_labels_to_aligned.py   # 给对齐表补标签列 + 列临床影像特征
add_labels_to_aligned.py
=========================
1) 给 2026-02 / 2026-05 的 *_aligned01.csv 补上分类标签列(ICD/AECOPD/COPD_BCOS)
   输出 *_labeled.csv
2) 列出"临床影像特征"(LAA/肺气肿/气道Pi10/WA/TD等)在三队列表中的位置
"""
import os
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOG = open(r"E:\DICOM\reports\add_labels.log", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()

CLIN_KW = ("LAA950", "Perc15", "Pi10", "WA_pct", "TD_", "wall_", "blur_",
           "Vessel_", "Lobe_", "Lung_", "Din_", "Dout_", "tortuosity",
           "n_branches", "pruning", "max_generation")


def main():
    log("===== 补标签 + 临床影像特征清单 =====")
    # ---- 2) 临床影像特征 ----
    h1 = pd.read_csv(r"E:\DICOM\2026-01-seg\2026-01-integrated_radiomics_aq.csv", nrows=0).columns
    clin = [c for c in h1 if any(k in c for k in CLIN_KW)]
    log(f"2026-01 标准中的临床影像特征共 {len(clin)} 列:")
    laa = [c for c in clin if "LAA" in c]
    perc = [c for c in clin if "Perc15" in c]
    airway = [c for c in clin if c.startswith(("Pi10", "WA_", "TD_", "wall_", "blur_", "Din_", "Dout_"))]
    vessel = [c for c in clin if c.startswith("Vessel_")]
    lobe = [c for c in clin if c.startswith("Lobe_")]
    lung = [c for c in clin if c.startswith("Lung_")]
    for name, cols in [("LAA(肺气肿%<-950HU)", laa), ("Perc15(肺气肿HU分位)", perc),
                       ("气道(Pi10/WA/TD/wall/blur/Din/Dout)", airway),
                       ("血管Vessel", vessel), ("肺叶Lobe", lobe), ("肺Lung", lung)]:
        log(f"\n[{name}] {len(cols)} 列:")
        for c in cols:
            log(f"    {c}")

    # ---- 1) 补标签 ----
    for tag in ["02", "05"]:
        src = {"02": r"E:\DICOM\2026-02-seg\2026-02-integrated_radiomics_aq_aligned01.csv",
               "05": r"E:\DICOM\2026-05-seg\2026-05-integrated_radiomics_aq_aligned01.csv"}[tag]
        lab = {"02": r"E:\DICOM\2026-02-seg\labels_ae_bcos_2026_02.csv",
               "05": r"E:\DICOM\2026-05-seg\labels_ae_bcos_2026_05.csv"}[tag]
        df = pd.read_csv(src)
        lb = pd.read_csv(lab)[["Patient_ID", "ICD", "AECOPD", "COPD_BCOS"]]
        m = df.merge(lb, on="Patient_ID", how="left")
        out = src.replace("_aligned01.csv", "_aligned01_labeled.csv")
        m.to_csv(out, index=False, encoding="utf-8-sig")
        log(f"\n== 2026-{tag} == 补标签后 {m.shape[0]} 行 x {m.shape[1]} 列 -> {out}")
        log(f"   AECOPD: 1={int((m['AECOPD']==1).sum())} 0={int((m['AECOPD']==0).sum())} 缺失={int(m['AECOPD'].isna().sum())}")
        log(f"   COPD_BCOS: 1={int((m['COPD_BCOS']==1).sum())} 0={int((m['COPD_BCOS']==0).sum())}")
    LOG.close()


if __name__ == "__main__":
    main()
