#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_airquant.py
=====================
把 AirQuant 的 branch 级指标聚合成 patient 级特征。

输入:
  E:\DICOM\2026-04-Airway_metrics_tmp\<patient>_airquant\<patient>_full_metrics.csv
  E:\DICOM\2026-04-Airway_metrics_tmp\<patient>_airquant\<patient>_airquant_info.json

输出:
  <out_dir>/airquant_patient_aggregated.csv  (key = patient folder name)

聚合方式:
  - 分支数、measured 数、骨架体素数、Pi10
  - 全体分支的均值/标准差: WA_pct, Wall_Thickness_mm, 内外径, Lumen/Wall 面积, 迂曲度
  - 按 generation 分层均值: Gen<=2 / Gen3 / Gen4 / Gen>=5
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

AQ_DIR = r"E:\DICOM\2026-04-Airway_metrics_tmp"

NUM_COLS = [
    "stats_arclength", "stats_change_deg", "stats_euclength",
    "stats_tortuosity", "LumenArea_mm2", "WallArea_mm2", "WA_pct",
    "Inner_Diameter_mm", "Outer_Diameter_mm", "Wall_Thickness_mm",
    "Pi_Perimeter_mm", "Sqrt_WallArea",
]
GEN_BINS = [(0, 2, "GenLe2"), (3, 3, "Gen3"), (4, 4, "Gen4"), (5, 99, "GenGe5")]
LAYER_COLS = ["WA_pct", "Wall_Thickness_mm", "Inner_Diameter_mm", "stats_tortuosity"]


def match_patient_id(name, id_set):
    """文件夹名 -> radiomics Patient_ID 对齐。
    处理序列后缀差异: 'XXX.1' -> 'XXX'"""
    if name in id_set:
        return name
    # 去掉结尾 '.数字' 后缀再试
    m = re.sub(r"\.\d+$", "", name)
    if m in id_set:
        return m
    # 前缀匹配
    for pid in id_set:
        if name.startswith(pid):
            return pid
    return name


def aggregate_one(patient_dir, id_set=None):
    """聚合单个患者的 branch 指标 -> dict"""
    name = os.path.basename(patient_dir)[:-len("_airquant")]
    key = match_patient_id(name, id_set) if id_set else name
    csv = os.path.join(patient_dir, f"{name}_full_metrics.csv")
    if not os.path.exists(csv):
        # 兜底：文件夹内任意 full_metrics.csv
        cand = glob.glob(os.path.join(patient_dir, "*_full_metrics.csv"))
        if cand:
            csv = cand[0]
        else:
            return None
    df = pd.read_csv(csv)
    d = {"patient": key, "aq_n_branches": len(df)}
    if len(df) == 0:
        return None

    # 全体分支均值 / 标准差
    for c in NUM_COLS:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            d[f"aq_{c}_mean"] = v.mean()
            d[f"aq_{c}_std"] = v.std()

    # 按 generation 分层
    gen = pd.to_numeric(df["generation"], errors="coerce")
    for lo, hi, tag in GEN_BINS:
        m = df[(gen >= lo) & (gen <= hi)]
        d[f"aq_{tag}_count"] = len(m)
        for c in LAYER_COLS:
            if c in df.columns:
                v = pd.to_numeric(m[c], errors="coerce")
                d[f"aq_{tag}_{c}_mean"] = v.mean() if len(m) else np.nan

    # info json: Pi10 等
    info = os.path.join(patient_dir, f"{name}_airquant_info.json")
    if not os.path.exists(info):
        cand = glob.glob(os.path.join(patient_dir, "*_airquant_info.json"))
        if cand:
            info = cand[0]
    if os.path.exists(info):
        try:
            j = json.load(open(info, encoding="utf-8"))
            d["aq_Pi10"] = j.get("Pi10")
            d["aq_num_measured"] = j.get("num_measured")
            d["aq_skeleton_voxels"] = j.get("skeleton_voxels")
        except Exception:
            pass
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aq-dir", default=AQ_DIR)
    ap.add_argument("--id-csv", default=r"E:\DICOM\2026-04-seg-part1\radiomics_all_patients.csv",
                    help="radiomics CSV（提供 Patient_ID 集合用于名称对齐，可省）")
    ap.add_argument("--out", default=r"E:\DICOM\2026-04-seg-part1\airquant_patient_aggregated.csv")
    args = ap.parse_args()

    id_set = None
    if args.id_csv and os.path.exists(args.id_csv):
        radi = pd.read_csv(args.id_csv)
        id_set = set(radi["Patient_ID"].astype(str))
        print(f"使用 radiomics Patient_ID 集合: {len(id_set)}")

    rows = []
    failed = []
    for d in sorted(glob.glob(os.path.join(args.aq_dir, "*_airquant"))):
        r = aggregate_one(d, id_set)
        if r:
            rows.append(r)
        else:
            failed.append(os.path.basename(d))
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"成功聚合: {len(out)} 患者, {len(out.columns)} 特征列")
    print(f"失败: {len(failed)}")
    if failed:
        print(failed[:5])
    print(f"Pi10 缺失: {out['aq_Pi10'].isna().sum() if 'aq_Pi10' in out.columns else 'N/A'}")
    print(f"输出: {args.out}")


if __name__ == "__main__":
    main()
