#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_2026_05_vessel_features.py
===================================
给 <nifti_dir> 下所有患者补充肺血管高级特征
（分形维度 / BV5-BV10 / 中心线图论），输出全部患者的 Vessel_* 特征。

- 患者来源: 扫描 nifti 目录所有文件夹（1106 例）
- 掩膜来源: <seg_dir>/<患者>_masks/lung_vessels.nii.gz
           （无掩膜的患者无法计算 -> NaN）
- 多进程并行 + 增量落盘缓存（每 25 例保存一次，重启不丢进度）
- 输出:
    vessel_feats_2026_05_all.csv      -> 全部 nifti 患者的 Vessel_* 特征
    radiomics_2026_05_features_vessel.csv -> 与 radiomics CSV 合并（重叠患者）
"""
import os
import sys
import json
import time
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import SimpleITK as sitk

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_patient_radiomics_fast import vessel_advanced_features

NIFTI = "nifti_input"
SEG = "seg_results"
RAD_CSV = os.path.join(SEG, "radiomics_2026_05_features.csv")
OUT_ALL = os.path.join(SEG, "vessel_feats_2026_05_all.csv")
OUT_RAD = os.path.join(SEG, "radiomics_2026_05_features_vessel.csv")
CACHE = os.path.join(SEG, "vessel_feats_2026_05.json")
WORKERS = 6
SAVE_EVERY = 25
VESSEL_COLS = [
    "Vessel_Fractal_Dim", "Vessel_BV5_pct", "Vessel_BV10_pct",
    "Vessel_Skeleton_Voxels", "Vessel_Skeleton_Length_mm",
    "Vessel_Branch_Count", "Vessel_Junction_Count", "Vessel_Endpoint_Count",
    "Vessel_Branching_Density_per_mm", "Vessel_Tortuosity_Mean",
    "Vessel_Tortuosity_Max",
]


def _clean(feats):
    out = {}
    for k, v in feats.items():
        if isinstance(v, (np.floating, float)) and np.isnan(v):
            out[k] = None
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _compute_one(patient):
    mask_path = os.path.join(SEG, f"{patient}_masks", "lung_vessels.nii.gz")
    if not os.path.exists(mask_path):
        return patient, {"_error": "no_mask"}
    try:
        img = sitk.ReadImage(mask_path)
        spacing = img.GetSpacing()
        arr = sitk.GetArrayFromImage(img) > 0
        return patient, _clean(vessel_advanced_features(arr, spacing))
    except Exception as e:
        return patient, {"_error": str(e)}


def _save_cache(cache):
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def main(patients_filter=None):
    patients = sorted(p for p in os.listdir(NIFTI)
                      if os.path.isdir(os.path.join(NIFTI, p)))
    if patients_filter:
        wanted = set(patients_filter)
        patients = [p for p in patients if p in wanted]
        print(f"🎯 指定患者过滤: {len(patients)} 个")
    print(f"nifti 患者总数: {len(patients)}")

    cache = {}
    if os.path.exists(CACHE):
        try:
            with open(CACHE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            # 断点续跑：no_mask 视为终态不重试；其余失败项丢弃以便重试
            cache = {k: v for k, v in cache.items()
                     if not ("_error" in v and v["_error"] != "no_mask")}
            print(f"缓存 {len(cache)} 个有效患者（断点续跑，非 no_mask 失败项将重试）")
        except Exception as e:
            print(f"[warn] 缓存读取失败: {e}")

    todo = [p for p in patients if p not in cache]
    print(f"待计算 {len(todo)} / {len(patients)}")
    if todo:
        t0 = time.time()
        done = 0
        with Pool(processes=WORKERS) as pool:
            for patient, feats in pool.imap_unordered(_compute_one, todo, chunksize=1):
                cache[patient] = feats
                done += 1
                if done % SAVE_EVERY == 0:
                    _save_cache(cache)
                    print(f"  [{done}/{len(todo)}] 已存缓存, 耗时 {time.time()-t0:.0f}s",
                          flush=True)
        _save_cache(cache)
        print(f"计算完成 {len(todo)} 个，总耗时 {time.time()-t0:.1f}s")
        n_ok = sum("_error" not in f for f in cache.values())
        print(f"有掩膜计算成功: {n_ok} | 无掩膜/失败: {len(cache)-n_ok}")

    # 输出 1：全部 nifti 患者的 Vessel_* 特征
    rows = []
    for p in patients:
        f = cache.get(p, {})
        ok = "_error" not in f
        rows.append({"Patient_ID": p, **{c: (f.get(c) if ok else None) for c in VESSEL_COLS}})
    vdf = pd.DataFrame(rows)
    vdf.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")
    print(f"[out] {OUT_ALL}: {len(vdf)} 行, Vessel_Fractal_Dim 非空 "
          f"{int(vdf['Vessel_Fractal_Dim'].notna().sum())}")

    # 输出 2：与 radiomics CSV 合并（重叠患者）
    if os.path.exists(RAD_CSV):
        rad = pd.read_csv(RAD_CSV)
        idcol = "Patient_ID" if "Patient_ID" in rad.columns else rad.columns[0]
        rad[idcol] = rad[idcol].astype(str)
        out = rad.merge(vdf, on="Patient_ID", how="left")
        for c in VESSEL_COLS:
            if c not in out.columns:
                out[c] = np.nan
        out.to_csv(OUT_RAD, index=False, encoding="utf-8-sig")
        print(f"[out] {OUT_RAD}: {len(out)} 行 x {len(out.columns)} 列")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量肺血管高级特征 (Vessel_*)")
    parser.add_argument("--patients", default=None,
                        help="只处理指定患者（逗号分隔），默认全部")
    args = parser.parse_args()
    wanted = args.patients.split(",") if args.patients else None
    main(wanted)
