#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 11 个 Vessel 高级特征并入 per-patient radiomics JSON（只加缺失键）。
来源：--vessel-csv 指定的 CSV（含 Patient_ID + Vessel_* 列），
或（--recompute-from-mask）从 lung_vessels 掩膜用 vessel_advanced_features 现算。

用法：
  python merge_vessel_into_json.py -s E:/DICOM/2026-05-seg --vessel-csv E:/DICOM/2026-05-seg/vessel_feats_2026_05_all.csv
  python merge_vessel_into_json.py -s E:/DICOM/2026-02-seg --recompute-from-mask
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

VESSEL_COLS = ["Vessel_Fractal_Dim", "Vessel_BV5_pct", "Vessel_BV10_pct",
               "Vessel_Skeleton_Voxels", "Vessel_Skeleton_Length_mm",
               "Vessel_Branch_Count", "Vessel_Junction_Count", "Vessel_Endpoint_Count",
               "Vessel_Branching_Density_per_mm", "Vessel_Tortuosity_Mean",
               "Vessel_Tortuosity_Max"]


def to_jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    return o


def parse_args():
    p = argparse.ArgumentParser(description="把 Vessel 特征并入 radiomics JSON")
    p.add_argument("--seg-dir", "-s", required=True, help="分割结果目录")
    p.add_argument("--vessel-csv", default=None, help="Vessel 特征 CSV（含 Patient_ID + Vessel_* 列）")
    p.add_argument("--recompute-from-mask", action="store_true",
                   help="不从 CSV，改从 lung_vessels 掩膜现算（较慢）")
    p.add_argument("--patients", default=None, help="只处理指定患者（逗号分隔）")
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def compute_from_mask(args):
    """从 lung_vessels 掩膜现算 Vessel 特征（worker）。"""
    patient, seg_dir = args
    json_path = os.path.join(seg_dir, f"{patient}_radiomics.json")
    mask_path = os.path.join(seg_dir, f"{patient}_masks", "lung_vessels.nii.gz")
    if not os.path.exists(mask_path):
        return patient, "no_mask"
    try:
        import SimpleITK as sitk
        import sys as _s
        _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from compute_patient_radiomics_lite import vessel_advanced_features
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        img = sitk.ReadImage(mask_path)
        spacing = img.GetSpacing()
        arr = sitk.GetArrayFromImage(img) > 0
        feats = vessel_advanced_features(arr, spacing)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        added = 0
        for k, v in feats.items():
            if k not in data:
                data[k] = to_jsonable(v)
                added += 1
        if added:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        return patient, f"+{added}"
    except Exception as e:
        return patient, f"err:{str(e)[:60]}"


def main():
    args = parse_args()
    seg_dir = os.path.abspath(args.seg_dir)

    if args.vessel_csv:
        # 从 CSV 合并
        v = pd.read_csv(args.vessel_csv)
        idcol = "Patient_ID" if "Patient_ID" in v.columns else v.columns[0]
        lookup = {}
        for _, r in v.iterrows():
            lookup[str(r[idcol]).strip()] = {c: r[c] for c in VESSEL_COLS if c in v.columns}
        files = sorted(glob.glob(os.path.join(seg_dir, "*_radiomics.json")))
        done = skip = 0
        for f in files:
            patient = os.path.basename(f)[:-len("_radiomics.json")]
            if args.patients and patient not in set(args.patients.split(",")):
                continue
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            if "Vessel_Fractal_Dim" in data:
                skip += 1
                continue
            pid = str(data.get("Patient_ID", "")).strip()
            row = lookup.get(pid)
            if not row:
                # 回退 PatientID
                pid2 = str(data.get("PatientID", "")).strip()
                row = lookup.get(pid2)
            if not row:
                print(f"  [warn] {patient}: CSV 中无此患者")
                continue
            added = 0
            for k, val in row.items():
                if k not in data:
                    data[k] = None if (isinstance(val, float) and np.isnan(val)) else to_jsonable(val)
                    added += 1
            if added:
                with open(f, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, ensure_ascii=False)
            done += 1
            print(f"  [OK] {patient}: +{added} Vessel 键")
        print(f"\nCSV 合并完成：新增 {done}，已含跳过 {skip}")
    elif args.recompute_from_mask:
        files = sorted(glob.glob(os.path.join(seg_dir, "*_radiomics.json")))
        patients = [os.path.basename(f)[:-len("_radiomics.json")] for f in files]
        if args.patients:
            wanted = set(args.patients.split(","))
            patients = [p for p in patients if p in wanted]
        todo = [(p, seg_dir) for p in patients]
        with Pool(processes=args.workers) as pool:
            for patient, res in pool.imap_unordered(compute_from_mask, todo):
                print(f"  {patient}: {res}")
        print(f"\n掩膜现算完成：{len(patients)} 例")
    else:
        print("必须提供 --vessel-csv 或 --recompute-from-mask")


if __name__ == "__main__":
    main()
