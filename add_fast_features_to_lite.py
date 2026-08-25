#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
给已有 Lite radiomics JSON 补上 Fast 独有的特征（只加缺失键，不动已有值）。
================================================================================
Fast vs Lite 的差异（仅以下两组，其余 ROI / 4 类 COPD 指标 / Vessel* 两版相同）：
  1) 5 个肺叶（lung_*_lobe_*）: original 的 GLCM/GLRLM/GLSZM 纹理
                               + LoG(sigma=[1.0]) 的 shape/firstorder/GLCM/GLRLM/GLSZM
  2) heart_myocardium         : original 的 GLCM/GLRLM/GLSZM/GLDM/NGTDM
                               + 8 个 Wavelet 子带全特征

用法：
  python add_fast_features_to_lite.py -n E:/DICOM/2026-02-nifti -s E:/DICOM/2026-02-seg \\
                                      --workers 4 [--patients id1,id2]
"""
import argparse
import glob
import json
import logging
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.getLogger('radiomics').setLevel(logging.ERROR)
for _lg in list(logging.root.manager.loggerDict):
    if _lg.startswith('radiomics'):
        logging.getLogger(_lg).setLevel(logging.ERROR)

# 与 compute_patient_radiomics_fast.py 完全一致
BASE_SETTINGS = {'binWidth': 25, 'force2D': False, 'voxelArrayShift': 1000,
                 'interpolator': sitk.sitkBSpline, 'preCrop': True}

LOBE_ROIS = ["lung_upper_lobe_left", "lung_lower_lobe_left",
             "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
MYOCARDIUM_ROI = "heart_myocardium"


def parse_args():
    p = argparse.ArgumentParser(description="给 Lite radiomics JSON 补 Fast 增量特征")
    p.add_argument("--nifti-dir", "-n", required=True, help="原始 CT 患者目录")
    p.add_argument("--seg-dir", "-s", required=True, help="分割结果目录（含 <患者>_masks 与 <患者>_radiomics.json）")
    p.add_argument("--patients", default=None, help="只处理指定患者（逗号分隔），默认全部")
    p.add_argument("--workers", type=int, default=4, help="并行进程数（默认 4）")
    return p.parse_args()


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


def build_extractor(roi):
    """与 Fast 版一致的引擎：肺叶 LoG[1.0]+纹理；心肌 Wavelet。"""
    ext = featureextractor.RadiomicsFeatureExtractor(**BASE_SETTINGS)
    if roi == MYOCARDIUM_ROI:
        ext.enableAllFeatures()
        ext.enableImageTypeByName('Wavelet')
    else:
        ext.enableAllFeatures()
        ext.enableFeaturesByName(shape=[], firstorder=[], glcm=[], glrlm=[], glszm=[])
        ext.enableImageTypeByName('LoG', customArgs={'sigma': [1.0]})
    return ext


def resolve_ct_path(info_json, patient, nifti_dir):
    if os.path.exists(info_json):
        try:
            with open(info_json, encoding="utf-8") as f:
                info = json.load(f)
            sel = info.get("selected_nifti", "")
            for cand in [sel,
                         os.path.join(info.get("input_dir", ""), sel),
                         os.path.join(nifti_dir, patient, sel)]:
                if cand and os.path.isfile(cand):
                    return cand
            ind = info.get("input_dir", "")
            if ind:
                files = glob.glob(os.path.join(ind, "*.nii.gz"))
                if files:
                    return sorted(files)[0]
        except Exception:
            pass
    pdir = os.path.join(nifti_dir, patient)
    if os.path.isdir(pdir):
        files = glob.glob(os.path.join(pdir, "*.nii.gz"))
        if files:
            best, bn = None, -1
            for f in files:
                try:
                    n = sitk.ReadImage(f).GetSize()[2]
                except Exception:
                    n = 0
                if n > bn:
                    best, bn = f, n
            return best
    return None


def worker(args):
    """单患者：对 6 个 ROI 跑 Fast 引擎，把缺失键并入现有 JSON。返回 (patient, added, err)。"""
    patient, nifti_dir, seg_dir = args
    json_path = os.path.join(seg_dir, f"{patient}_radiomics.json")
    if not os.path.exists(json_path):
        return patient, 0, "no_json"
    mask_dir = os.path.join(seg_dir, f"{patient}_masks")
    if not os.path.isdir(mask_dir):
        return patient, 0, "no_masks"
    info_json = os.path.join(mask_dir, f"{patient}_segmentation_info.json")

    ct_path = resolve_ct_path(info_json, patient, nifti_dir)
    if ct_path is None:
        return patient, 0, "no_ct"

    try:
        # 关键：每个 worker 只用 1 个 ITK 线程，避免多 worker 内存爆炸
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        ct_img = sitk.ReadImage(ct_path)
        added = 0
        for roi in LOBE_ROIS + [MYOCARDIUM_ROI]:
            mask_path = os.path.join(mask_dir, f"{roi}.nii.gz")
            if not os.path.exists(mask_path):
                continue
            try:
                mask_img = sitk.ReadImage(mask_path)
                ext = build_extractor(roi)
                vec = ext.execute(ct_img, mask_img)
                for k, v in vec.items():
                    if k.startswith('diagnostics_'):
                        continue
                    key = f"{roi}::{k}"
                    if key not in data:
                        data[key] = to_jsonable(v)
                        added += 1
            except Exception as e:
                print(f"    [warn] {patient} {roi}: {e}", flush=True)
        if added > 0:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        return patient, added, None
    except Exception as e:
        return patient, 0, str(e)


def main():
    args = parse_args()
    nifti_dir = os.path.abspath(args.nifti_dir)
    seg_dir = os.path.abspath(args.seg_dir)

    patients = []
    for f in sorted(glob.glob(os.path.join(seg_dir, "*_radiomics.json"))):
        p = os.path.basename(f)[:-len("_radiomics.json")]
        if os.path.isdir(os.path.join(seg_dir, f"{p}_masks")):
            patients.append(p)
    if args.patients:
        wanted = set(args.patients.split(","))
        patients = [p for p in patients if p in wanted]
    print(f"待处理患者: {len(patients)}（补 Fast 增量特征：肺叶 LoG+纹理 / 心肌 Wavelet）")

    todo = [(p, nifti_dir, seg_dir) for p in patients]
    t0 = time.time()
    ok = 0
    errs = []
    if args.workers > 1 and len(todo) > 1:
        with Pool(processes=args.workers) as pool:
            for patient, added, err in pool.imap_unordered(worker, todo):
                if err:
                    errs.append((patient, err))
                    print(f"  [FAIL] {patient}: {err}", flush=True)
                else:
                    ok += 1
                    if added:
                        print(f"  [OK] {patient}: +{added} 特征", flush=True)
    else:
        for t in todo:
            patient, added, err = worker(t)
            if err:
                errs.append((patient, err))
                print(f"  [FAIL] {patient}: {err}", flush=True)
            else:
                ok += 1
                if added:
                    print(f"  [OK] {patient}: +{added} 特征", flush=True)

    print(f"\n完成！成功 {ok}/{len(patients)}，失败 {len(errs)}，总耗时 {time.time()-t0:.0f}s")
    if errs:
        print("失败名单:")
        for p, e in errs:
            print(f"  - {p}: {e}")


if __name__ == "__main__":
    main()
