"""
重算 missing/异常患者（完整 Fast 特征集），修复两个已知问题：

1. 死锁修复：不再用「mp.Process(每患者) -> 内层 Pool」的嵌套多进程（Windows spawn
   嵌套会 0% CPU 卡死）。改为「单层 Pool」：把 (患者 x 16 黄金掩膜) 拍平成一个大任务
   表一次 pool.map，worker 只做单掩膜 pyRadiomics，完全无嵌套。
2. 123 掩膜问题修复：只处理 KEEP_FILES 里的 16 个黄金靶区（脑/结肠/胆囊等无关器官
   掩膜直接忽略），避免 4444 键的脏 JSON，也避免白算一堆空器官。

用法：
  python recompute_missing_fast.py --seg-dir E:/DICOM/2026-02-seg \
      --nifti-dir E:/DICOM/2026-02-nifti \
      --list E:/DICOM/2026-02-seg/recompute_list.txt --skip 20130805 --workers 8
"""
import argparse
import glob
import json
import logging
import os
import sys
import time

import numpy as np
import SimpleITK as sitk
from multiprocessing import Pool

# 复用 fast 版的引擎/指标函数（模块级、无嵌套），KEEP_FILES 复用 lite 版
from compute_patient_radiomics_fast import (
    find_patients, resolve_ct_path, extract_patient_id, _build_extractor,
    _to_jsonable, lobe_emphysema_features, cardiopulmonary_features,
    airway_lobe_coupling, diaphragm_flattening, vessel_advanced_features,
)
from compute_patient_radiomics_lite import KEEP_FILES

logging.getLogger('radiomics').setLevel(logging.ERROR)


def parse_args():
    p = argparse.ArgumentParser(description="重算 missing/异常患者（完整 Fast 特征集，单层 Pool 防死锁）")
    p.add_argument("--seg-dir", "-s", required=True, help="分割结果目录（含 <患者>_masks/ 与 *_radiomics.json）")
    p.add_argument("--nifti-dir", "-n", required=True, help="原始 CT 患者目录")
    p.add_argument("--list", "-l", default=None, help="患者名单文件（每行一个患者名）；缺省=seg-dir 内全部患者")
    p.add_argument("--skip", default="", help="要跳过的患者（逗号分隔，如 20130805 或全名）")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4), help="Pool 进程数")
    return p.parse_args()


def _init_worker(threads):
    """每个 worker 固定 ITK 线程数，避免多进程各自吃满全核导致内存/调度爆炸。"""
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)


def _mask_job(args):
    """单掩膜 pyRadiomics（供 Pool.map；与 fast._worker_pyradiomics 等价但不改线程数）。"""
    ct_path, mask_path, roi_name = args
    try:
        ct = sitk.ReadImage(ct_path)
        mask = sitk.ReadImage(mask_path)
        ext = _build_extractor(roi_name)
        vec = ext.execute(ct, mask)
        out = {}
        for k, v in vec.items():
            if not k.startswith('diagnostics_'):
                out[k] = _to_jsonable(v)
        return out
    except Exception as e:
        print(f"    [warn] pyradiomics 失败 {os.path.basename(mask_path)}: {e}")
        return {}


def _assemble_patient(meta, nifti_dir, seg_dir, mask_names, roi_results):
    """在主进程内组装一个患者的完整特征 dict（含 4 类 COPD 指标 + 血管高级特征）。"""
    patient = meta["patient"]
    ct_path = resolve_ct_path(meta, nifti_dir)
    if ct_path is None:
        print(f"  [FAIL] {patient}: 找不到 CT")
        return None, None

    feats = {"Patient_ID": patient, "CT_Series": os.path.basename(ct_path)}
    feats["PatientID"] = extract_patient_id(meta, nifti_dir)
    for roi_name, roi_feats in roi_results.items():
        for k, v in roi_feats.items():
            feats[f"{roi_name}::{k}"] = v

    # 4 类 COPD 指标 + 血管高级特征（只加载黄金靶区，很快）
    mask_dir = meta["mask_dir"]
    masks = {}
    for mf in mask_names:
        p = os.path.join(mask_dir, mf)
        if os.path.isfile(p):
            masks[mf[:-len('.nii.gz')]] = sitk.ReadImage(p)
    mask_arrays = {k: sitk.GetArrayFromImage(v) for k, v in masks.items()}

    ct_img = sitk.ReadImage(ct_path)
    ct_arr = sitk.GetArrayFromImage(ct_img)
    spacing = ct_img.GetSpacing()

    feats.update(lobe_emphysema_features(ct_arr, spacing, mask_arrays))
    feats.update(cardiopulmonary_features(ct_arr, spacing, mask_arrays))
    feats.update(airway_lobe_coupling(mask_arrays, spacing))
    feats.update(diaphragm_flattening(ct_arr, mask_arrays))
    vessel = mask_arrays.get('lung_vessels')
    if vessel is not None:
        feats.update(vessel_advanced_features(vessel, spacing))

    out_json = os.path.join(seg_dir, f"{patient}_radiomics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(feats), f, indent=2, ensure_ascii=False)
    return feats, out_json


def main():
    args = parse_args()
    seg_dir = os.path.abspath(args.seg_dir)
    nifti_dir = os.path.abspath(args.nifti_dir)

    # 1) 选定患者集合
    all_patients = find_patients(seg_dir)
    wanted = None
    if args.list:
        with open(args.list, encoding="utf-8") as f:
            wanted = {ln.strip() for ln in f if ln.strip()}
    skip = [s.strip() for s in args.skip.split(",") if s.strip()]
    patients = [m for m in all_patients
                if (wanted is None or m["patient"] in wanted)
                and not any(m["patient"] == s or m["patient"].startswith(s) for s in skip)]
    print(f"候选患者 {len(all_patients)} -> 选中 {len(patients)} 个（前缀跳过 {len(skip)} 个规则）")

    # 2) 拍平任务表：(患者 x 16 黄金掩膜)
    plan = []          # (patient_meta, [存在的 KEEP_FILES 掩膜名])
    jobs = []          # (ct_path, mask_path, roi_name)
    for meta in patients:
        mask_dir = meta["mask_dir"]
        all_masks = sorted(f for f in os.listdir(mask_dir) if f.endswith('.nii.gz'))
        keep = [f for f in all_masks if f in KEEP_FILES]
        if len(all_masks) > len(keep):
            print(f"  [info] {meta['patient']}: 掩膜 {len(all_masks)} -> 保留黄金靶区 {len(keep)}"
                  f"（忽略 {len(all_masks)-len(keep)} 个无关器官）")
        ct_path = resolve_ct_path(meta, nifti_dir)
        if ct_path is None:
            print(f"  [FAIL] {meta['patient']}: 找不到 CT，跳过")
            continue
        plan.append((meta, keep))
        for mf in keep:
            jobs.append((ct_path, os.path.join(mask_dir, mf), mf[:-len('.nii.gz')]))
    print(f"任务表：{len(plan)} 个患者 x 共 {len(jobs)} 个掩膜任务")

    if not jobs:
        print("无任务，退出。")
        return

    # 3) 单层 Pool 跑全部掩膜 pyRadiomics
    # 关键：必须用【保序】的 pool.map / imap（不能用 imap_unordered！）。
    # 之前用 imap_unordered + 顺序索引配对，多 worker 并发完成顺序随机 →
    # 掩膜结果错位（心肌特征被标成 heart_atrium_right 等），产生脏 JSON。
    n_cpu = os.cpu_count() or 4
    itk_threads = max(1, n_cpu // args.workers)
    t0 = time.time()
    with Pool(processes=args.workers, initializer=_init_worker, initargs=(itk_threads,)) as pool:
        results = pool.map(_mask_job, jobs)   # 保序：results[i] 对应 jobs[i]
    assert len(results) == len(jobs), f"结果数 {len(results)} != 任务数 {len(jobs)}"
    print(f"pyRadiomics 全部完成：{len(jobs)} 个掩膜，耗时 {time.time()-t0:.1f}s "
          f"(Pool={args.workers}, ITK线程/worker={itk_threads})")

    # 4) 按患者归组
    per_patient = {}
    idx = 0
    for meta, keep in plan:
        roi = {}
        for mf in keep:
            roi[mf[:-len('.nii.gz')]] = results[idx]
            idx += 1
        per_patient[meta["patient"]] = (meta, keep, roi)

    # 5) 主进程组装 + 写 JSON（覆盖）
    ok, failed = [], []
    for meta, keep, roi in per_patient.values():
        patient = meta["patient"]
        t1 = time.time()
        feats, out_json = _assemble_patient(meta, nifti_dir, seg_dir, keep, roi)
        if feats is None:
            failed.append(patient)
            continue
        n_feat = len(feats) - 2
        ok.append(patient)
        print(f"  [OK] {patient}: 特征维度 {n_feat}，耗时 {time.time()-t1:.1f}s -> {os.path.basename(out_json)}")

    print(f"\n完成！成功 {len(ok)}/{len(plan)}，失败 {len(failed)} 个。")
    if failed:
        print("  失败名单:")
        for p in failed:
            print(f"    - {p}")


if __name__ == "__main__":
    main()
