#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯 PyRadiomics Lite 特征提取入口（Mac 版）
==========================================
只读取【已有的 <患者>_masks 分割结果】+ 原始 CT，用**旧 Lite 策略**计算特征：
所有 ROI 只用 shape + firstorder（无 LoG / 无 Wavelet / 无纹理），
加上 11 个 Vessel/BV 高级特征 + 四类 COPD 表型指标。

与 Fast 版（run_radiomics.py / run_pipeline.py）的区别：
  - Lite：全部 16 掩膜 shape+firstorder，速度快（约 1~2 分钟/患者）
  - Fast：肺叶加 LoG+纹理、心肌加 Wavelet，特征更多但稍慢
⚠️ 两者特征列不同，同一队列只能用一种模式，勿混用。

pyradiomics 是纯 CPU 计算（SimpleITK + numpy），无 CUDA 依赖，Mac 原生运行。

输入结构（与 TotalSegmentator 输出一致）：
  seg_dir/<患者>_masks/                # 16 个 .nii.gz 掩膜
  seg_dir/<患者>_masks/<患者>_segmentation_info.json   # 含 selected_nifti 定位 CT

用法：
  python run_radiomics_lite.py --nifti-dir /path/CT --seg-dir /path/seg \
                               [--patients id1,id2] [--force] [--workers 2]
"""
import argparse
import glob
import json
import os
import sys
import time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radiomics_extract import extract_patient_radiomics, _to_jsonable  # noqa: E402
from merge_radiomics import merge_to_csv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Mac 纯 PyRadiomics Lite 特征提取（全 ROI shape+firstorder）")
    p.add_argument("--nifti-dir", "-n", required=True, help="原始 CT 患者目录（含 <患者>/xxx.nii.gz）")
    p.add_argument("--seg-dir", "-s", required=True, help="分割结果目录（含 <患者>_masks/）")
    p.add_argument("--patients", default=None, help="只处理指定患者（逗号分隔），默认全部")
    p.add_argument("--force", action="store_true", help="radiomics json 已存在也重算")
    p.add_argument("--skip-merge", action="store_true", help="跳过合并 CSV")
    p.add_argument("--workers", type=int, default=2,
                   help="掩膜并行进程数（Mac 建议 2~3，内存不足用 1）")
    p.add_argument("--timeout", type=int, default=2400,
                   help="单患者超时秒数（默认 2400=40min；超时自动终止该患者并继续下一个，避免整体卡死）")
    return p.parse_args()


def find_patients(seg_dir):
    """从 seg 目录收集患者（<患者>_masks 文件夹）及其信息 json。"""
    patients = []
    for d in sorted(os.listdir(seg_dir)):
        if not d.endswith("_masks") or not os.path.isdir(os.path.join(seg_dir, d)):
            continue
        patient = d[:-len("_masks")]
        info_json = os.path.join(seg_dir, d, f"{patient}_segmentation_info.json")
        patients.append({"patient": patient, "mask_dir": os.path.join(seg_dir, d),
                         "info_json": info_json})
    return patients


def resolve_ct_path(patient_meta, nifti_dir):
    """优先从 segmentation_info.json 的 selected_nifti 定位 CT；否则搜索患者目录。"""
    if os.path.exists(patient_meta["info_json"]):
        try:
            with open(patient_meta["info_json"], encoding="utf-8") as f:
                info = json.load(f)
            sel = info.get("selected_nifti", "")
            for cand in [sel,
                         os.path.join(nifti_dir, patient_meta["patient"], sel),
                         os.path.join(info.get("input_dir", ""), sel)]:
                if cand and os.path.isfile(cand):
                    return cand
            ind = info.get("input_dir", "")
            if ind:
                files = glob.glob(os.path.join(ind, "*.nii.gz"))
                if files:
                    return sorted(files)[0]
        except Exception:
            pass
    pdir = os.path.join(nifti_dir, patient_meta["patient"])
    if os.path.isdir(pdir):
        files = glob.glob(os.path.join(pdir, "*.nii.gz"))
        if files:
            return sorted(files)[0]
    return None


def process_patient(meta, nifti_dir, seg_dir, force=False, workers=2):
    patient = meta["patient"]
    mask_dir = meta["mask_dir"]
    out_json = os.path.join(seg_dir, f"{patient}_radiomics.json")

    if os.path.exists(out_json) and not force:
        print(f"  [skip] {patient} 已存在 radiomics json，跳过（--force 重算）")
        return None

    ct_path = resolve_ct_path(meta, nifti_dir)
    if ct_path is None:
        print(f"  [FAIL] {patient}: 找不到 CT")
        return None
    print(f"  CT: {os.path.basename(ct_path)}")

    t0 = time.time()
    # Lite 策略：全部 ROI 只用 shape+firstorder
    feats = extract_patient_radiomics(ct_path, mask_dir, patient, workers=workers, lite=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(feats), f, indent=2, ensure_ascii=False)
    print(f"  [OK] {patient}: {len(feats)-2} 特征, 耗时 {time.time()-t0:.0f}s -> {os.path.basename(out_json)}")
    return feats


def main():
    args = parse_args()
    nifti_dir = os.path.abspath(args.nifti_dir)
    seg_dir = os.path.abspath(args.seg_dir)

    patients = find_patients(seg_dir)
    if args.patients:
        wanted = set(args.patients.split(","))
        patients = [p for p in patients if p["patient"] in wanted]

    print(f"发现 {len(patients)} 个患者，开始 Lite 特征提取（纯 CPU；单患者超时 {args.timeout}s）...")
    results = []
    failed = []
    for i, meta in enumerate(patients, 1):
        patient = meta["patient"]
        out_json = os.path.join(seg_dir, f"{patient}_radiomics.json")
        if os.path.exists(out_json) and not args.force:
            print(f"\n[{i}/{len(patients)}] {patient}  [skip] 已存在 radiomics json")
            continue
        print(f"\n[{i}/{len(patients)}] {patient}")
        # 关键修复：每个患者跑在独立子进程 + 墙钟超时：挂死的患者只跳过，整批不再卡死
        proc = mp.Process(target=process_patient,
                          args=(meta, nifti_dir, seg_dir, args.force, args.workers),
                          daemon=False)
        proc.start()
        proc.join(args.timeout)
        if proc.is_alive():
            print(f"  [TIMEOUT] {patient} 超过 {args.timeout}s 未完成，终止并跳过（其余患者继续）")
            proc.terminate()
            proc.join(10)
            if proc.is_alive():
                proc.kill()
                proc.join()
            failed.append(patient)
            continue
        if os.path.exists(out_json):
            try:
                with open(out_json, encoding="utf-8") as f:
                    results.append(json.load(f))
            except Exception as e:
                print(f"  [warn] 读取 {patient} json 失败: {e}")
                failed.append(patient)
        else:
            print(f"  [FAIL] {patient} 未生成 radiomics json")
            failed.append(patient)

    print(f"\n完成！成功 {len(results)}/{len(patients)}，失败/超时 {len(failed)} 个患者。")
    if failed:
        print("  失败/超时名单:")
        for p in failed:
            print(f"    - {p}")

    if results and not args.skip_merge:
        print("\n合并全部患者 -> CSV ...")
        try:
            csv_path = merge_to_csv(seg_dir, nifti_dir=nifti_dir)
            print(f"CSV: {csv_path}")
        except Exception as e:
            print(f"合并 CSV 失败: {e}")


if __name__ == "__main__":
    main()
