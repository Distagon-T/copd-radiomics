#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一流水线：TotalSegmentator 分割 + PyRadiomics 特征提取
=========================================================
在 Docker 容器内运行，对一批患者完成：
  1) TotalSegmentator -> 16 黄金靶区掩膜 (肺叶×5, 肺血管, 气管支气管,
                         主动脉, 肺动脉, 气管, 心脏+四腔, 心肌)
  2) PyRadiomics      -> 每个掩膜的 shape/firstorder 特征 + 四类 COPD 表型指标
                         输出 <患者>_radiomics.json
  3) 合并 CSV         -> 全部患者汇成 radiomics_all_patients.csv

断点续传:
  - 患者掩膜已齐全(>=16)且存在 info json -> 跳过分割，只做特征提取
  - 患者 radiomics.json 已存在            -> 完全跳过（除非 --force）

用法 (容器内):
  python /app/run_pipeline.py --input-dir /data/input --output-dir /data/output \
                              --device cuda:0 [--force] [--seg-only] [--radiomics-only]
"""
import argparse
import glob
import json
import os
import sys
import time

# ---- 导入本地模块 (同目录) ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radiomics_extract import (  # noqa: E402
    extract_patient_radiomics, _to_jsonable,
)
from merge_radiomics import merge_to_csv  # noqa: E402

# TotalSegmentator 黄金 16 靶区 (与下游特征提取匹配)
KEEP_FILES = [
    "lung_upper_lobe_left.nii.gz", "lung_lower_lobe_left.nii.gz",
    "lung_upper_lobe_right.nii.gz", "lung_middle_lobe_right.nii.gz",
    "lung_lower_lobe_right.nii.gz",
    "lung_vessels.nii.gz", "lung_trachea_bronchia.nii.gz",
    "aorta.nii.gz", "pulmonary_artery.nii.gz", "trachea.nii.gz",
    "heart.nii.gz",
    "heart_myocardium.nii.gz", "heart_atrium_left.nii.gz",
    "heart_ventricle_left.nii.gz", "heart_atrium_right.nii.gz",
    "heart_ventricle_right.nii.gz",
]

CONTRAST_KEYWORD = "+C"


def parse_args():
    p = argparse.ArgumentParser(description="TotalSeg 分割 + PyRadiomics 提取")
    p.add_argument("--input-dir", "-i", required=True, help="患者 NIfTI 目录")
    p.add_argument("--output-dir", "-o", required=True, help="输出目录（含 <患者>_masks 与 json/csv）")
    p.add_argument("--device", "-d", default="cuda:0", help="推理设备 (cuda:0/cuda:1/cpu)")
    p.add_argument("--force", action="store_true", help="radiomics json 已存在也重算")
    p.add_argument("--seg-only", action="store_true", help="只分割，不做特征提取")
    p.add_argument("--radiomics-only", action="store_true", help="只做特征提取（用已有掩膜）")
    return p.parse_args()


def find_niftis(patient_dir):
    return sorted(glob.glob(os.path.join(patient_dir, "*.nii")) +
                  glob.glob(os.path.join(patient_dir, "*.nii.gz")))


def read_dicom_info(patient_dir):
    for f in os.listdir(patient_dir):
        if f.endswith("_dicom_info.json"):
            with open(os.path.join(patient_dir, f), encoding="utf-8") as fh:
                return json.load(fh)
    return None


def pick_input_nifti(patient_dir):
    """自动选输入：优先 +C 增强序列 -> 层数最多序列 -> 唯一/最大层数 nii。"""
    import nibabel as nib
    niftis = find_niftis(patient_dir)
    if not niftis:
        return None, None, "无 NIfTI"
    data = read_dicom_info(patient_dir)
    if data and data.get("Series"):
        contrast = [s for s in data["Series"]
                    if CONTRAST_KEYWORD in s.get("Series", {}).get("SeriesDescription", "")]
        series_list = contrast if contrast else data["Series"]
        best, best_n, best_folder = None, -1, None
        for s in series_list:
            folder = str(s.get("SeriesFolder", ""))
            if not folder:
                continue
            cand = os.path.join(patient_dir, f"{os.path.basename(patient_dir)}_{folder}.nii.gz")
            if not os.path.isfile(cand):
                continue
            inst = s.get("Instances")
            n = int(inst) if isinstance(inst, (int, float)) else None
            if n is None:
                try:
                    n = int(nib.load(cand).shape[2])
                except Exception:
                    n = 0
            if n > best_n:
                best, best_n, best_folder = cand, n, folder
        if best:
            return best, best_n, "对比增强序列" if contrast else "层数最多序列"
    if len(niftis) == 1:
        return niftis[0], int(nib.load(niftis[0]).shape[2]), "唯一NIfTI"
    best, best_n = None, -1
    for f in niftis:
        try:
            n = int(nib.load(f).shape[2])
        except Exception:
            n = 0
        if n > best_n:
            best, best_n = f, n
    return best, best_n, "最大层数"


def run_totalsegmentator(nifti_path, out_dir, device):
    """三个引擎：全器官 -> 肺微观 -> 高精心血管；最后清理到 16 靶区。"""
    from totalsegmentator.python_api import totalsegmentator
    print("  [TotalSeg] 全器官 + 血管 + 心脏 ...")
    totalsegmentator(nifti_path, out_dir, device=device)
    print("  [TotalSeg] 肺血管/支气管树 (LEGACY) ...")
    totalsegmentator(nifti_path, out_dir, task="lung_vessels_LEGACY", device=device)
    print("  [TotalSeg] 高精心血管 ...")
    totalsegmentator(nifti_path, out_dir, task="heartchambers_highres", device=device)
    # 清理：只保留黄金 16 靶区
    for f in os.listdir(out_dir):
        if f.endswith(".nii.gz") and f not in KEEP_FILES:
            try:
                os.remove(os.path.join(out_dir, f))
            except OSError:
                pass
    masks = sorted(f for f in os.listdir(out_dir) if f.endswith(".nii.gz"))
    print(f"  [TotalSeg] 完成，保留 {len(masks)} 个靶区掩膜。")
    return masks


def process_patient(patient_dir, output_base, device, args):
    patient_name = os.path.basename(patient_dir.rstrip("/\\"))
    info = {
        "patient_folder": patient_name,
        "input_dir": patient_dir,
        "selected_nifti": None,
        "slice_count": None,
        "selection_note": None,
        "totalseg_masks": [],
        "radiomics_json": None,
        "status": "pending",
        "elapsed_seconds": None,
        "error": None,
    }
    t0 = time.time()
    out_dir = os.path.join(output_base, f"{patient_name}_masks")
    info_json = os.path.join(out_dir, f"{patient_name}_segmentation_info.json")
    radiomics_json = os.path.join(output_base, f"{patient_name}_radiomics.json")

    # ============ 阶段一：分割（可跳过） ============
    if not args.radiomics_only:
        nifti_path, n_slices, note = pick_input_nifti(patient_dir)
        if nifti_path is None:
            info["status"] = "failed"
            info["error"] = note
            return info
        info["selected_nifti"] = os.path.basename(nifti_path)
        info["slice_count"] = n_slices
        info["selection_note"] = note
        print(f"  选中: {os.path.basename(nifti_path)} ({note}, {n_slices} 层)")

        os.makedirs(out_dir, exist_ok=True)
        existing = [f for f in os.listdir(out_dir) if f.endswith(".nii.gz")]
        if len(existing) >= len(KEEP_FILES) and os.path.exists(info_json):
            print(f"  ⏭️ 掩膜已齐全 ({len(existing)} 个)，跳过分割。")
            info["totalseg_masks"] = sorted(existing)
        else:
            try:
                info["totalseg_masks"] = run_totalsegmentator(nifti_path, out_dir, device)
                info["selected_nifti"] = os.path.basename(nifti_path)
                info["slice_count"] = n_slices
                info["selection_note"] = note
                with open(info_json, "w", encoding="utf-8") as f:
                    json.dump(info, f, indent=2, ensure_ascii=False)
            except Exception as e:
                info["status"] = "failed"
                info["error"] = str(e)
                info["elapsed_seconds"] = round(time.time() - t0, 2)
                with open(info_json, "w", encoding="utf-8") as f:
                    json.dump(info, f, indent=2, ensure_ascii=False)
                print(f"  ❌ {patient_name} 分割失败: {e}")
                return info
    else:
        # radiomics-only: 掩膜已存在
        os.makedirs(out_dir, exist_ok=True)
        info["totalseg_masks"] = sorted(f for f in os.listdir(out_dir) if f.endswith(".nii.gz"))
        info["selected_nifti"] = "from existing masks"

    # ============ 阶段二：特征提取（可跳过） ============
    if not args.seg_only:
        if not args.radiomics_only:
            # 分割模式：用选中的 nifti
            ct_path = os.path.join(patient_dir, info["selected_nifti"])
        else:
            ct_path = info.get("selected_nifti", "") or ""
            if not os.path.isfile(ct_path):
                ct_path = pick_input_nifti(patient_dir)[0]
        if ct_path and os.path.isfile(ct_path):
            try:
                print(f"  [Radiomics] 提取特征中 ...")
                feats = extract_patient_radiomics(ct_path, out_dir, patient_name)
                with open(radiomics_json, "w", encoding="utf-8") as f:
                    json.dump(_to_jsonable(feats), f, indent=2, ensure_ascii=False)
                info["radiomics_json"] = os.path.basename(radiomics_json)
                print(f"  [Radiomics] {patient_name}: {len(feats)-2} 特征")
            except Exception as e:
                info["status"] = "failed"
                info["error"] = f"radiomics: {e}"
                info["elapsed_seconds"] = round(time.time() - t0, 2)
                print(f"  ❌ {patient_name} 特征提取失败: {e}")
                return info

    info["elapsed_seconds"] = round(time.time() - t0, 2)
    info["status"] = "success"
    with open(info_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  ✅ {patient_name} 完成，耗时 {info['elapsed_seconds']/60:.1f} 分钟")
    return info


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("TotalSeg 分割 + PyRadiomics 提取流水线")
    print(f"输入: {input_dir}")
    print(f"输出: {output_dir}")
    print(f"设备: {args.device}")
    print("=" * 60)

    patients = sorted(d for d in os.listdir(input_dir)
                      if os.path.isdir(os.path.join(input_dir, d)))
    if not patients:
        # 单患者目录
        patients = [os.path.basename(input_dir.rstrip("/\\"))]
        input_dir = os.path.dirname(input_dir.rstrip("/\\"))

    results = []
    for i, p in enumerate(patients, 1):
        print(f"\n[{i}/{len(patients)}] 患者: {p}")
        info = process_patient(os.path.join(input_dir, p), output_dir, args.device, args)
        results.append(info)

    # 合并 CSV（传入 input_dir 以便提取 PatientID）
    if not args.seg_only:
        print("\n合并全部患者 radiomics -> CSV ...")
        try:
            csv_path = merge_to_csv(output_dir, nifti_dir=input_dir)
            print(f"CSV: {csv_path}")
        except Exception as e:
            print(f"合并 CSV 失败: {e}")

    summary = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "device": args.device,
        "total": len(results),
        "success": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "patients": results,
    }
    with open(os.path.join(output_dir, "segmentation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 60)
    print(f"完成！成功 {summary['success']} / 失败 {summary['failed']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
