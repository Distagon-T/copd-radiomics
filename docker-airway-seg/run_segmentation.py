#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一分割入口：TotalSegmentator + Connectivity-Aware Airway Segmentation
=====================================================================
在 Docker 容器内运行，完成一个患者的全部分割：
  1) TotalSegmentator  -> 全器官/肺叶/血管/心脏 (16 靶区，黄金子集)
  2) AirwaySegModel    -> 精细气道树 (connectivity-aware)

输入形态自动识别：
  - 患者目录含 <患者>_dicom_info.json：依据 Series 选层数最多 / 增强(+C) 序列
  - 患者目录只有单个 .nii.gz：直接分割
  - 患者目录有多个 .nii.gz：选 z 层数最多者

用法 (容器内)：
  python /app/run_segmentation.py --input-dir /data/input \
                                  --output-dir /data/output \
                                  --device cuda:0 [--airway-only] [--totalseg-only]
"""
import argparse
import glob
import json
import os
import sys
import time

import nibabel as nib

# ---- 保持代码可移植：添加 airway 代码路径 ----
# 容器内默认 /app/airway_seg；本地调试可用环境变量 AIRWAY_CODE_DIR 覆盖。
# airway_model.py 内部基于 __file__ 解析 'checkpoints/airway_model.pth'，
# 与 cwd 无关，因此无需 chdir。
AIRWAY_CODE = os.environ.get("AIRWAY_CODE_DIR", "/app/airway_seg")
if os.path.isdir(AIRWAY_CODE):
    sys.path.insert(0, AIRWAY_CODE)

# TotalSegmentator 黄金 16 靶区
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
    p = argparse.ArgumentParser(description="TotalSeg + Airway 统一分割")
    p.add_argument("--input-dir", "-i", required=True, help="患者 NIfTI 目录")
    p.add_argument("--output-dir", "-o", required=True, help="分割结果输出目录")
    p.add_argument("--device", "-d", default="cuda:0", help="推理设备 (cuda:0 / cuda:1 / cpu)")
    p.add_argument("--airway-only", action="store_true", help="只跑气道分割")
    p.add_argument("--totalseg-only", action="store_true", help="只跑 TotalSegmentator")
    p.add_argument("--totalseg-device", default=None,
                   help="TotalSegmentator 设备 (默认同 --device)")
    return p.parse_args()


def find_niftis(patient_dir):
    """返回目录下所有 .nii / .nii.gz 绝对路径。"""
    return sorted(glob.glob(os.path.join(patient_dir, "*.nii")) +
                  glob.glob(os.path.join(patient_dir, "*.nii.gz")))


def read_dicom_info(patient_dir):
    for f in os.listdir(patient_dir):
        if f.endswith("_dicom_info.json"):
            with open(os.path.join(patient_dir, f), encoding="utf-8") as fh:
                return json.load(fh)
    return None


def pick_input_nifti(patient_dir):
    """
    自动选择输入 NIfTI：
      1) 有 dicom_info.json 且有 +C 增强序列 -> 选增强序列 (层数最多者)
      2) 有 dicom_info.json -> 选层数最多序列
      3) 只有一个 nii.gz -> 直接用它
      4) 多个 nii.gz -> 用 nibabel 选 z 层数最多者
    返回 (nifti_path, n_slices, note)。
    """
    niftis = find_niftis(patient_dir)
    if not niftis:
        return None, None, "无 NIfTI"

    data = read_dicom_info(patient_dir)
    if data and data.get("Series"):
        # 优先增强序列
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
            note = "对比增强序列" if contrast else "层数最多序列"
            return best, best_n, note

    # 兜底
    if len(niftis) == 1:
        n = int(nib.load(niftis[0]).shape[2])
        return niftis[0], n, "唯一NIfTI"
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
    from totalsegmentator.python_api import totalsegmentator
    print(f"  [TotalSeg] 全器官 + 血管 + 心脏 ...")
    totalsegmentator(nifti_path, out_dir, device=device)
    print(f"  [TotalSeg] 肺血管/支气管树 (LEGACY) ...")
    totalsegmentator(nifti_path, out_dir, task="lung_vessels_LEGACY", device=device)
    print(f"  [TotalSeg] 高精心血管 ...")
    totalsegmentator(nifti_path, out_dir, task="heartchambers_highres", device=device)

    # 清理无关器官
    for f in os.listdir(out_dir):
        if f.endswith(".nii.gz") and f not in KEEP_FILES:
            try:
                os.remove(os.path.join(out_dir, f))
            except OSError:
                pass
    masks = sorted(f for f in os.listdir(out_dir) if f.endswith(".nii.gz"))
    print(f"  [TotalSeg] 完成，保留 {len(masks)} 个靶区掩膜。")
    return masks


def load_airway_model(device):
    """加载气道模型（只调用一次，供所有患者复用）。"""
    from models.airway_model import AirwayExtractionModel
    print(f"  [Airway] 加载气道模型 ...")
    if device:
        try:
            from configs import airway_config
            airway_config.config["device"] = device
        except Exception:
            pass
    return AirwayExtractionModel()


def run_airway(model, nifti_path, out_dir, device, patient_name):
    from util.utils import load_itk_image, save_itk

    image, origin, spacing, direction = load_itk_image(nifti_path)
    print(f"  [Airway] 推理中 ({nifti_path}) ...")
    pred = model.predict(image)
    # 与本机命名保持一致：掩膜用 <患者>_airway.nii.gz
    out_file = os.path.join(out_dir, patient_name + "_airway.nii.gz")
    save_itk(pred, out_file, origin, spacing, direction)
    print(f"  [Airway] 完成 -> {os.path.basename(out_file)}")
    return out_file


def process_patient(patient_dir, output_base, device, args, airway_model=None):
    patient_name = os.path.basename(patient_dir.rstrip("/\\"))
    info = {
        "patient_folder": patient_name,
        "input_dir": patient_dir,
        "selected_nifti": None,
        "slice_count": None,
        "selection_note": None,
        "totalseg_masks": [],
        "airway_mask": None,
        "status": "pending",
        "elapsed_seconds": None,
        "error": None,
    }
    t0 = time.time()

    # 选输入
    nifti_path, n_slices, note = pick_input_nifti(patient_dir)
    if nifti_path is None:
        info["status"] = "failed"
        info["error"] = note
        return info
    info["selected_nifti"] = os.path.basename(nifti_path)
    info["slice_count"] = n_slices
    info["selection_note"] = note
    print(f"  选中: {os.path.basename(nifti_path)} ({note}, {n_slices} 层)")

    out_dir = os.path.join(output_base, f"{patient_name}_masks")
    os.makedirs(out_dir, exist_ok=True)
    info_json = os.path.join(out_dir, f"{patient_name}_segmentation_info.json")

    # 断点续传：16 靶区齐全且已完成即跳过
    if not args.airway_only:
        existing = [f for f in os.listdir(out_dir) if f.endswith(".nii.gz")]
        if len(existing) >= len(KEEP_FILES) and os.path.exists(info_json):
            print(f"  ⏭️ 已存在完整 TotalSeg 结果，跳过。")
            info["status"] = "skipped"
            info["totalseg_masks"] = sorted(existing)
            return info

    try:
        if not args.airway_only:
            info["totalseg_masks"] = run_totalsegmentator(nifti_path, out_dir,
                                                          args.totalseg_device or device)
        if not args.totalseg_only:
            info["airway_mask"] = run_airway(airway_model, nifti_path, out_dir, device, patient_name)

        info["elapsed_seconds"] = round(time.time() - t0, 2)
        info["status"] = "success"
        with open(info_json, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print(f"  ✅ {patient_name} 完成，耗时 {info['elapsed_seconds']/60:.1f} 分钟")
    except Exception as e:
        info["status"] = "failed"
        info["error"] = str(e)
        info["elapsed_seconds"] = round(time.time() - t0, 2)
        with open(info_json, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print(f"  ❌ {patient_name} 失败: {e}")
    return info


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("统一气道/器官分割流水线")
    print(f"输入: {input_dir}")
    print(f"输出: {output_dir}")
    print(f"设备: {args.device}")
    print("=" * 60)

    # 收集患者（子文件夹）
    patients = sorted(d for d in os.listdir(input_dir)
                      if os.path.isdir(os.path.join(input_dir, d)))
    if not patients:
        # 输入目录本身就是单患者文件目录
        patients = [os.path.basename(input_dir.rstrip("/\\"))]
        input_dir = os.path.dirname(input_dir.rstrip("/\\"))
        if not any(f.endswith((".nii", ".nii.gz"))
                   for f in os.listdir(os.path.join(input_dir, patients[0]))):
            print("输入目录下没有找到患者文件夹或 NIfTI。")
            sys.exit(1)

    results = []

    # 🔑 关键修复：气道模型只在循环外加载一次，供所有患者复用
    # （之前每个患者都重新 AirwayExtractionModel()，CUDA 显存碎片/内存
    #   持续累积，多患者场景下会显著变慢甚至 OOM。）
    airway_model = None
    if not args.totalseg_only:
        airway_model = load_airway_model(args.device)

    for i, p in enumerate(patients, 1):
        print(f"\n[{i}/{len(patients)}] 患者: {p}")
        info = process_patient(os.path.join(input_dir, p), output_dir,
                               args.device, args, airway_model)
        results.append(info)

    # 汇总
    summary = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "device": args.device,
        "total": len(results),
        "success": sum(1 for r in results if r["status"] == "success"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "patients": results,
    }
    with open(os.path.join(output_dir, "segmentation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 60)
    print(f"完成！成功 {summary['success']} / 跳过 {summary['skipped']} / 失败 {summary['failed']}")
    print(f"汇总: {os.path.join(output_dir, 'segmentation_summary.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
