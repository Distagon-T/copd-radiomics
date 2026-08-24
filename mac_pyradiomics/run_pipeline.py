#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mac 版统一流水线：TotalSegmentator 分割（CPU）+ PyRadiomics 特征提取
====================================================================
pyradiomics 是纯 CPU 计算，无 CUDA 依赖，可在 Mac 上原生运行。
TotalSegmentator 在 Mac 上无 CUDA，只能走 CPU（较慢，但可用）。

两种用法：
  1) 完整流程：TotalSeg 分割 + PyRadiomics（--device cpu）
  2) 只做特征提取：复用已有的 <患者>_masks 分割结果（--radiomics-only），
     无需再分割，速度快很多 —— 推荐在 Mac 上使用此模式

输出: <患者>_radiomics.json + radiomics_all_patients.csv

用法:
  python run_pipeline.py --input-dir /path/patients --output-dir /path/out [--radiomics-only]
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
    p = argparse.ArgumentParser(description="Mac: TotalSeg(CPU) + PyRadiomics 提取")
    p.add_argument("--input-dir", "-i", required=True, help="患者 NIfTI 目录")
    p.add_argument("--output-dir", "-o", required=True, help="输出目录")
    p.add_argument("--device", "-d", default="cpu", help="TotalSeg 设备 (cpu/mps)")
    p.add_argument("--force", action="store_true", help="radiomics json 已存在也重算")
    p.add_argument("--seg-only", action="store_true", help="只分割，不做特征提取")
    p.add_argument("--radiomics-only", action="store_true",
                   help="只做特征提取（复用已有掩膜，Mac 上推荐）")
    p.add_argument("--workers", type=int, default=2,
                   help="掩膜并行进程数（Mac 建议 2~3，内存不足用 1）")
    p.add_argument("--patients", default=None,
                   help="只处理指定患者文件夹（逗号分隔，与文件夹名精确匹配），默认全部")
    p.add_argument("--timeout", type=int, default=2400,
                   help="单患者超时秒数（默认 2400=40min；超时自动终止该患者并继续下一个，避免整体卡死）")
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
    """Mac 上 TotalSegmentator 走 CPU（无 CUDA）。三引擎 + 清理到 16 靶区。"""
    from totalsegmentator.python_api import totalsegmentator
    print(f"  [TotalSeg] 全器官 + 血管 + 心脏 (device={device}) ...")
    totalsegmentator(nifti_path, out_dir, device=device)
    print("  [TotalSeg] 肺血管/支气管树 (LEGACY) ...")
    totalsegmentator(nifti_path, out_dir, task="lung_vessels_LEGACY", device=device)
    print("  [TotalSeg] 高精心血管 ...")
    totalsegmentator(nifti_path, out_dir, task="heartchambers_highres", device=device)
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
        os.makedirs(out_dir, exist_ok=True)
        existing = [f for f in os.listdir(out_dir) if f.endswith(".nii.gz")]
        if len(existing) >= len(KEEP_FILES) and os.path.exists(info_json):
            print(f"  ⏭️ 掩膜已齐全 ({len(existing)} 个)，跳过分割。")
            info["totalseg_masks"] = sorted(existing)
        else:
            try:
                info["totalseg_masks"] = run_totalsegmentator(nifti_path, out_dir, device)
                with open(info_json, "w", encoding="utf-8") as f:
                    json.dump(info, f, indent=2, ensure_ascii=False)
            except Exception as e:
                info["status"] = "failed"
                info["error"] = str(e)
                print(f"  ❌ {patient_name} 分割失败: {e}")
                return info
    else:
        os.makedirs(out_dir, exist_ok=True)
        info["totalseg_masks"] = sorted(f for f in os.listdir(out_dir) if f.endswith(".nii.gz"))
        info["selected_nifti"] = "from existing masks"

    # ============ 阶段二：特征提取（可跳过） ============
    if not args.seg_only:
        if not args.radiomics_only:
            ct_path = os.path.join(patient_dir, info["selected_nifti"])
        else:
            ct_path = info.get("selected_nifti", "") or ""
            if not os.path.isfile(ct_path):
                ct_path = pick_input_nifti(patient_dir)[0]
        if ct_path and os.path.isfile(ct_path):
            try:
                print(f"  [Radiomics] 提取特征中 (纯 CPU, 无 GPU 依赖) ...")
                feats = extract_patient_radiomics(ct_path, out_dir, patient_name,
                                                  workers=args.workers)
                with open(radiomics_json, "w", encoding="utf-8") as f:
                    json.dump(_to_jsonable(feats), f, indent=2, ensure_ascii=False)
                info["radiomics_json"] = os.path.basename(radiomics_json)
                print(f"  [Radiomics] {patient_name}: {len(feats)-2} 特征")
            except Exception as e:
                info["status"] = "failed"
                info["error"] = f"radiomics: {e}"
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
    print("Mac: TotalSeg(CPU) + PyRadiomics 流水线")
    print(f"输入: {input_dir}")
    print(f"输出: {output_dir}")
    print(f"设备: {args.device}")
    print("=" * 60)

    patients = sorted(d for d in os.listdir(input_dir)
                      if os.path.isdir(os.path.join(input_dir, d)))
    if not patients:
        patients = [os.path.basename(input_dir.rstrip("/\\"))]
        input_dir = os.path.dirname(input_dir.rstrip("/\\"))

    # --patients 过滤：只处理指定患者文件夹（逗号分隔，按文件夹名精确匹配）
    if args.patients:
        wanted = {s.strip() for s in args.patients.split(",") if s.strip()}
        patients = [p for p in patients if p in wanted]
        missing = sorted(wanted - set(patients))
        if missing:
            print(f"  [warn] 以下患者文件夹未在输入目录中找到: {missing}")
        print(f"  [过滤] 本次仅处理 {len(patients)} 个患者: {patients}")

    results = []
    for i, p in enumerate(patients, 1):
        print(f"\n[{i}/{len(patients)}] 患者: {p}")
        info_json = os.path.join(output_dir, f"{p}_masks", f"{p}_segmentation_info.json")
        # 关键修复：每个患者跑在独立子进程 + 墙钟超时：某个患者挂死只跳过它，整批不再卡死
        proc = mp.Process(target=process_patient,
                          args=(os.path.join(input_dir, p), output_dir, args.device, args),
                          daemon=False)
        proc.start()
        proc.join(args.timeout)
        if proc.is_alive():
            print(f"  [TIMEOUT] {p} 超过 {args.timeout}s 未完成，终止并跳过（其余患者继续）")
            proc.terminate()
            proc.join(10)
            if proc.is_alive():
                proc.kill()
                proc.join()
            results.append({"patient_folder": p, "status": "timeout"})
            continue
        # 子进程正常结束：读回其写出的 info json
        if os.path.exists(info_json):
            try:
                with open(info_json, encoding="utf-8") as f:
                    results.append(json.load(f))
            except Exception:
                results.append({"patient_folder": p, "status": "failed"})
        else:
            print(f"  [FAIL] {p} 未生成 info json")
            results.append({"patient_folder": p, "status": "failed"})

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
        "failed": sum(1 for r in results if r["status"] in ("failed", "timeout")),
        "timeout": sum(1 for r in results if r["status"] == "timeout"),
        "patients": results,
    }
    with open(os.path.join(output_dir, "segmentation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 60)
    print(f"完成！成功 {summary['success']} / 失败 {summary['failed']}（含超时 {summary['timeout']}）")
    print("=" * 60)


if __name__ == "__main__":
    main()
