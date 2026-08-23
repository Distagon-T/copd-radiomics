# -*- coding: utf-8 -*-
"""
Batch airway segmentation inference script for Connectivity-Aware-Airway-Segmentaion.

遍历「已转换的」患者文件夹（例如 <nifti_dir>）：
  1. 每个患者文件夹读取 <患者名>_dicom_info.json（由 dcm2nii_batch_arg.py 生成）；
  2. 依据 JSON 中的序列信息，找到层数最多的那个 .nii.gz（若 JSON 缺失则直接用 nibabel 量 z 轴层数）；
  3. 用 AirwayExtractionModel 对选中的 NIfTI 做气道分割；
  4. 结果保存到输出目录，并按患者生成 inference json + 汇总 json。

Usage examples:
  python run_inference_batch.py -i <nifti_dir> -o <output_dir>
  python run_inference_batch.py -i <nifti_dir> -o <output_dir> -d cuda:0
"""
from pathlib import Path
import argparse
import sys
import os
import json
import time
import traceback

import nibabel as nib


def parse_args():
    p = argparse.ArgumentParser(description="批量气道分割：遍历患者文件夹，选层数最多的 NIfTI 进行推理")
    p.add_argument("--input-dir", "-i", required=True,
                   help="患者 NIfTI 文件夹所在目录（每个子文件夹 = 一个患者，内含 dicom_info.json）")
    p.add_argument("--output-dir", "-o", required=True, help="目录用于保存气道分割结果")
    p.add_argument("--device", "-d", default=None, help="Override device (e.g. cpu or cuda:0)")
    p.add_argument("--recursive", "-r", action="store_true",
                   help="递归搜索患者子文件夹（默认只处理 input-dir 直接子目录）")
    return p.parse_args()


def read_dicom_info(patient_dir):
    """读取患者文件夹内的 <患者名>_dicom_info.json；不存在返回 None。"""
    for f in os.listdir(patient_dir):
        if f.endswith("_dicom_info.json"):
            with open(os.path.join(patient_dir, f), encoding="utf-8") as fh:
                return json.load(fh)
    return None


def find_largest_slice_nifti(patient_dir):
    """
    找到患者文件夹中层数最多的 .nii.gz 文件。
    优先依据 dicom_info.json 中的 Instances（DICOM 实例数=层数）判断；
    若 JSON 缺失则用 nibabel 读取 z 轴尺寸。
    返回 (最佳文件绝对路径, 层数)；找不到返回 (None, None)。
    """
    data = read_dicom_info(patient_dir)

    # 依据 JSON 的序列信息选层数最多者
    if data and data.get("Series"):
        best = None
        for s in data["Series"]:
            folder = str(s.get("SeriesFolder", ""))
            instances = s.get("Instances")
            if not folder:
                continue
            nii = os.path.join(patient_dir, f"{os.path.basename(patient_dir)}_{folder}.nii.gz")
            if not os.path.isfile(nii):
                continue
            n_slices = int(instances) if isinstance(instances, (int, float)) else None
            if n_slices is None:
                try:
                    img = nib.load(nii)
                    shape = img.shape
                    n_slices = int(shape[2]) if len(shape) >= 3 else int(shape[0])
                except Exception:
                    continue
            if best is None or n_slices > best[1]:
                best = (nii, n_slices)
        if best:
            return best

    # JSON 缺失或无法匹配时的兜底：直接用 nibabel 找 z 轴层数最多
    nii_files = sorted([f for f in os.listdir(patient_dir) if f.endswith((".nii", ".nii.gz"))])
    best_path, best_slices = None, -1
    for f in nii_files:
        try:
            img = nib.load(os.path.join(patient_dir, f))
            shape = img.shape
            n_slices = int(shape[2]) if len(shape) >= 3 else int(shape[0])
            if n_slices > best_slices:
                best_slices, best_path = n_slices, os.path.join(patient_dir, f)
        except Exception:
            continue
    return (best_path, best_slices) if best_path else (None, None)


def find_patient_dirs(input_dir: Path, recursive: bool):
    """收集患者文件夹（input-dir 的直接子目录；recursive 时递归）。"""
    if recursive:
        return sorted([d for d in input_dir.rglob("*") if d.is_dir() and list(d.glob("*.nii*"))])
    return sorted([d for d in input_dir.iterdir() if d.is_dir()])


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    # device 覆盖
    if args.device is not None:
        try:
            from configs import airway_config
            airway_config.config['device'] = args.device
        except Exception:
            pass

    # 导入模型与工具
    from models.airway_model import AirwayExtractionModel
    from util.utils import load_itk_image, save_itk

    print("Instantiating model...")
    model = AirwayExtractionModel()

    patient_dirs = find_patient_dirs(input_dir, args.recursive)
    if not patient_dirs:
        print(f"No patient folders found in {input_dir}")
        return

    print(f"Found {len(patient_dirs)} patient folders. Starting inference...\n")
    overall_start = time.time()
    results = []
    success_count = 0
    skip_count = 0
    fail_count = 0

    for idx, pdir in enumerate(patient_dirs, 1):
        patient_name = pdir.name
        patient_out_dir = output_dir / patient_name
        info_json_path = patient_out_dir / f"{patient_name}_inference_info.json"
        print(f"\n[{idx}/{len(patient_dirs)}] {patient_name}")

        info = {
            "patient_folder": patient_name,
            "input_dir": str(pdir),
            "selected_nifti": None,
            "slice_count": None,
            "status": "pending",
            "elapsed_seconds": None,
            "error": None,
            "output_files": [],
        }

        # 【断点续传】
        if patient_out_dir.exists() and info_json_path.exists():
            airway_file = list(patient_out_dir.glob("*_airway.nii.gz"))
            if airway_file:
                info["status"] = "skipped"
                info["selected_nifti"] = str(airway_file[0].name)
                info["output_files"] = [f.name for f in patient_out_dir.iterdir() if f.is_file()]
                results.append(info)
                skip_count += 1
                print(f"   ⏭️ 已存在气道结果，跳过。")
                continue

        # 选层数最多的 NIfTI
        nifti_path, n_slices = find_largest_slice_nifti(str(pdir))
        if nifti_path is None:
            info["status"] = "failed"
            info["error"] = "未找到任何 .nii/.nii.gz 文件"
            results.append(info)
            fail_count += 1
            print(f"   ❌ {info['error']}")
            continue

        info["selected_nifti"] = os.path.basename(nifti_path)
        info["slice_count"] = n_slices
        patient_out_dir.mkdir(parents=True, exist_ok=True)

        try:
            patient_start = time.time()
            print(f"   🎯 选中: {info['selected_nifti']} (层数 {n_slices})")

            image, origin, spacing, direction = load_itk_image(nifti_path)
            pred = model.predict(image)

            out_file = patient_out_dir / f"{patient_name}_airway.nii.gz"
            save_itk(pred, str(out_file), origin, spacing, direction)

            info["output_files"] = sorted(f.name for f in patient_out_dir.iterdir() if f.is_file())
            info["elapsed_seconds"] = round(time.time() - patient_start, 2)
            info["status"] = "success"
            with open(info_json_path, "w", encoding="utf-8") as fh:
                json.dump(info, fh, indent=2, ensure_ascii=False)

            success_count += 1
            print(f"   ✅ 完成，耗时 {info['elapsed_seconds']/60:.2f} 分钟 -> {out_file}")

        except Exception as e:
            info["status"] = "failed"
            info["error"] = str(e)
            info["elapsed_seconds"] = round(time.time() - patient_start, 2)
            with open(info_json_path, "w", encoding="utf-8") as fh:
                json.dump(info, fh, indent=2, ensure_ascii=False)
            fail_count += 1
            print(f"   ❌ 失败: {e}")
            traceback.print_exc()

        results.append(info)

    # 汇总 json
    summary_path = output_dir / "inference_summary.json"
    summary = {
        "input_base_dir": str(input_dir),
        "output_base_dir": str(output_dir),
        "total_patients": len(patient_dirs),
        "success": success_count,
        "skipped": skip_count,
        "failed": fail_count,
        "total_elapsed_seconds": round(time.time() - overall_start, 2),
        "patients": results,
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    overall_time = (time.time() - overall_start) / 3600
    print("\n" + "=" * 50)
    print("🎉 批量推理结束！")
    print(f"⏱️ 总耗时: {overall_time:.2f} 小时")
    print(f"📊 统计: 成功 {success_count} 例，跳过 {skip_count} 例，失败 {fail_count} 例")
    print(f"💾 汇总信息: {summary_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
