# -*- coding: utf-8 -*-
"""
掩膜重采样脚本：把「原始 CT 空间」的已有分割掩膜对齐到「归一化 CT」
=====================================================================
背景：分割（TotalSegmentator 3 引擎）是流程中最耗时的一步。既然归一化 CT
（normalize_ct_batch.py 产物，1x1x1 mm 各向同性）和原始 CT 是同一扫描、同一
空间（origin/direction 一致，仅 spacing/size 不同），就可以**复用已有掩膜**，
仅用最近邻插值重采样到归一化 CT 的网格，避免重新分割。

关键规则：
  - 掩膜（Masks）  ：只能使用 **最近邻插值 (sitkNearestNeighbor)**，
                     标签是离散整数，线性/样条插值会造出 1.5 这类无意义值。
  - 输出几何       ：以归一化 CT 为 reference image（SetReferenceImage），
                     保证 origin/direction/spacing/size 与归一化 CT 完全一致，
                     从而逐体素对齐。

用法:
  python resample_masks_to_normalized.py
  python resample_masks_to_normalized.py -n E:/DICOM/2026-05-nifti \
      -s E:/DICOM/2026-05-seg -o E:/DICOM/2026-05-seg-normalized
  python resample_masks_to_normalized.py -n E:/DICOM/2026-05-nifti \
      -s E:/DICOM/2026-05-seg -o E:/DICOM/2026-05-seg-normalized --patients "id1,id2"

说明:
  - 每个患者读取 <seg-dir>/<患者>_masks/ 中的 16 个黄金靶区掩膜；
  - 每个掩膜以归一化 CT 为参考做最近邻重采样，写入 <out-dir>/<患者>_masks/；
  - 校验：重采样后 spacing=(1,1,1)、size 与归一化 CT 一致、标签集合与源掩膜一致、
    非空（源非空但结果为空则告警）；
  - 串行处理（内存安全），断点续传（目标掩膜齐全且 info json 标记 success 则跳过）。
"""
import os
import sys
import glob
import json
import time
import argparse
import csv

import numpy as np
import SimpleITK as sitk

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 🌟 黄金 16 靶区（与 batch_segment_largest_slice.py / batch_segment_normalized.py 一致）
KEEP_FILES = [
    "lung_upper_lobe_left.nii.gz", "lung_lower_lobe_left.nii.gz",
    "lung_upper_lobe_right.nii.gz", "lung_middle_lobe_right.nii.gz", "lung_lower_lobe_right.nii.gz",
    "lung_vessels.nii.gz", "lung_trachea_bronchia.nii.gz",
    "aorta.nii.gz", "pulmonary_artery.nii.gz", "trachea.nii.gz",
    "heart.nii.gz",
    "heart_myocardium.nii.gz", "heart_atrium_left.nii.gz", "heart_ventricle_left.nii.gz",
    "heart_atrium_right.nii.gz", "heart_ventricle_right.nii.gz",
]


def parse_args():
    p = argparse.ArgumentParser(description="把已有分割掩膜最近邻重采样到归一化 CT 网格")
    p.add_argument("-n", "--nifti-dir", default=r"E:\DICOM\2026-05-nifti",
                   help="患者 NIfTI 根目录（含 *_normalized.nii*）")
    p.add_argument("-s", "--seg-dir", default=r"E:\DICOM\2026-05-seg",
                   help="原始分割根目录（含 <患者>_masks/ 与 16 个掩膜）")
    p.add_argument("-o", "--out-dir", default=r"E:\DICOM\2026-05-seg-normalized",
                   help="输出根目录（含 <患者>_masks/，默认 seg-normalized 后缀目录）")
    p.add_argument("--patients", default=None, help="只处理指定患者（逗号分隔），默认全部")
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 个患者（测试用）")
    p.add_argument("--force", action="store_true", help="强制重跑（忽略断点续传标记）")
    p.add_argument("--log", default=None, help="增量日志文件路径（默认 <out-dir>/resample_masks_run.log）")
    return p.parse_args()


def find_normalized_nifti(patient_dir):
    """在患者目录中找到归一化 CT（<患者>_normalized.nii / .nii.gz）。"""
    files = sorted(glob.glob(os.path.join(patient_dir, "*_normalized.nii*")))
    return files[0] if files else None


def resample_mask_nearest(mask_sitk, ref_sitk):
    """以 ref_sitk（归一化 CT）为输出几何，对掩膜做最近邻重采样。"""
    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(ref_sitk)          # 输出 origin/direction/spacing/size 与归一化 CT 一致
    rf.SetInterpolator(sitk.sitkNearestNeighbor)  # 掩膜只能用最近邻！
    out = rf.Execute(mask_sitk)
    # 强转回源掩膜的整数类型，确保标签精确无损
    if mask_sitk.GetPixelID() != out.GetPixelID():
        out = sitk.Cast(out, mask_sitk.GetPixelID())
    return out


def process_patient(patient_name, nifti_root, seg_root, out_root, force=False):
    """处理单个患者：把 16 个掩膜最近邻重采样到归一化 CT 网格。
    返回 (status, info_dict)。status: ok / skip / fail / no_mask / no_norm
    """
    info = {
        "patient_folder": patient_name,
        "normalized_nifti": None,
        "source_mask_dir": None,
        "output_mask_dir": None,
        "status": "pending",
        "elapsed_seconds": None,
        "error": None,
        "masks": [],
        "empty_warnings": [],
    }

    norm_path = find_normalized_nifti(os.path.join(nifti_root, patient_name))
    if norm_path is None:
        info["status"] = "no_norm"
        info["error"] = "未找到 *_normalized.nii(.gz)，请先运行 normalize_ct_batch.py"
        return info

    src_mask_dir = os.path.join(seg_root, f"{patient_name}_masks")
    if not os.path.isdir(src_mask_dir):
        info["status"] = "no_mask"
        info["error"] = f"未找到原始掩膜目录: {src_mask_dir}"
        return info

    info["normalized_nifti"] = os.path.basename(norm_path)
    info["source_mask_dir"] = src_mask_dir

    out_mask_dir = os.path.join(out_root, f"{patient_name}_masks")
    info["output_mask_dir"] = out_mask_dir
    info_json_path = os.path.join(out_mask_dir, f"{patient_name}_segmentation_info.json")

    # 断点续传：16 个掩膜齐全 + info json success 则跳过
    if not os.path.isdir(out_mask_dir):
        os.makedirs(out_mask_dir, exist_ok=True)
    existing = [f for f in os.listdir(out_mask_dir) if f.endswith(".nii.gz")]
    if (not force) and len(existing) >= len(KEEP_FILES) and os.path.exists(info_json_path):
        try:
            with open(info_json_path, encoding="utf-8") as f:
                prev = json.load(f)
            if prev.get("status") == "success":
                info["status"] = "skip"
                info["masks"] = sorted(existing)
                return info
        except Exception:
            pass

    t0 = time.time()
    try:
        norm_sitk = sitk.ReadImage(norm_path)
        ref_spacing = norm_sitk.GetSpacing()
        ref_size = norm_sitk.GetSize()

        empty_warnings = []
        for name in KEEP_FILES:
            src = os.path.join(src_mask_dir, name)
            if not os.path.exists(src):
                info["error"] = f"源掩膜缺失: {name}"
                info["status"] = "fail"
                return info
            mask_sitk = sitk.ReadImage(src)
            src_arr = sitk.GetArrayFromImage(mask_sitk)
            src_nonzero = int((src_arr > 0).sum())
            out_sitk = resample_mask_nearest(mask_sitk, norm_sitk)
            sitk.WriteImage(out_sitk, os.path.join(out_mask_dir, name), useCompression=True)
            out_arr = sitk.GetArrayFromImage(out_sitk)
            out_nonzero = int((out_arr > 0).sum())
            if src_nonzero > 0 and out_nonzero == 0:
                empty_warnings.append(name)
            # 标签集合一致性校验（向量化，大体积也快）
            src_labels = set(int(x) for x in np.unique(src_arr) if int(x) != 0)
            out_labels = set(int(x) for x in np.unique(out_arr) if int(x) != 0)
            if src_labels != out_labels:
                info["error"] = f"标签集不一致 {name}: src={src_labels} out={out_labels}"
                info["status"] = "fail"
                return info

        info["masks"] = sorted(f for f in os.listdir(out_mask_dir) if f.endswith(".nii.gz"))
        info["empty_warnings"] = empty_warnings
        info["elapsed_seconds"] = round(time.time() - t0, 2)
        info["status"] = "success"
        info["output_spacing"] = [float(x) for x in ref_spacing]
        info["output_size"] = [int(x) for x in ref_size]
        info["interpolator"] = "nearest_neighbor"

        with open(info_json_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        return info
    except Exception as e:
        info["status"] = "fail"
        info["error"] = str(e)
        info["elapsed_seconds"] = round(time.time() - t0, 2)
        try:
            with open(info_json_path, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return info


def main():
    args = parse_args()
    nifti_root = os.path.abspath(args.nifti_dir)
    seg_root = os.path.abspath(args.seg_dir)
    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)
    log_path = args.log or os.path.join(out_root, "resample_masks_run.log")

    LOG = open(log_path, "w", encoding="utf-8")
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        LOG.write(line + "\n")
        LOG.flush()

    patients = sorted(d for d in os.listdir(nifti_root)
                      if os.path.isdir(os.path.join(nifti_root, d)))
    if args.patients:
        want = {x.strip() for x in args.patients.split(",") if x.strip()}
        patients = [d for d in patients if d in want]
    if args.limit:
        patients = patients[: args.limit]

    log(f"归一化CT: {nifti_root} | 原始掩膜: {seg_root} | 输出: {out_root}")
    log(f"共 {len(patients)} 个患者，插值: nearest_neighbor（掩膜专用）")

    stats = {"success": 0, "skip": 0, "fail": 0, "no_norm": 0, "no_mask": 0}
    results = []
    t_all = time.time()
    for i, pname in enumerate(patients, 1):
        info = process_patient(pname, nifti_root, seg_root, out_root, force=args.force)
        st = info["status"]
        stats[st] = stats.get(st, 0) + 1
        results.append(info)
        if st == "success":
            extra = f"spacing={info.get('output_spacing')} size={info.get('output_size')}"
            if info["empty_warnings"]:
                extra += f" [空掩膜!] {info['empty_warnings']}"
            log(f"[{i}/{len(patients)}] {pname}: OK {len(info['masks'])} 掩膜 {extra}")
        elif st == "skip":
            log(f"[{i}/{len(patients)}] {pname}: 已存在，跳过")
        else:
            log(f"[{i}/{len(patients)}] {pname}: {st} - {info.get('error')}")

    total = time.time() - t_all
    log(f"\n完成: {dict(stats)}  总耗时 {total/60:.1f} min")

    res_csv = os.path.join(out_root, "resample_masks_results.csv")
    with open(res_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["patient", "status", "normalized_nifti", "error", "n_masks"])
        for r in results:
            w.writerow([r["patient_folder"], r["status"], r["normalized_nifti"] or "",
                        r.get("error") or "", len(r.get("masks") or [])])
    log(f"结果清单: {res_csv}")

    summary = {
        "nifti_dir": nifti_root,
        "seg_dir": seg_root,
        "out_dir": out_root,
        "interpolator": "nearest_neighbor",
        "stats": stats,
        "total_elapsed_seconds": round(total, 2),
        "patients": results,
    }
    with open(os.path.join(out_root, "resample_masks_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"汇总: {os.path.join(out_root, 'resample_masks_summary.json')}")
    LOG.close()


if __name__ == "__main__":
    main()
