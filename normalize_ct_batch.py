# -*- coding: utf-8 -*-
"""
CT 各向同性重采样归一化脚本（1x1x1 mm 默认）
================================================
对每个患者：
  1. 读取 <患者>_dicom_info.json 中的 Series 信息，按 Instances(层数) 选最大的 CT；
     （json 缺失时兜底用 nibabel 量 z 轴层数）
  2. 把选中的最大层数 .nii.gz 重采样为 1x1x1 mm 各向同性体素（保持 origin/direction）；
  3. 存储为 <原名>_normalized.nii.gz（--out-suffix 可改，如 _normalized.nii）；
  4. 把归一化信息写回 json（顶层 Normalization dict + 选中 Series 打标 SelectedForNormalization）。

用法:
  python normalize_ct_batch.py
  python normalize_ct_batch.py -i E:/DICOM/2026-05-nifti --spacing 1 1 1 --interp bspline
  python normalize_ct_batch.py -i E:/DICOM/2026-05-nifti --patients "id1,id2" --force
  python normalize_ct_batch.py -i E:/DICOM/2026-05-nifti --out-suffix _normalized.nii --no-compress
说明:
  - 串行处理（每例 ~512x512x~500 层，重采样内存峰值 ~1GB，串行最安全，避免 OOM）。
  - 断点续传：若 _normalized 文件已存在且 json 已标记，则跳过；--force 强制重跑。
  - 插值默认 sitkLinear；--interp bspline 可换 BSpline（更平滑但慢）。
  - 输出默认 .nii.gz（压缩，省磁盘）；如确实需要纯 .nii 用 --no-compress + --out-suffix。
"""
import os
import sys
import json
import time
import glob
import argparse
import csv

import numpy as np
import nibabel as nib
import SimpleITK as sitk

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

INTERP_MAP = {"linear": sitk.sitkLinear, "bspline": sitk.sitkBSpline}


def parse_args():
    p = argparse.ArgumentParser(description="CT 各向同性重采样归一化")
    p.add_argument("--nifti-dir", "-i", default=r"E:\DICOM\2026-05-nifti",
                   help="患者 NIfTI 根目录（每个子文件夹=一个患者）")
    p.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                   help="目标体素间距 x y z (mm)，默认 1 1 1")
    p.add_argument("--interp", choices=list(INTERP_MAP), default="linear",
                   help="插值方式：linear 或 bspline，默认 linear")
    p.add_argument("--patients", default=None,
                   help="只处理指定患者（逗号分隔），默认全部")
    p.add_argument("--limit", type=int, default=None,
                   help="只处理前 N 个患者（测试用）")
    p.add_argument("--force", action="store_true", help="强制重跑（忽略断点续传标记）")
    p.add_argument("--out-suffix", default="_normalized.nii.gz",
                   help="输出文件名后缀，默认 _normalized.nii.gz")
    p.add_argument("--compress", dest="compress", action="store_true", default=True,
                   help="gzip 压缩输出（默认）")
    p.add_argument("--no-compress", dest="compress", action="store_false",
                   help="不压缩（配合 --out-suffix _normalized.nii 输出纯 .nii）")
    p.add_argument("--log", default=None, help="增量日志文件路径（默认 <nifti-dir>/normalize_run.log）")
    return p.parse_args()


def read_dicom_info(patient_dir):
    """读取患者文件夹内的 <患者名>_dicom_info.json；不存在返回 None。"""
    for f in os.listdir(patient_dir):
        if f.endswith("_dicom_info.json"):
            try:
                with open(os.path.join(patient_dir, f), encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                return None
    return None


def find_largest_slice_nifti(patient_dir):
    """
    依据 dicom_info.json 的 Series[].Instances 选层数最多的 .nii.gz。
    返回 (nii绝对路径, 层数, SeriesFolder)；找不到返回 (None, None, None)。
    """
    data = read_dicom_info(patient_dir)
    pname = os.path.basename(patient_dir)

    if data and data.get("Series"):
        best = None
        for s in data["Series"]:
            folder = str(s.get("SeriesFolder", ""))
            instances = s.get("Instances")
            if not folder:
                continue
            nii = os.path.join(patient_dir, f"{pname}_{folder}.nii.gz")
            if not os.path.isfile(nii):
                continue
            n_slices = int(instances) if isinstance(instances, (int, float)) else None
            if n_slices is None:
                try:
                    img = nib.load(nii)
                    shp = img.shape
                    n_slices = int(shp[2]) if len(shp) >= 3 else int(shp[0])
                except Exception:
                    continue
            if best is None or n_slices > best[1]:
                best = (nii, n_slices, folder)
        if best:
            return best

    # 兜底：json 缺失/无匹配时用 nibabel 量 z 轴层数
    nii_files = sorted(glob.glob(os.path.join(patient_dir, "*.nii*")))
    best_path, best_slices, best_folder = None, -1, None
    for f in nii_files:
        try:
            img = nib.load(f)
            shp = img.shape
            n_slices = int(shp[2]) if len(shp) >= 3 else int(shp[0])
            if n_slices > best_slices:
                best_path, best_slices = f, n_slices
        except Exception:
            continue
    return (best_path, best_slices, None) if best_path else (None, None, None)


def resample_isotropic(image, target_spacing, interpolator):
    """把 SimpleITK 图像重采样为 target_spacing (x,y,z) 各向同性，保持 origin/direction。"""
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    new_size = [int(round(sz * sp / ts)) for sz, sp, ts in zip(orig_size, orig_spacing, target_spacing)]
    new_size = [s if s >= 1 else 1 for s in new_size]

    rf = sitk.ResampleImageFilter()
    rf.SetOutputSpacing(list(target_spacing))
    rf.SetOutputDirection(image.GetDirection())
    rf.SetOutputOrigin(image.GetOrigin())
    rf.SetSize(new_size)
    rf.SetInterpolator(interpolator)
    rf.SetDefaultPixelValue(-1024)  # 越界体素填空气 HU
    return rf.Execute(image)


def update_json_with_normalization(info_json, nii_src, nii_norm, folder, target_spacing, shape, interp_name):
    """把归一化信息写回 dicom_info.json（保留原字段，追加 Normalization）。"""
    if not os.path.exists(info_json):
        return None
    with open(info_json, encoding="utf-8") as f:
        data = json.load(f)

    data["Normalization"] = {
        "source_series_folder": folder if folder else "",
        "source_nifti": os.path.basename(nii_src),
        "normalized_nifti": os.path.basename(nii_norm),
        "target_spacing": [float(x) for x in target_spacing],
        "shape": [int(x) for x in shape],
        "interpolator": interp_name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # 给选中的 Series 打标
    for s in data.get("Series", []):
        if str(s.get("SeriesFolder", "")) == str(folder):
            s["SelectedForNormalization"] = True
            s["NormalizedNifti"] = os.path.basename(nii_norm)
            break

    tmp = info_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, info_json)
    return data


def main():
    args = parse_args()
    root = os.path.abspath(args.nifti_dir)
    spacing = tuple(args.spacing)
    interp = INTERP_MAP[args.interp]
    log_path = args.log or os.path.join(root, "normalize_run.log")

    LOG = open(log_path, "w", encoding="utf-8")
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        LOG.write(line + "\n")
        LOG.flush()

    dirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if args.patients:
        want = {x.strip() for x in args.patients.split(",") if x.strip()}
        dirs = [d for d in dirs if d in want]
    if args.limit:
        dirs = dirs[: args.limit]

    log(f"扫描 {root}: {len(dirs)} 个患者，目标间距 {list(spacing)}mm，插值 {args.interp}")
    log(f"输出后缀: {args.out_suffix} (compress={args.compress})，日志: {log_path}")

    ok = skip = fail = 0
    results = []
    t0 = time.time()
    for i, pname in enumerate(dirs, 1):
        pdir = os.path.join(root, pname)
        nii_src, n_slices, folder = find_largest_slice_nifti(pdir)
        if nii_src is None:
            log(f"[{i}/{len(dirs)}] {pname}: 未找到任何 CT，跳过")
            fail += 1
            continue

        stem = os.path.splitext(os.path.basename(nii_src))[0]  # 去掉 .gz
        if stem.endswith(".nii"):
            stem = stem[:-4]
        out_name = stem + args.out_suffix
        out_path = os.path.join(pdir, out_name)
        info_json = os.path.join(pdir, pname + "_dicom_info.json")

        # 断点续传
        if (not args.force) and os.path.exists(out_path):
            try:
                with open(info_json, encoding="utf-8") as f:
                    existing = json.load(f)
                if existing.get("Normalization", {}).get("normalized_nifti") == out_name:
                    log(f"[{i}/{len(dirs)}] {pname}: 已存在归一化输出，跳过 (层数={n_slices})")
                    skip += 1
                    results.append((pname, os.path.basename(nii_src), out_name, "skip"))
                    continue
            except Exception:
                pass

        t_p = time.time()
        try:
            img = sitk.ReadImage(nii_src)
            resampled = resample_isotropic(img, spacing, interp)
            sitk.WriteImage(resampled, out_path, useCompression=args.compress)
            shape = resampled.GetSize()
            new_spacing = resampled.GetSpacing()
            update_json_with_normalization(info_json, nii_src, out_path, folder, spacing, shape, args.interp)
            dt = time.time() - t_p
            log(f"[{i}/{len(dirs)}] {pname}: OK 层数={n_slices} "
                f"shape={tuple(shape)} spacing={tuple(round(x,4) for x in new_spacing)} "
                f"-> {out_name} ({dt:.1f}s)")
            ok += 1
            results.append((pname, os.path.basename(nii_src), out_name, "ok"))
        except Exception as e:
            log(f"[{i}/{len(dirs)}] {pname}: FAIL {e}")
            fail += 1
            results.append((pname, os.path.basename(nii_src), out_name, "fail"))

    total = time.time() - t0
    log(f"\n完成: OK={ok} 跳过={skip} 失败={fail}  总耗时 {total/60:.1f} min")

    # 结果清单
    res_csv = os.path.join(root, "normalize_results.csv")
    with open(res_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["patient", "source_nifti", "normalized_nifti", "status"])
        w.writerows(results)
    log(f"结果清单: {res_csv}")
    LOG.close()


if __name__ == "__main__":
    main()
