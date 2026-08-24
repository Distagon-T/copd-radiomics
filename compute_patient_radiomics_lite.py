# -*- coding: utf-8 -*-
"""
逐患者 Radiomics 特征提取 - 精简版 (compute_patient_radiomics_lite.py)
========================================================================
专用于快速验证 & 并行池诊断。与全量版 (compute_patient_radiomics_fast.py)
的区别：

  1. 肺叶去掉 LoG 与全部纹理类（GLCM/GLRLM/GLSZM/GLDM/NGTDM）
     -> 只保留 shape + firstorder，单患者从 ~39 分钟降到 1~2 分钟
  2. 心肌也只用 shape + firstorder（去掉 Wavelet 全纹理）
     -> 消除最大计算热点
  3. worker 内打印诊断日志（PID / 掩膜名 / 耗时）
     -> 用于排查「并行池负载不均」问题

输出：<患者>_radiomics.json（与全量版同格式，可共用 merge_radiomics_to_csv.py）
全量版保留在 compute_patient_radiomics_fast.py（肺叶 LoG+纹理、心肌 Wavelet 全开）。

用法：
  python compute_patient_radiomics_lite.py -n <nifti_dir> \\
                                            -s <seg_dir> \\
                                            --patients <id> --force --workers 8
"""
import os
import sys
import glob
import json
import time
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
import scipy.ndimage as ndi
from skimage.morphology import skeletonize
from multiprocessing import Pool
import multiprocessing as mp
from radiomics import featureextractor
import logging

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.getLogger('radiomics').setLevel(logging.WARNING)
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(0)


def parse_args():
    p = argparse.ArgumentParser(description="逐患者 Radiomics（精简版/去纹理/诊断用）")
    p.add_argument("--nifti-dir", "-n", required=True, help="原始 CT 患者目录")
    p.add_argument("--seg-dir", "-s", required=True, help="分割结果目录（含 <患者>_masks/）")
    p.add_argument("--patients", default=None, help="只处理指定患者（逗号分隔），默认全部")
    p.add_argument("--force", action="store_true", help="已存在 json 也重算")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                   help="并行进程数")
    p.add_argument("--timeout", type=int, default=2400,
                   help="单患者超时秒数（默认 2400=40min；超时自动终止该患者并继续下一个，避免整体卡死）")
    return p.parse_args()


def find_patients(seg_dir):
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
    """
    定位原始 CT 路径，按优先级：
      1. segmentation_info.json 的 selected_nifti（最准确）
      2. segmentation_info.json 的 input_dir 里的 nii.gz
      3. 按「原始 CT 文件夹名 = 患者名」在 nifti_dir 下找同名文件夹，
         取其中层数最多的 nii.gz（兼容无 info json 的部分分割结果）
    返回 CT 绝对路径；找不到返回 None。
    """
    # ① 优先 segmentation_info.json
    if os.path.exists(patient_meta["info_json"]):
        try:
            with open(patient_meta["info_json"], encoding="utf-8") as f:
                info = json.load(f)
            sel = info.get("selected_nifti", "")
            for cand in [sel,
                         os.path.join(info.get("input_dir", ""), sel),
                         os.path.join(nifti_dir, patient_meta["patient"], sel)]:
                if cand and os.path.isfile(cand):
                    return cand
            ind = info.get("input_dir", "")
            if ind:
                files = glob.glob(os.path.join(ind, "*.nii.gz"))
                if files:
                    return sorted(files)[0]
        except Exception:
            pass

    # ② 兜底：原始 CT 文件夹名 = 患者名（= 分割文件夹名去掉 _masks），
    #    在 nifti_dir 下找同名文件夹并取层数最多的 nii.gz
    pdir = os.path.join(nifti_dir, patient_meta["patient"])
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


def extract_patient_id(patient_meta, nifti_dir):
    """
    提取 DICOM PatientID，按优先级：
      1. 分割 info json 的 series_info.candidates[].series_info.Patient.PatientID
      2. 回退：原始 CT 文件夹 <患者>/<患者>_dicom_info.json 的 Series[].Patient.PatientID
    找不到返回 None。
    """
    if os.path.exists(patient_meta["info_json"]):
        try:
            with open(patient_meta["info_json"], encoding="utf-8") as f:
                info = json.load(f)
            for c in info.get("series_info", {}).get("candidates", []):
                pid = c.get("series_info", {}).get("Patient", {}).get("PatientID")
                if pid:
                    return str(pid).strip()
        except Exception:
            pass
    # 回退到原始 CT 文件夹的 dicom_info.json
    pdir = os.path.join(nifti_dir, patient_meta["patient"])
    if os.path.isdir(pdir):
        for f in os.listdir(pdir):
            if f.endswith("_dicom_info.json"):
                try:
                    with open(os.path.join(pdir, f), encoding="utf-8") as fh:
                        data = json.load(fh)
                    for s in data.get("Series", []):
                        pid = s.get("Patient", {}).get("PatientID")
                        if pid:
                            return str(pid).strip()
                except Exception:
                    pass
                break
    return None


def _to_jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    return o


# =========================================================================
# PyRadiomics 引擎配置（精简版：全部只用 shape + firstorder，无纹理/无滤波）
# =========================================================================
# preCrop=True 裁剪到掩膜包围盒加速；ITK 多线程用满核心
BASE_SETTINGS = {'binWidth': 25, 'force2D': False, 'voxelArrayShift': 1000,
                 'interpolator': sitk.sitkBSpline, 'preCrop': True}

# lung_vessels 特殊优化：只算 firstorder，跳过最慢的 shape 特征
# （Maximum3DDiameter / SurfaceArea / Compactness 在肺血管几百万体素上
#   单独要 ~35 分钟，是 lite/fast 版共同的性能瓶颈；
#   只算 firstorder 可把单患者从 ~36 分钟降到 ~2-3 分钟）
LUNG_VESSELS_ONLY_FIRSTORDER = True


def _build_extractor(roi_name):
    """
    构建 shape+firstorder 引擎。
    - lung_vessels（肺血管）: 只算 firstorder（跳过最慢 shape，提速 ~175x）
    - 其余掩膜             : shape + firstorder
    """
    ext = featureextractor.RadiomicsFeatureExtractor(**BASE_SETTINGS)
    ext.disableAllFeatures()
    if LUNG_VESSELS_ONLY_FIRSTORDER and 'lung_vessels' in roi_name:
        ext.enableFeaturesByName(firstorder=[])
    else:
        ext.enableFeaturesByName(shape=[], firstorder=[])
    return ext


def _worker_pyradiomics(args):
    """单掩膜 pyRadiomics；带诊断日志：打印 worker PID、掩膜名、耗时。"""
    ct_path, mask_path, roi_name = args
    pid = os.getpid()
    t0 = time.time()
    try:
        # 关键修复：每个 worker 只用 1 个 ITK 线程。
        # 之前模块级 SetGlobalDefaultNumberOfThreads(0)=用满全部核心，4 个 worker
        # 各自开满核跑滤波 -> 内存成倍爆炸，遇到 500+ 层大 CT 直接挂死。
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        ct = sitk.ReadImage(ct_path)
        mask = sitk.ReadImage(mask_path)
        ext = _build_extractor(roi_name)
        vec = ext.execute(ct, mask)
        out = {}
        for k, v in vec.items():
            if not k.startswith('diagnostics_'):
                out[k] = _to_jsonable(v)
        dt = time.time() - t0
        print(f"      [worker {pid}] {roi_name}: {len(out)} 特征, 耗时 {dt:.1f}s", flush=True)
        return out
    except Exception as e:
        dt = time.time() - t0
        print(f"      [worker {pid}] {roi_name}: 失败 ({e}) 耗时 {dt:.1f}s", flush=True)
        return {}


# =========================================================================
# 四类 COPD 表型指标（与全量版一致）
# =========================================================================
LOBE_MAP = {
    'lung_upper_lobe_left': 'LLU', 'lung_lower_lobe_left': 'LLL',
    'lung_upper_lobe_right': 'RUL', 'lung_middle_lobe_right': 'RML',
    'lung_lower_lobe_right': 'RLL',
}


def lobe_emphysema_features(ct_arr, spacing, masks):
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    out = {}
    full_lung = np.zeros_like(ct_arr, dtype=bool)
    total_lung_vol = 0.0
    for mask_name, arr in masks.items():
        key = LOBE_MAP.get(mask_name)
        if key is None:
            continue
        lobe = arr > 0
        if lobe.sum() == 0:
            out[f'Lobe_{key}_LAA950_pct'] = np.nan
            out[f'Lobe_{key}_Perc15_HU'] = np.nan
            out[f'Lobe_{key}_Volume_mm3'] = 0.0
            out[f'Lobe_{key}_Vol_pct_of_lung'] = 0.0
            continue
        hu = ct_arr[lobe]
        vol = lobe.sum() * voxel_vol
        out[f'Lobe_{key}_LAA950_pct'] = float(np.sum(hu < -950) / len(hu) * 100)
        out[f'Lobe_{key}_Perc15_HU'] = float(np.percentile(hu, 15))
        out[f'Lobe_{key}_Volume_mm3'] = float(vol)
        full_lung |= lobe
        total_lung_vol += vol
    out['Lung_Total_Volume_mm3'] = float(total_lung_vol)
    if full_lung.sum() > 0:
        hu_all = ct_arr[full_lung]
        out['Lung_LAA950_pct'] = float(np.sum(hu_all < -950) / len(hu_all) * 100)
        out['Lung_Perc15_HU'] = float(np.percentile(hu_all, 15))
    for mask_name in LOBE_MAP:
        if f'Lobe_{LOBE_MAP[mask_name]}_Volume_mm3' in out and total_lung_vol > 0:
            out[f'Lobe_{LOBE_MAP[mask_name]}_Vol_pct_of_lung'] = \
                out[f'Lobe_{LOBE_MAP[mask_name]}_Volume_mm3'] / total_lung_vol
    return out


def cardiopulmonary_features(ct_arr, spacing, masks):
    out = {}
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    pa = masks.get('pulmonary_artery')
    ao = masks.get('aorta')
    if pa is not None and ao is not None:
        pa_v = int((pa > 0).sum()); ao_v = int((ao > 0).sum())
        if pa_v > 0 and ao_v > 0:
            d_pa = (6 * pa_v / np.pi) ** (1 / 3)
            d_ao = (6 * ao_v / np.pi) ** (1 / 3)
            out['PA_Ao_Diameter_Ratio'] = float(d_pa / d_ao)
            out['PA_Ao_Volume_Ratio'] = float(pa_v / ao_v)
    rv = masks.get('heart_ventricle_right'); lv = masks.get('heart_ventricle_left')
    if rv is not None and lv is not None:
        rv_v = int((rv > 0).sum()); lv_v = int((lv > 0).sum())
        out['RV_LV_Volume_Ratio'] = float(rv_v / lv_v) if lv_v > 0 else np.nan
        out['RV_Volume_mm3'] = float(rv_v * voxel_vol)
        out['LV_Volume_mm3'] = float(lv_v * voxel_vol)
    heart = masks.get('heart')
    if heart is not None:
        calc = int(np.sum((ct_arr > 130) & (ct_arr < 3000) & (heart > 0)))
        out['CAC_Volume_mm3'] = float(calc * voxel_vol)
    return out


def airway_lobe_coupling(masks, spacing):
    out = {}
    airway = masks.get('lung_trachea_bronchia')
    if airway is None:
        return out
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    aw = airway > 0
    aw_total = int(aw.sum())
    out['Airway_Total_Volume_mm3'] = float(aw_total * voxel_vol)
    for mask_name, key in LOBE_MAP.items():
        lobe = masks.get(mask_name)
        if lobe is None:
            continue
        inter = int(np.sum(aw & (lobe > 0)))
        out[f'Airway_Lobe_{key}_Volume_pct'] = float(inter / aw_total * 100) if aw_total > 0 else np.nan
    return out


def diaphragm_flattening(ct_arr, masks):
    out = {}
    for mask_name, key in [('lung_lower_lobe_left', 'Left'),
                           ('lung_lower_lobe_right', 'Right')]:
        lobe = masks.get(mask_name)
        if lobe is None:
            continue
        arr = lobe > 0
        zs = np.where(arr.any(axis=(1, 2)))[0]
        if len(zs) == 0:
            continue
        z_bottom = zs[-1]
        slice_2d = arr[z_bottom]
        if slice_2d.sum() == 0:
            continue
        ys, xs = np.where(slice_2d)
        bb_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        fill_ratio = slice_2d.sum() / bb_area
        out[f'Diaphragm_{key}_Fill_Ratio_bottom'] = float(fill_ratio)
    return out


# =========================================================================
# 肺血管高级特征（替代慢速 shape）：分形维度 / BV5-BV10 / 中心线图论
# =========================================================================
def vessel_fractal_dimension(mask):
    """3D 计盒分形维度（box-counting），量化血管网空间复杂度/密集度。
    COPD 远端毛细血管床修剪(pruning) -> 维度下降；咯血畸形增生 -> 维度上升。"""
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) == 0:
        return np.nan
    p = int(min(mask.shape))
    if p < 4:
        return np.nan
    n = int(2 ** np.floor(np.log2(p)))
    sizes = 2 ** np.arange(int(np.log2(n)), 1, -1)   # 盒子尺寸：2 的幂次递减

    def boxcount(Z, k):
        k = int(k)
        Zi = Z.astype(np.int32)                       # int32 省内存
        S = np.add.reduceat(
            np.add.reduceat(
                np.add.reduceat(Zi, np.arange(0, Z.shape[0], k), axis=0),
                np.arange(0, Z.shape[1], k), axis=1),
            np.arange(0, Z.shape[2], k), axis=2)
        return int(np.count_nonzero(S))              # 含血管的盒子数

    counts = np.array([boxcount(mask, s) for s in sizes], dtype=float)
    valid = counts > 0
    if int(valid.sum()) < 3:
        return np.nan
    coeffs = np.polyfit(np.log(sizes[valid]), np.log(counts[valid]), 1)
    return float(-coeffs[0])


def vessel_bv5_bv10(mask, spacing):
    """BV5/BV10：小血管血容量占比。截面积 <5mm² 与 <10mm² 的血管体积占全血管体积 %。
    用 3D 距离变换估计局部半径 r = edt * px；r < sqrt(5/pi)=1.26mm 对应 BV5。"""
    out = {}
    mask = np.asarray(mask, dtype=bool)
    total = int(mask.sum())
    if total == 0:
        out['Vessel_BV5_pct'] = np.nan
        out['Vessel_BV10_pct'] = np.nan
        return out
    try:
        import edt as _edt
        edt = _edt.edt(mask.astype(np.uint8), parallel=1).astype(np.float32)
    except Exception:
        edt = ndi.distance_transform_edt(mask).astype(np.float32)  # 体素单位
    px = float(min(spacing[0], spacing[1]))            # 轴向平面分辨率（血管截面在平面内）
    r_mm = edt * px
    thr5 = float(np.sqrt(5.0 / np.pi))                 # ≈1.2616 mm
    thr10 = float(np.sqrt(10.0 / np.pi))               # ≈1.7841 mm
    r_vessel = r_mm[mask]
    out['Vessel_BV5_pct'] = float(np.count_nonzero(r_vessel < thr5) / total * 100.0)
    out['Vessel_BV10_pct'] = float(np.count_nonzero(r_vessel < thr10) / total * 100.0)
    return out


def vessel_graph_features(mask, spacing):
    """中心线（骨架）图论特征：迂曲度 + 分支点密度。
    skimage.skeletonize 提骨架(3D自动分派) -> 26 邻域计数分叉点/端点 -> 去分叉点分段求迂曲度。"""
    out = {}
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) == 0:
        return out
    skel = skeletonize(mask)
    n_skel = int(skel.sum())
    out['Vessel_Skeleton_Voxels'] = n_skel
    if n_skel == 0:
        return out

    # 26 邻域计数（中心点自身不计；用 26 偏移逐点求和，避免 scipy 卷积 3x 缓冲 OOM）
    pad = np.pad(skel, 1, mode='constant')              # 布尔，+1 边框 False
    nbr = np.zeros(skel.shape, dtype=np.uint8)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                nbr += pad[1+dz:1+dz+skel.shape[0],
                           1+dy:1+dy+skel.shape[1],
                           1+dx:1+dx+skel.shape[2]]
    nbr = nbr * skel.astype(np.uint8)                    # 只保留骨架体素
    junctions = nbr >= 3
    n_junc = int(junctions.sum())
    out['Vessel_Junction_Count'] = n_junc
    out['Vessel_Endpoint_Count'] = int((nbr == 1).sum())

    # 去掉分叉点，分割成分支段（26 连通）
    seg_mask = skel & (~junctions)
    lbl, nseg = ndi.label(seg_mask, structure=np.ones((3, 3, 3), dtype=bool))
    out['Vessel_Branch_Count'] = int(nseg)

    spacing = np.asarray(spacing, dtype=float)
    voxel_len = float(np.mean(spacing))                # 骨架单步长度(mm)近似
    skel_len_mm = n_skel * voxel_len
    out['Vessel_Skeleton_Length_mm'] = float(skel_len_mm)
    out['Vessel_Branching_Density_per_mm'] = (float(n_junc / skel_len_mm)
                                              if skel_len_mm > 0 else np.nan)

    # 高效分组：只取骨架(段)体素小数组，按标签排序后 split（避免 O(段数 x 容积)）
    seg_coords = np.argwhere(seg_mask)                  # (N,3) zyx，仅段体素
    seg_lbls = lbl[seg_mask]
    order = np.argsort(seg_lbls, kind='stable')
    sorted_coords = seg_coords[order]
    sorted_lbls = seg_lbls[order]
    if len(sorted_lbls) and sorted_lbls[0] == 0:        # 剔除可能的背景标签
        cut = np.searchsorted(sorted_lbls, 1)
        sorted_coords = sorted_coords[cut:]
        sorted_lbls = sorted_lbls[cut:]
    split = np.flatnonzero(np.diff(sorted_lbls)) + 1
    groups = np.split(sorted_coords, split) if len(split) else [sorted_coords]

    torts = []
    for coords in groups:
        if len(coords) < 3:
            continue
        c = coords.astype(float)
        cc = c - c.mean(axis=0)
        _, _, v = np.linalg.svd(cc, full_matrices=False)
        proj = cc @ v[0]                                # PCA 第一主成分投影
        p0 = coords[int(np.argmin(proj))]
        p1 = coords[int(np.argmax(proj))]
        euclid = float(np.linalg.norm((p1 - p0) * spacing))
        arc = float(len(coords) * voxel_len)
        if euclid > 0:
            torts.append(arc / euclid)

    out['Vessel_Tortuosity_Mean'] = float(np.mean(torts)) if torts else np.nan
    out['Vessel_Tortuosity_Max'] = float(np.max(torts)) if torts else np.nan
    return out


def vessel_advanced_features(vessels_mask, spacing):
    """肺血管高级特征总入口：分形维度 + BV5/BV10 + 中心线图论。
    三组各自独立 try/except：某组失败（如骨架 OOM）不影响其余特征。"""
    out = {}
    try:
        out['Vessel_Fractal_Dim'] = vessel_fractal_dimension(vessels_mask)
    except Exception:
        out['Vessel_Fractal_Dim'] = None
    try:
        out.update(vessel_bv5_bv10(vessels_mask, spacing))
    except Exception:
        out['Vessel_BV5_pct'] = out['Vessel_BV10_pct'] = None
    try:
        out.update(vessel_graph_features(vessels_mask, spacing))
    except Exception:
        for c in ('Vessel_Skeleton_Voxels', 'Vessel_Skeleton_Length_mm',
                  'Vessel_Branch_Count', 'Vessel_Junction_Count',
                  'Vessel_Endpoint_Count', 'Vessel_Branching_Density_per_mm',
                  'Vessel_Tortuosity_Mean', 'Vessel_Tortuosity_Max'):
            out[c] = None
    return out


# =========================================================================
# 主流程：逐患者（掩膜并行，带诊断）
# =========================================================================
# 黄金 16 靶区：只对这些掩膜计算特征，忽略其余无关器官
# （2026-04-seg-part1 里有部分患者是暂停的 TotalSegmentator 结果，
#   含 117+ 个全器官掩膜如 vertebrae/rib/kidney/liver；只算这 16 个靶区
#   既保证特征一致，也避免把无关器官特征混入结果）
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


def process_patient(meta, nifti_dir, seg_dir, workers=8, force=False):
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

    ct_img = sitk.ReadImage(ct_path)
    ct_arr = sitk.GetArrayFromImage(ct_img)
    spacing = ct_img.GetSpacing()

    # 只保留黄金 16 靶区（过滤 vertebrae/rib/kidney 等无关器官）
    all_mask_files = sorted(f for f in os.listdir(mask_dir) if f.endswith('.nii.gz'))
    mask_files = [f for f in all_mask_files if f in KEEP_FILES]
    dropped = len(all_mask_files) - len(mask_files)
    if dropped > 0:
        print(f"  掩膜: 共 {len(all_mask_files)} 个，保留黄金靶区 {len(mask_files)} 个（忽略 {dropped} 个无关器官）")
    else:
        print(f"  掩膜: {len(mask_files)} 个，并行 workers={workers}")

    # 1) 并行跑 pyRadiomics（带 worker 诊断日志）
    jobs = [(ct_path, os.path.join(mask_dir, f), f[:-len('.nii.gz')]) for f in mask_files]
    t0 = time.time()
    with Pool(processes=workers) as pool:
        results = pool.map(_worker_pyradiomics, jobs)
    print(f"  pyRadiomics 并行完成，总耗时 {time.time()-t0:.1f}s")

    feats = {"Patient_ID": patient, "CT_Series": os.path.basename(ct_path)}
    feats["PatientID"] = extract_patient_id(meta, nifti_dir)
    for (ctp, mpath, name), roi_feats in zip(jobs, results):
        for k, v in roi_feats.items():
            feats[f"{name}::{k}"] = v

    # 2) 四类新指标
    masks = {f[:-len('.nii.gz')]: sitk.ReadImage(os.path.join(mask_dir, f)) for f in mask_files}
    mask_arrays = {k: sitk.GetArrayFromImage(v) for k, v in masks.items()}
    print("    -> 分肺叶气肿 / 心肺血管 / 气道耦合 / 膈肌 ...")
    feats.update(lobe_emphysema_features(ct_arr, spacing, mask_arrays))
    feats.update(cardiopulmonary_features(ct_arr, spacing, mask_arrays))
    feats.update(airway_lobe_coupling(mask_arrays, spacing))
    feats.update(diaphragm_flattening(ct_arr, mask_arrays))
    # 肺血管高级特征：分形维度 / BV5-BV10 / 中心线迂曲度与分支密度
    vessel = mask_arrays.get('lung_vessels')
    if vessel is not None:
        feats.update(vessel_advanced_features(vessel, spacing))

    # 3) 保存 json
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(feats), f, indent=2, ensure_ascii=False)
    print(f"  [OK] {patient}: 特征维度 {len(feats)-2}，已保存 {os.path.basename(out_json)}")
    return feats


def main():
    args = parse_args()
    nifti_dir = os.path.abspath(args.nifti_dir)
    seg_dir = os.path.abspath(args.seg_dir)

    patients = find_patients(seg_dir)
    if args.patients:
        wanted = set(args.patients.split(","))
        patients = [p for p in patients if p["patient"] in wanted]

    print(f"发现 {len(patients)} 个患者，开始提取（精简版/去纹理/诊断；单患者超时 {args.timeout}s）...")
    results = []
    failed = []
    for i, meta in enumerate(patients, 1):
        patient = meta["patient"]
        out_json = os.path.join(seg_dir, f"{patient}_radiomics.json")
        if os.path.exists(out_json) and not args.force:
            print(f"\n[{i}/{len(patients)}] {patient}  [skip] 已存在 radiomics json")
            continue

        print(f"\n[{i}/{len(patients)}] {patient}")
        # 关键修复：每个患者跑在独立子进程里 + 墙钟超时。
        # 之前直接在主进程跑，pool.map 遇挂死的 worker 永不返回，整个批处理卡死
        #（本次 26/150 就卡在 0:17 之后 2 小时无进展）。现在超时自动终止该患者并继续。
        proc = mp.Process(target=process_patient,
                          args=(meta, nifti_dir, seg_dir, args.workers, args.force),
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
        # 子进程正常结束：读取其写出的 json
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
    if results:
        df = pd.DataFrame(results)
        summary = os.path.join(seg_dir, "radiomics_all_patients.json")
        df.to_json(summary, orient="records", force_ascii=False)
        print(f"汇总 JSON: {summary}")


if __name__ == "__main__":
    main()
