#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyRadiomics 特征提取核心模块（Docker 版）
=========================================
对单个患者的 16 个掩膜 + 原始 CT 计算：
  A. PyRadiomics shape + firstorder（每个掩膜）
  B. 四类 COPD 表型指标：
     1) 肺叶级肺气肿   (LAA-950% / Perc15 / 体积)
     2) 气道-肺叶耦合  (气道在各肺叶占比)
     3) 心肺共病       (PA/Ao 比值 / RV/LV 容积比 / CAC)
     4) 膈肌形态       (肺底轮廓填充比)

注意: 本模块与本地 compute_patient_radiomics_lite.py 同源，
     但针对 Docker 部署简化（去掉多进程，单进程串行，保证内存可控）。
"""
import os
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
import logging

logging.getLogger('radiomics').setLevel(logging.WARNING)

# 允许 SimpleITK/ITK 滤波用满全部核心
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(0)

# 肺叶掩膜名 -> 缩写
LOBE_MAP = {
    'lung_upper_lobe_left': 'LLU', 'lung_lower_lobe_left': 'LLL',
    'lung_upper_lobe_right': 'RUL', 'lung_middle_lobe_right': 'RML',
    'lung_lower_lobe_right': 'RLL',
}

# preCrop: 裁剪到掩膜包围盒再算，大幅加速（心腔/主动脉等小 ROI 提速 5~10x）
BASE_SETTINGS = {'binWidth': 25, 'force2D': False, 'voxelArrayShift': 1000,
                 'interpolator': sitk.sitkBSpline, 'preCrop': True}


def _to_jsonable(o):
    """递归把 numpy 类型转成 Python 原生类型，便于 json.dump。"""
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


def _build_extractor(roi_name):
    """
    构建 pyRadiomics 引擎。
    - lung_vessels（肺血管，体积巨大）: 只算 firstorder，跳过最慢的 shape 特征
      （Maximum3DDiameter / SurfaceArea / Compactness 等在几百万体素上耗时 ~35 分钟，
       且肺血管的 shape 特征对 COPD 研究价值有限）
    - 其余掩膜: shape + firstorder
    """
    ext = featureextractor.RadiomicsFeatureExtractor(**BASE_SETTINGS)
    ext.disableAllFeatures()
    if 'lung_vessels' in roi_name:
        ext.enableFeaturesByName(firstorder=[])
    else:
        ext.enableFeaturesByName(shape=[], firstorder=[])
    return ext


def _run_pyradiomics(ct_img, mask_img, ext):
    """对单个掩膜跑 pyradiomics，返回清洗后的 {特征名: 值}。"""
    out = {}
    try:
        vec = ext.execute(ct_img, mask_img)
        for k, v in vec.items():
            if not k.startswith('diagnostics_'):
                out[k] = _to_jsonable(v)
    except Exception as e:
        print(f"      [warn] pyradiomics 失败: {e}")
    return out


# ---------------------------------------------------------------------------
# B. 肺叶级肺气肿
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# C. 心肺共病
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# D. 气道-肺叶耦合
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# E. 膈肌形态
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 主入口：单患者
# ---------------------------------------------------------------------------
def extract_patient_radiomics(ct_path, mask_dir, patient_name):
    """
    对一个患者做完整特征提取。
    ct_path : 原始 CT .nii.gz 路径
    mask_dir: <患者>_masks 目录（含 16 个 .nii.gz 掩膜）
    patient_name: 患者名
    返回 {特征名: 值} 字典。
    """
    # 读 CT
    ct_img = sitk.ReadImage(ct_path)
    ct_arr = sitk.GetArrayFromImage(ct_img)
    spacing = ct_img.GetSpacing()

    # 读掩膜
    masks = {}
    mask_files = sorted(f for f in os.listdir(mask_dir) if f.endswith('.nii.gz'))
    for f in mask_files:
        name = f[:-len('.nii.gz')]
        try:
            masks[name] = sitk.ReadImage(os.path.join(mask_dir, f))
        except Exception as e:
            print(f"      [warn] 读掩膜失败 {f}: {e}")
    mask_arrays = {k: sitk.GetArrayFromImage(v) for k, v in masks.items()}
    print(f"  掩膜: {len(mask_arrays)} 个")

    # 1) pyradiomics（串行，内存可控；lung_vessels 只算 firstorder 提速）
    feats = {"Patient_ID": patient_name, "CT_Series": os.path.basename(ct_path)}
    for roi_name, mask_img in masks.items():
        ext = _build_extractor(roi_name)
        roi_feats = _run_pyradiomics(ct_img, mask_img, ext)
        for k, v in roi_feats.items():
            feats[f"{roi_name}::{k}"] = v

    # 2) 四类新指标
    print("    -> 分肺叶气肿 / 心肺血管 / 气道耦合 / 膈肌 ...")
    feats.update(lobe_emphysema_features(ct_arr, spacing, mask_arrays))
    feats.update(cardiopulmonary_features(ct_arr, spacing, mask_arrays))
    feats.update(airway_lobe_coupling(mask_arrays, spacing))
    feats.update(diaphragm_flattening(ct_arr, mask_arrays))
    return feats
