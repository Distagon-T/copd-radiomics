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
import scipy.ndimage as ndi
from skimage.morphology import skeletonize
from declared_features_lib import declared_features  # noqa: E402  申报清单补算特征（与 Windows 单脚本双输出一致）


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

logging.getLogger('radiomics').setLevel(logging.WARNING)

# 允许 SimpleITK/ITK 滤波用满全部核心
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(0)

# 肺叶掩膜名 -> 缩写
LOBE_MAP = {
    'lung_upper_lobe_left': 'LLU', 'lung_lower_lobe_left': 'LLL',
    'lung_upper_lobe_right': 'RUL', 'lung_middle_lobe_right': 'RML',
    'lung_lower_lobe_right': 'RLL',
}

# 黄金 16 靶区：只对这些掩膜计算特征，忽略其余无关器官
# （TotalSegmentator 默认输出 ~117 个全器官掩膜，很多在胸部 CT 里是空的或无关）
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

    # 读掩膜（只保留黄金 16 靶区，忽略 TotalSegmentator 全器官输出里的无关/空掩膜）
    masks = {}
    all_mask_files = sorted(f for f in os.listdir(mask_dir) if f.endswith('.nii.gz'))
    mask_files = [f for f in all_mask_files if f in KEEP_FILES]
    dropped = len(all_mask_files) - len(mask_files)
    for f in mask_files:
        name = f[:-len('.nii.gz')]
        try:
            masks[name] = sitk.ReadImage(os.path.join(mask_dir, f))
        except Exception as e:
            print(f"      [warn] 读掩膜失败 {f}: {e}")
    mask_arrays = {k: sitk.GetArrayFromImage(v) for k, v in masks.items()}
    if dropped > 0:
        print(f"  掩膜: 共 {len(all_mask_files)} 个，保留黄金靶区 {len(mask_files)} 个（忽略 {dropped} 个无关器官）")
    else:
        print(f"  掩膜: {len(mask_files)} 个")

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
    # 肺血管高级特征：分形维度 / BV5-BV10 / 中心线迂曲度与分支密度
    vessel = mask_arrays.get('lung_vessels')
    if vessel is not None:
        feats.update(vessel_advanced_features(vessel, spacing))
    # 申报清单补算特征（CAC Agatston/MS · 心包脂肪 · FAI · 主动脉 · 心胸比 · 血管体积/CSA）
    # 与 Windows compute_patient_radiomics*.py 一致：radiomics 与补算特征单次输出（见 declared_features_lib.py）
    feats.update(declared_features(ct_img, masks))
    return feats
