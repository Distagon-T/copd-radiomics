# -*- coding: utf-8 -*-
"""
逐患者 Radiomics 特征提取 + COPD 高级表型指标
================================================
基于 TotalSegmentator 分割结果 + 原始增强 CT，为每个患者计算：

A. PyRadiomics 全特征（shape + firstorder + glcm/glszm/gldm/ngtdm/rlng + wavelet/LoG）
   - 对每个分割掩膜（肺叶×5、肺血管、气管、心脏+四腔、心肌、主动脉、肺动脉）
   - 心肌用 wavelet 高频纹理；肺叶用 LoG 斑点；其余用 shape+firstorder

B. 肺叶级肺气肿定量（LAA-950 / Perc15 / 肺容积 / 过度膨胀指数）
C. 心肺共病大血管（PA/A 比值、RV/LV 容积比）
D. 气道-肺叶耦合（若存在 airway 掩膜，则按肺叶统计气道占比）
E. 膈肌平坦度（肺底平面曲率/平坦度估算）

输出：<患者>_radiomics.json 保存到分割结果文件夹（<患者>_masks/ 同级）
用法：
  python compute_patient_radiomics.py --nifti-dir <nifti_dir> \\
                                      --seg-dir <seg_dir>
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
import scipy.ndimage as ndi
from scipy.stats import linregress
from radiomics import featureextractor
import logging

# Windows 控制台 GBK 编码兼容：统一用 UTF-8 输出，避免 emoji/特殊字符崩溃
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.getLogger('radiomics').setLevel(logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser(description="逐患者 Radiomics + COPD 表型特征")
    p.add_argument("--nifti-dir", "-n", required=True, help="原始 CT 患者目录")
    p.add_argument("--seg-dir", "-s", required=True, help="分割结果目录（含 <患者>_masks/）")
    p.add_argument("--patients", default=None, help="只处理指定患者（逗号分隔），默认全部")
    p.add_argument("--force", action="store_true", help="已存在 json 也重算")
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
    """优先从 segmentation_info.json 的 selected_nifti 定位 CT；否则搜索患者目录最大层数。"""
    if os.path.exists(patient_meta["info_json"]):
        try:
            with open(patient_meta["info_json"], encoding="utf-8") as f:
                info = json.load(f)
            # selected_nifti 可能是相对/绝对路径
            sel = info.get("selected_nifti", "")
            for cand in [sel, os.path.join(patient_meta["input_dir"] if "input_dir" in info else "", sel),
                         os.path.join(nifti_dir, patient_meta["patient"], sel)]:
                if cand and os.path.isfile(cand):
                    return cand
            # 兜底：input_dir 里的 nii.gz
            ind = info.get("input_dir", "")
            if ind:
                files = glob.glob(os.path.join(ind, "*.nii.gz"))
                if files:
                    return sorted(files)[0]
        except Exception:
            pass
    # 搜索患者目录所有 nii.gz 取最大层数
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


# =========================================================================
# PyRadiomics 引擎配置（与 extract_featureX 一致的设置）
# =========================================================================
# binWidth=25          : 灰度离散化区间宽度，影响 GLCM/GLRLM 等纹理矩阵的尺寸
# force2D=False        : 用 3D 体素计算（2D 需单独开启 shape2D）
# voxelArrayShift=1000 : 把负 HU 值平移为正值（保证灰度非负，纹理特征才有意义）
# interpolator         : 重采样用的插值器（B 样条）
BASE_SETTINGS = {'binWidth': 25, 'force2D': False, 'voxelArrayShift': 1000,
                 'interpolator': sitk.sitkBSpline}


def make_extractors():
    """
    构建三个专用 pyRadiomics 引擎（不同 ROI 用不同图像变换）：
      - ext_shape : 仅 shape + firstorder（物理形态 + 一阶灰度统计），用于心脏/血管等
      - ext_myo   : 全特征 + Wavelet 高频子带，用于心肌（纹理细节最丰富）
      - ext_lung  : 全特征 + LoG 斑点滤波（sigma=1.0/3.0），用于肺叶（强调不同尺度病灶）
    返回 (ext_shape, ext_myo, ext_lung)
    """
    ext_shape = featureextractor.RadiomicsFeatureExtractor(**BASE_SETTINGS)
    ext_shape.disableAllFeatures()
    # 注意: pyradiomics 中「空列表/None」= 启用该类的全部特征（传 '*' 会报 Feature not found）
    ext_shape.enableFeaturesByName(shape=[], firstorder=[])

    ext_myo = featureextractor.RadiomicsFeatureExtractor(**BASE_SETTINGS)
    ext_myo.enableAllFeatures()
    ext_myo.enableImageTypeByName('Wavelet')

    ext_lung = featureextractor.RadiomicsFeatureExtractor(**BASE_SETTINGS)
    ext_lung.enableAllFeatures()
    ext_lung.enableImageTypeByName('LoG', customArgs={'sigma': [1.0, 3.0]})
    return ext_shape, ext_myo, ext_lung


def pick_extractor(roi_name, ext_shape, ext_myo, ext_lung):
    if 'myocardium' in roi_name:
        return ext_myo, "wavelet高频纹理"
    if 'lobe' in roi_name:
        return ext_lung, "LoG空间斑点"
    return ext_shape, "shape+firstorder"


def _to_jsonable(o):
    """递归把 numpy 类型转成 Python 原生类型，便于 json.dump。"""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)   # NaN -> null
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    return o


def run_pyradiomics(ct_img, mask_img, ext):
    """对单个 mask 跑 pyradiomics，返回 {特征名: 值}（过滤 diagnostics）。"""
    out = {}
    try:
        vec = ext.execute(ct_img, mask_img)
        for k, v in vec.items():
            if not k.startswith('diagnostics_'):
                out[k] = _to_jsonable(v)
    except Exception as e:
        print(f"      [warn] pyradiomics 失败: {e}")
    return out


# =========================================================================
# B. 肺叶级肺气肿定量与表型
# =========================================================================
# TotalSegmentator 的肺叶命名 -> 标准肺叶缩写
#   lung_upper_lobe_left   -> LLU (左上肺叶)
#   lung_lower_lobe_left   -> LLL (左下肺叶)
#   lung_upper_lobe_right  -> RUL (右上肺叶)
#   lung_middle_lobe_right -> RML (右中肺叶)
#   lung_lower_lobe_right  -> RLL (右下肺叶)
LOBE_MAP = {
    'lung_upper_lobe_left': 'LLU', 'lung_lower_lobe_left': 'LLL',
    'lung_upper_lobe_right': 'RUL', 'lung_middle_lobe_right': 'RML',
    'lung_lower_lobe_right': 'RLL',
}


def lobe_emphysema_features(ct_arr, spacing, masks):
    """
    对 5 个肺叶分别计算：
      - LAA-950% ：低衰减区占比（< -950 HU 体素百分比），是公认的肺气肿量化指标
      - Perc15    ：第 15 百分位密度值（越低 = 肺气肿越重），对噪声更稳健
      - 肺叶体积  ：体素数 × 体素体积 (mm^3)
      - 占全肺比例：该肺叶体积 / 全肺总容积
    并汇总全肺的 LAA-950% / Perc15。
    输入：
      ct_arr   : numpy 数组 [z,y,x]，原始 CT 灰度（HU）
      spacing  : (x,y,z) 体素间距（mm）
      masks    : {掩膜名: SimpleITK 图像}
    输出：{特征名: 值}，键带 Lobe_<缩写>_ 前缀
    """
    voxel_vol = spacing[0] * spacing[1] * spacing[2]   # 单个体素体积 mm^3
    out = {}
    full_lung = np.zeros_like(ct_arr, dtype=bool)      # 全肺并集（用于全肺统计）
    total_lung_vol = 0.0
    for mask_name, arr in masks.items():
        key = LOBE_MAP.get(mask_name)
        if key is None:
            continue
        lobe = arr > 0
        if lobe.sum() == 0:
            # 该肺叶为空掩膜（例如左肺无中叶），记为 NaN 或 0
            out[f'Lobe_{key}_LAA950_pct'] = np.nan
            out[f'Lobe_{key}_Perc15_HU'] = np.nan
            out[f'Lobe_{key}_Volume_mm3'] = 0.0
            out[f'Lobe_{key}_Vol_pct_of_lung'] = 0.0
            continue
        hu = ct_arr[lobe]                              # 提取该肺叶内所有体素 HU
        vol = lobe.sum() * voxel_vol
        out[f'Lobe_{key}_LAA950_pct'] = float(np.sum(hu < -950) / len(hu) * 100)
        out[f'Lobe_{key}_Perc15_HU'] = float(np.percentile(hu, 15))
        out[f'Lobe_{key}_Volume_mm3'] = float(vol)
        full_lung |= lobe
        total_lung_vol += vol
    # 全肺汇总
    out['Lung_Total_Volume_mm3'] = float(total_lung_vol)
    if full_lung.sum() > 0:
        hu_all = ct_arr[full_lung]
        out['Lung_LAA950_pct'] = float(np.sum(hu_all < -950) / len(hu_all) * 100)
        out['Lung_Perc15_HU'] = float(np.percentile(hu_all, 15))
    # 各肺叶占全肺比例
    for mask_name in LOBE_MAP:
        if f'Lobe_{LOBE_MAP[mask_name]}_Volume_mm3' in out and total_lung_vol > 0:
            out[f'Lobe_{LOBE_MAP[mask_name]}_Vol_pct_of_lung'] = \
                out[f'Lobe_{LOBE_MAP[mask_name]}_Volume_mm3'] / total_lung_vol
    return out


# =========================================================================
# C. 心肺共病与大血管特征
# =========================================================================
def cardiopulmonary_features(ct_arr, spacing, masks):
    """
    心肺共病相关指标：
      - PA/Ao 直径比：肺动脉主干 vs 升主动脉的等效直径比。
        正常 < 1；> 1 提示肺动脉高压（Pulmonary Hypertension）。
        由于没有 2D 截面切片，这里用体积等效球直径近似：
            V = (4/3)πr^3 -> d = (6V/π)^(1/3)
      - RV/LV 容积比：右室/左室体积比，COPD 伴肺动脉高压时 RV 增大。
      - CAC 钙化体积：heart 掩膜内 CT>130 HU（且 <3000 防金属伪影）的体素体积 mm^3。
    输入/输出约定同 lobe_emphysema_features。
    """
    out = {}
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    pa = masks.get('pulmonary_artery')
    ao = masks.get('aorta')
    if pa is not None and ao is not None:
        pa_v = int((pa > 0).sum()); ao_v = int((ao > 0).sum())
        if pa_v > 0 and ao_v > 0:
            # 等效直径：V = (4/3)πr^3 近似 d = (6V/π)^(1/3)；比值用直径比
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
        # CAC：heart 掩膜内 >130 HU 且 <3000 HU（排除金属）
        calc = int(np.sum((ct_arr > 130) & (ct_arr < 3000) & (heart > 0)))
        out['CAC_Volume_mm3'] = float(calc * voxel_vol)
    return out


# =========================================================================
# D. 气道-肺叶耦合（若存在 airway 掩膜）
# =========================================================================
def airway_lobe_coupling(masks, spacing):
    """
    气道-肺叶耦合分析：统计气管/支气管掩膜 (lung_trachea_bronchia) 落在
    每个肺叶内的体积占比。意义：判断远端气道在哪个肺叶分布最密集，
    辅助评估「气道病变-肺气肿」的空间关联（例如上叶型 COPD 的耦合模式）。
    输入：masks 为 {名称: 图像}；spacing 为 (x,y,z)。
    输出：Airway_Total_Volume_mm3 + 每个肺叶的 Airway_Lobe_<缩写>_Volume_pct
    """
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
        # 气道与该肺叶的重叠体素数
        inter = int(np.sum(aw & (lobe > 0)))
        out[f'Airway_Lobe_{key}_Volume_pct'] = float(inter / aw_total * 100) if aw_total > 0 else np.nan
    return out


# =========================================================================
# E. 膈肌形态学评估（肺底平面平坦度）
# =========================================================================
def diaphragm_flattening(ct_arr, masks):
    """
    膈肌平坦度估算（无需专门膈肌掩膜的近似方法）：
      对左右下肺叶，取包含该肺叶的最低 z 切片（即肺底/膈肌穹窿所在层面），
      计算该层面肺叶轮廓的「外接矩形填充比」：
          fill_ratio = 轮廓体素数 / 外接矩形面积
      意义：正常膈肌呈穹窿状 -> 底切片轮廓窄而填充比低；
            肺过度充气使膈肌变平 -> 底切片轮廓宽而填充比高。
    输出：Diaphragm_Left/Right_Fill_Ratio_bottom
    """
    out = {}
    for mask_name, key in [('lung_lower_lobe_left', 'Left'),
                           ('lung_lower_lobe_right', 'Right')]:
        lobe = masks.get(mask_name)
        if lobe is None:
            continue
        arr = lobe > 0
        # 找到该肺叶覆盖的所有 z 切片
        zs = np.where(arr.any(axis=(1, 2)))[0]
        if len(zs) == 0:
            continue
        z_bottom = zs[-1]              # 最低切片（膈肌层面）
        slice_2d = arr[z_bottom]
        if slice_2d.sum() == 0:
            continue
        # 轮廓外接框
        ys, xs = np.where(slice_2d)
        bb_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        fill_ratio = slice_2d.sum() / bb_area
        # 平坦度指标：填充比越高 → 膈肌越平（正常穹窿→填充比低）
        out[f'Diaphragm_{key}_Fill_Ratio_bottom'] = float(fill_ratio)
    return out


# =========================================================================
# 主流程：逐患者
# =========================================================================
from declared_features_lib import declared_features


def process_patient(meta, nifti_dir, seg_dir, extractors, force=False):
    patient = meta["patient"]
    mask_dir = meta["mask_dir"]
    out_json = os.path.join(seg_dir, f"{patient}_radiomics.json")

    if os.path.exists(out_json) and not force:
        print(f"  [skip] {patient} 已存在 radiomics json，跳过（--force 重算）")
        return None

    # 定位 CT
    ct_path = resolve_ct_path(meta, nifti_dir)
    if ct_path is None:
        print(f"  [FAIL] {patient}: 找不到 CT")
        return None
    print(f"  CT: {os.path.basename(ct_path)}")

    # 读图像 + 掩膜
    ct_img = sitk.ReadImage(ct_path)
    ct_arr = sitk.GetArrayFromImage(ct_img)
    spacing = ct_img.GetSpacing()  # (x,y,z)

    masks = {}
    for f in sorted(os.listdir(mask_dir)):
        if f.endswith('.nii.gz'):
            name = f[:-len('.nii.gz')]
            try:
                masks[name] = sitk.ReadImage(os.path.join(mask_dir, f))
            except Exception as e:
                print(f"      [warn] 读掩膜失败 {f}: {e}")
    mask_arrays = {k: sitk.GetArrayFromImage(v) for k, v in masks.items()}

    print(f"  掩膜: {len(mask_arrays)} 个")

    # 1) 全特征 radiomics（对每个 mask）
    ext_shape, ext_myo, ext_lung = extractors
    feats = {"Patient_ID": patient, "CT_Series": os.path.basename(ct_path)}
    feats["PatientID"] = extract_patient_id(meta, nifti_dir)
    for roi_name, mask_img in masks.items():
        ext, desc = pick_extractor(roi_name, ext_shape, ext_myo, ext_lung)
        print(f"    -> {roi_name} ({desc})")
        roi_feats = run_pyradiomics(ct_img, mask_img, ext)
        for k, v in roi_feats.items():
            feats[f"{roi_name}::{k}"] = v

    # 2) 四类新指标
    print("    -> 分肺叶气肿 / 心肺血管 / 气道耦合 / 膈肌 ...")
    feats.update(lobe_emphysema_features(ct_arr, spacing, mask_arrays))
    feats.update(cardiopulmonary_features(ct_arr, spacing, mask_arrays))
    feats.update(airway_lobe_coupling(mask_arrays, spacing))
    feats.update(diaphragm_flattening(ct_arr, mask_arrays))

    # 2.5) 申报清单补算特征（CAC Agatston/MS · 心包脂肪 · FAI · 主动脉 · 心胸比 · 血管体积/CSA）
    #      与 radiomics 一并输出，实现单脚本双输出（见 declared_features_lib.py / README §5.1）
    feats.update(declared_features(ct_img, masks))

    # 3) 保存 json（先清洗 numpy 类型，避免 ndarray / np.float64 无法序列化）
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

    print(f"发现 {len(patients)} 个患者，开始提取 radiomics + COPD 表型特征...")
    extractors = make_extractors()
    results = []
    for i, meta in enumerate(patients, 1):
        print(f"\n[{i}/{len(patients)}] {meta['patient']}")
        r = process_patient(meta, nifti_dir, seg_dir, extractors, args.force)
        if r:
            results.append(r)

    print(f"\n完成！{len(results)}/{len(patients)} 个患者生成了 radiomics json。")
    # 汇总到 seg 目录
    if results:
        df = pd.DataFrame(results)
        summary = os.path.join(seg_dir, "radiomics_all_patients.json")
        df.to_json(summary, orient="records", force_ascii=False)
        print(f"汇总 JSON: {summary}")


if __name__ == "__main__":
    main()
