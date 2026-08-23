import os
import glob
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
import scipy.ndimage as ndi
from scipy.stats import linregress
from radiomics import featureextractor
import logging

# 屏蔽刷屏日志
logging.getLogger('radiomics').setLevel(logging.WARNING)

def calculate_box_counting_fractal_dimension(binary_array):
    """计算 3D 计盒分形维度 (Box-counting Fractal Dimension)"""
    # 找到包含血管的最小 3D 边界框，减小计算量
    coords = np.argwhere(binary_array)
    if len(coords) == 0: return np.nan
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0) + 1
    cropped_arr = binary_array[z_min:z_max, y_min:y_max, x_min:x_max]

    # 定义不同尺寸的盒子
    sizes = [2, 4, 8, 16, 32]
    counts = []
    for size in sizes:
        # 使用最大池化模拟 3D 盒子覆盖
        shape = (cropped_arr.shape[0] // size + 1, size,
                 cropped_arr.shape[1] // size + 1, size,
                 cropped_arr.shape[2] // size + 1, size)
        padded_arr = np.zeros((shape[0]*size, shape[2]*size, shape[4]*size), dtype=bool)
        padded_arr[:cropped_arr.shape[0], :cropped_arr.shape[1], :cropped_arr.shape[2]] = cropped_arr
        
        # 统计包含非零元素的盒子数量
        reduced = padded_arr.reshape(shape).max(axis=(1, 3, 5))
        counts.append(np.sum(reduced))

    # 线性拟合 log(N) vs log(1/s)
    coeffs = linregress(np.log(1.0 / np.array(sizes)), np.log(counts))
    return coeffs.slope

def extract_clinical_and_topology(ct_img, mask_dict):
    """提取黄金临床指标、BV5 与 3D 拓扑分形特征"""
    metrics = {
        'Clinical_LAA_950_percent': np.nan, 'Clinical_Perc15_HU': np.nan,
        'Clinical_CAC_Volume_mm3': np.nan,
        'Clinical_PA_Ao_Ratio': np.nan, 'Clinical_RV_LV_Ratio': np.nan,
        'Topology_Vessel_BV5_Volume_mm3': np.nan, 'Topology_Vessel_BV5_LV_Ratio': np.nan,
        'Topology_Vessel_Fractal_Dimension': np.nan,
        'Topology_Trachea_Fractal_Dimension': np.nan
    }
    
    ct_arr = sitk.GetArrayFromImage(ct_img)
    spacing = ct_img.GetSpacing() # (x, y, z)
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    
    # 1. 肺部与气肿指标 (LAA & Perc15)
    lung_keys = [k for k in mask_dict if "lung_upper" in k or "lung_middle" in k or "lung_lower" in k]
    if len(lung_keys) > 0:
        full_lung_mask = np.zeros_like(ct_arr, dtype=bool)
        for k in lung_keys:
            full_lung_mask |= (sitk.GetArrayFromImage(mask_dict[k]) > 0)
        lung_pixels = ct_arr[full_lung_mask]
        total_lung_vol = np.sum(full_lung_mask) * voxel_vol
        
        if len(lung_pixels) > 0:
            metrics['Clinical_LAA_950_percent'] = np.sum(lung_pixels < -950) / len(lung_pixels) * 100
            metrics['Clinical_Perc15_HU'] = np.percentile(lung_pixels, 15)

    # 2. 冠脉钙化 (CAC)
    if 'heart' in mask_dict:
        heart_arr = sitk.GetArrayFromImage(mask_dict['heart'])
        calc_voxels = np.sum((ct_arr > 130) & (ct_arr < 3000) & (heart_arr > 0))
        metrics['Clinical_CAC_Volume_mm3'] = calc_voxels * voxel_vol

    # 3. 肺动脉高压心室比 (RV/LV)
    if 'heart_ventricle_right' in mask_dict and 'heart_ventricle_left' in mask_dict:
        rv_vol = np.sum(sitk.GetArrayFromImage(mask_dict['heart_ventricle_right']) > 0)
        lv_vol = np.sum(sitk.GetArrayFromImage(mask_dict['heart_ventricle_left']) > 0)
        metrics['Clinical_RV_LV_Ratio'] = rv_vol / lv_vol if lv_vol > 0 else 0

    # 4. 血管修剪金标准 (BV5 & 分形维度)
    if 'lung_vessels' in mask_dict:
        vessel_arr = sitk.GetArrayFromImage(mask_dict['lung_vessels']) > 0
        if np.sum(vessel_arr) > 0:
            # 物理真实距离的 3D 欧氏距离变换 (采样间距设为 CT spacing 的 z, y, x)
            edt = ndi.distance_transform_edt(vessel_arr, sampling=(spacing[2], spacing[1], spacing[0]))
            # 截面积 < 5mm2 对应的半径临界值 r = sqrt(5/pi) ≈ 1.2615 mm
            bv5_mask = (edt > 0) & (edt <= 1.2615)
            bv5_vol = np.sum(bv5_mask) * voxel_vol
            metrics['Topology_Vessel_BV5_Volume_mm3'] = bv5_vol
            if 'total_lung_vol' in locals() and total_lung_vol > 0:
                metrics['Topology_Vessel_BV5_LV_Ratio'] = bv5_vol / total_lung_vol
            
            # 血管分形维度
            metrics['Topology_Vessel_Fractal_Dimension'] = calculate_box_counting_fractal_dimension(vessel_arr)

    # 5. 气管分形维度
    if 'lung_trachea_bronchia' in mask_dict:
        trachea_arr = sitk.GetArrayFromImage(mask_dict['lung_trachea_bronchia']) > 0
        if np.sum(trachea_arr) > 0:
            metrics['Topology_Trachea_Fractal_Dimension'] = calculate_box_counting_fractal_dimension(trachea_arr)

    return metrics

def batch_extract_ultimate(ct_dir, mask_base_dir, output_csv):
    print("🌌 启动 [全景大一统] 特征提取引擎 (临床 + 拓扑 + 组学)...")
    
    ct_files = sorted(glob.glob(os.path.join(ct_dir, "patient_*_ct.nii.gz")))
    if not ct_files: return
    
    # 实例化三大组学引擎 (逻辑与之前一致)
    base_settings = {'binWidth': 25, 'force2D': False, 'voxelArrayShift': 1000, 'interpolator': sitk.sitkBSpline}
    extractor_shape = featureextractor.RadiomicsFeatureExtractor(**base_settings)
    extractor_shape.disableAllFeatures()
    extractor_shape.enableFeaturesByName(shape=True, firstorder=True)
    
    extractor_myo = featureextractor.RadiomicsFeatureExtractor(**base_settings)
    extractor_myo.enableAllFeatures()
    extractor_myo.enableImageTypeByName('Wavelet')
    
    extractor_lung = featureextractor.RadiomicsFeatureExtractor(**base_settings)
    extractor_lung.enableAllFeatures()
    extractor_lung.enableImageTypeByName('LoG', customArgs={'sigma': [1.0, 3.0]})

    all_data = []
    for idx, ct_path in enumerate(ct_files, 1):
        patient_id = os.path.basename(ct_path).replace("_ct.nii.gz", "")
        patient_mask_dir = os.path.join(mask_base_dir, f"{patient_id}_masks")
        print(f"\n[{idx}/{len(ct_files)}] 正在高通量扫描: {patient_id} ...")
        
        if not os.path.exists(patient_mask_dir): continue
        patient_dict = {'Patient_ID': patient_id}
        
        # 1. 载入图像与所有 Mask
        ct_img = sitk.ReadImage(ct_path)
        mask_files = [f for f in os.listdir(patient_mask_dir) if f.endswith('.nii.gz')]
        loaded_masks = {f.replace('.nii.gz', ''): sitk.ReadImage(os.path.join(patient_mask_dir, f)) for f in mask_files}
        
        # 2. 🔥 启动并融合：临床指标与 CV 拓扑学引擎 (BV5 & FD)
        print("   -> 正在计算: 临床生理金标准 + 血管/气管 3D 拓扑...")
        clin_top_metrics = extract_clinical_and_topology(ct_img, loaded_masks)
        patient_dict.update(clin_top_metrics)

        # 3. 🔥 启动并融合：智能组学引擎
        for roi_name, mask_img in loaded_masks.items():
            if 'myocardium' in roi_name:
                ext = extractor_myo; print(f"   -> 组学引擎 [高频纹理]: {roi_name}")
            elif 'lobe' in roi_name:
                ext = extractor_lung; print(f"   -> 组学引擎 [空间斑点]: {roi_name}")
            else:
                ext = extractor_shape; print(f"   -> 组学引擎 [物理形态]: {roi_name}")
                
            try:
                vec = ext.execute(ct_img, mask_img)
                for k, v in vec.items():
                    if not k.startswith('diagnostics_'): patient_dict[f"{roi_name}_{k}"] = v
            except: pass

        all_data.append(patient_dict)
        pd.DataFrame(all_data).to_csv(output_csv, index=False)
        print(f"   ✅ {patient_id} 融合提取完成！当前累计特征维度: {len(patient_dict)-1} 列")

if __name__ == "__main__":
    CT_FOLDER = "." 
    MASK_FOLDER = "seg_results"
    FINAL_CSV = "The_Ultimate_COPD_Features.csv"
    batch_extract_ultimate(CT_FOLDER, MASK_FOLDER, FINAL_CSV)