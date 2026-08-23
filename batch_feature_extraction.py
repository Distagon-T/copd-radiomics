import os
import glob
import time
import SimpleITK as sitk
import numpy as np
import pandas as pd
from radiomics import featureextractor

def calculate_clinical_metrics(ct_img, mask_dict):
    """计算核心临床影像学指标 (带安全校验机制)"""
    metrics = {}
    
    # 1. 肺气肿指标: LAA%-950 & Perc15
    lung_masks = [mask_dict.get(k) for k in mask_dict if "lung" in k and mask_dict.get(k) is not None]
    if len(lung_masks) > 0:
        # 将现有的肺叶全部合并
        full_lung_mask = sum(lung_masks) > 0
        lung_arr = sitk.GetArrayFromImage(ct_img)[sitk.GetArrayFromImage(full_lung_mask) > 0]
        
        if len(lung_arr) > 0:
            metrics['Clinical_LAA_950_percent'] = np.sum(lung_arr < -950) / len(lung_arr) * 100
            metrics['Clinical_Perc15_HU'] = np.percentile(lung_arr, 15)

    # 2. 冠脉钙化体积代理 (CAC Proxy)
    if 'heart' in mask_dict:
        heart_arr = sitk.GetArrayFromImage(mask_dict['heart'])
        ct_arr = sitk.GetArrayFromImage(ct_img)
        calcification_voxels = np.sum((ct_arr > 130) & (ct_arr < 3000) & (heart_arr > 0))
        spacing = ct_img.GetSpacing()
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        metrics['Clinical_CAC_Volume_mm3'] = calcification_voxels * voxel_volume

    # 3. 肺动脉/主动脉比值 (安全机制：如果你没有肺动脉mask，这里会自动填 NaN 而不报错)
    if 'pulmonary_artery' in mask_dict and 'aorta' in mask_dict:
        pa_voxels = np.sum(sitk.GetArrayFromImage(mask_dict['pulmonary_artery']) > 0)
        ao_voxels = np.sum(sitk.GetArrayFromImage(mask_dict['aorta']) > 0)
        metrics['Clinical_PA_Ao_Ratio'] = pa_voxels / ao_voxels if ao_voxels > 0 else np.nan
    else:
        metrics['Clinical_PA_Ao_Ratio'] = np.nan

    return metrics

def extract_radiomics(ct_path, mask_path, extractor):
    """调用 PyRadiomics 提取单靶区特征"""
    result = extractor.execute(ct_path, mask_path)
    return {k: v for k, v in result.items() if not k.startswith('diagnostics_')}

def run_batch_extraction(ct_dir, mask_base_dir, output_csv):
    print("🚀 启动全队列高维组学与临床特征批量提取流水线...")
    
    # 初始化 PyRadiomics 提取器 (只初始化一次，极大提升批量速度)
    settings = {'binWidth': 25, 'resampledPixelSpacing': None, 'interpolator': sitk.sitkBSpline}
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableAllFeatures()
    extractor.enableImageTypeByName('Wavelet')
    extractor.enableImageTypeByName('LoG', customArgs={'sigma': [1.0, 2.0, 3.0]})

    search_pattern = os.path.join(ct_dir, "patient_*_ct.nii.gz")
    ct_files = sorted(glob.glob(search_pattern))
    total_patients = len(ct_files)
    
    if total_patients == 0:
        print("❌ 未找到 CT 文件，请检查路径！")
        return

    # 准备一个超级大列表，存放所有病人的字典
    all_patients_data = []
    overall_start_time = time.time()

    for idx, ct_path in enumerate(ct_files, 1):
        patient_id = os.path.basename(ct_path).split("_ct.nii.gz")[0]
        patient_mask_dir = os.path.join(mask_base_dir, f"{patient_id}_masks")
        
        print(f"\n[{idx}/{total_patients}] 正在处理病人: {patient_id}")
        
        if not os.path.exists(patient_mask_dir):
            print(f"   ⚠️ 找不到对应的 Mask 文件夹，跳过该病人。")
            continue

        try:
            pt_start = time.time()
            # 初始化当前病人的数据字典
            pt_features = {'Patient_ID': patient_id}
            
            # 1. 加载 CT 和可用的 Mask
            ct_img = sitk.ReadImage(ct_path)
            mask_dict = {}
            available_rois = []
            
            for mask_file in os.listdir(patient_mask_dir):
                if mask_file.endswith('.nii.gz'):
                    roi_name = mask_file.replace('.nii.gz', '')
                    mask_dict[roi_name] = sitk.ReadImage(os.path.join(patient_mask_dir, mask_file))
                    available_rois.append(roi_name)
            
            # 2. 计算临床指标
            clinical_metrics = calculate_clinical_metrics(ct_img, mask_dict)
            pt_features.update(clinical_metrics)
            print(f"   🏥 临床指标提取完毕 (LAA%-950: {clinical_metrics.get('Clinical_LAA_950_percent', 0):.2f}%)")

            # 3. 提取高维组学特征
            print(f"   🧬 正在提取 {len(available_rois)} 个器官的高维组学特征...")
            for roi in available_rois:
                roi_mask_path = os.path.join(patient_mask_dir, f"{roi}.nii.gz")
                # 提取并添加前缀
                roi_radiomics = extract_radiomics(ct_path, roi_mask_path, extractor)
                for feature_name, value in roi_radiomics.items():
                    pt_features[f"{roi}_{feature_name}"] = value
            
            # 把当前病人的完整字典加入总列表
            all_patients_data.append(pt_features)
            
            pt_time = time.time() - pt_start
            print(f"   ✅ {patient_id} 处理成功！耗时: {pt_time/60:.2f} 分钟 (共获取 {len(pt_features)-1} 个特征)")

        except Exception as e:
            print(f"   ❌ {patient_id} 提取失败！错误: {e}")
            continue

    # ================= 汇总与保存 =================
    print("\n" + "="*50)
    print("💾 正在将所有数据拼装合并为超级大宽表...")
    
    # 转换为 DataFrame。Pandas 会极其智能地对齐列名，如果某个病人少了个肺叶，缺失的特征会自动填为 NaN
    df = pd.DataFrame(all_patients_data)
    
    # 强制将 Patient_ID 移动到第一列
    cols = ['Patient_ID'] + [c for c in df.columns if c != 'Patient_ID']
    df = df[cols]
    
    # 保存为 CSV
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    overall_time = (time.time() - overall_start_time) / 3600
    print(f"🎉 批量提取彻底结束！总耗时: {overall_time:.2f} 小时")
    print(f"📊 最终输出表维度: {df.shape[0]} 行 (病人) × {df.shape[1]} 列 (特征)")
    print(f"📁 文件已保存至: {output_csv}")
    print("="*50)

# ================= 运行区 =================
if __name__ == "__main__":
    # 【请确认以下三个路径与你的服务器环境一致】
    CT_FOLDER = "ct_source"               # 存放 99 个 patient_xx_ct.nii.gz 的目录
    MASKS_FOLDER = "seg_results"       # 存放 patient_xx_masks 文件夹的根目录
    OUTPUT_CSV = "COPD_Radiomics_Cohort.csv"    # 最终汇总的 CSV 文件路径
    
    run_batch_extraction(CT_FOLDER, MASKS_FOLDER, OUTPUT_CSV)