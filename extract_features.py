import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from radiomics import featureextractor

def calculate_clinical_metrics(ct_img, mask_dict):
    """计算 4 个核心临床影像学指标"""
    metrics = {}
    
    # 1 & 2: 肺部指标 (合并双侧肺叶)
    lung_masks = [mask_dict.get(k) for k in mask_dict if "lung" in k]
    if len(lung_masks) > 0:
        # 将所有肺叶 mask 合并为一个全肺 mask
        full_lung_mask = sum(lung_masks) > 0
        lung_arr = sitk.GetArrayFromImage(ct_img)[sitk.GetArrayFromImage(full_lung_mask) > 0]
        
        # LAA%-950 (肺气肿指数)
        laa_950 = np.sum(lung_arr < -950) / len(lung_arr) * 100
        # Perc15 (第15百分位密度)
        perc15 = np.percentile(lung_arr, 15)
        
        metrics['LAA_950_percent'] = laa_950
        metrics['Perc15_HU'] = perc15

    # 3. CAC Proxy (冠脉钙化体积代理，单位 mm^3)
    if 'heart' in mask_dict:
        heart_arr = sitk.GetArrayFromImage(mask_dict['heart'])
        ct_arr = sitk.GetArrayFromImage(ct_img)
        # 提取心脏掩膜内，CT值 > 130 HU 且 < 3000 HU (排除金属伪影) 的体素
        calcification_voxels = np.sum((ct_arr > 130) & (ct_arr < 3000) & (heart_arr > 0))
        # 乘以单个体素的物理体积
        spacing = ct_img.GetSpacing()
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        metrics['CAC_Volume_mm3'] = calcification_voxels * voxel_volume

    # 4. PA/Ao Ratio (肺动脉与主动脉体积比)
    if 'pulmonary_artery' in mask_dict and 'aorta' in mask_dict:
        pa_voxels = np.sum(sitk.GetArrayFromImage(mask_dict['pulmonary_artery']) > 0)
        ao_voxels = np.sum(sitk.GetArrayFromImage(mask_dict['aorta']) > 0)
        metrics['PA_Ao_Volume_Ratio'] = pa_voxels / ao_voxels if ao_voxels > 0 else 0

    return metrics

def extract_radiomics(ct_path, mask_path):
    """提取 1500+ 高维影像组学特征"""
    # 初始化特征提取器
    settings = {'binWidth': 25, 'resampledPixelSpacing': None, 'interpolator': sitk.sitkBSpline}
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    
    # 开启所有基础特征类
    extractor.enableAllFeatures()
    # 🌟 核心：开启 Wavelet 和 LoG 滤波器，这会让特征数量从 100+ 暴增到 1500+
    extractor.enableImageTypeByName('Wavelet')
    extractor.enableImageTypeByName('LoG', customArgs={'sigma': [1.0, 2.0, 3.0]})

    print(f"正在提取组学特征: {os.path.basename(mask_path)} ...")
    result = extractor.execute(ct_path, mask_path)
    # 过滤掉系统诊断信息，只保留特征数值
    features = {k: v for k, v in result.items() if not k.startswith('diagnostics_')}
    return features



# ================= 运行区 =================
if __name__ == "__main__":
    ct_path = r"D:\copd-radiomics\patient_01_ct.nii.gz"
    mask_dir = r"D:\copd-radiomics\patient01_masks_full_res"
    
    # 读取原始 CT
    ct_img = sitk.ReadImage(ct_path)
    
    # 读取所有 Mask
    mask_dict = {}
    for mask_file in os.listdir(mask_dir):
        if mask_file.endswith('.nii.gz'):
            name = mask_file.replace('.nii.gz', '')
            mask_dict[name] = sitk.ReadImage(os.path.join(mask_dir, mask_file))
            
    # 1. 计算临床指标
    print("🏥 正在计算四大临床影像学指标...")
    clinical_metrics = calculate_clinical_metrics(ct_img, mask_dict)
    for k, v in clinical_metrics.items():
        print(f"   {k}: {v:.2f}")
        
    # 2. 提取某一个靶区（例如左上肺）的高维组学特征
    target_mask = os.path.join(mask_dir, "lung_upper_lobe_left.nii.gz")
    radiomics_features = extract_radiomics(ct_path, target_mask)
    print(f"\n🧬 成功提取高维组学特征数量: {len(radiomics_features)} 个")

 # === 在这里加上保存结果的代码 ===
    
    # 1. 把临床指标和高维组学特征合并到一个大字典里
    all_results = {**clinical_metrics, **radiomics_features}
    
    # 2. 加入病人的 ID，方便以后批量跑的时候做区分
    all_results['Patient_ID'] = 'patient01'
    
    # 3. 把字典转换成 pandas 的 DataFrame 一维表格
    df = pd.DataFrame([all_results])
    
    # 4. 把 Patient_ID 移动到表格的第一列（强迫症福音）
    cols = ['Patient_ID'] + [c for c in df if c != 'Patient_ID']
    df = df[cols]
    
    # 5. 保存为 CSV 文件 (utf-8-sig 编码防止在 Windows Excel 里打开乱码)
    output_csv = r"D:\copd-radiomics\patient01_features.csv"
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print(f"\n💾 完美！所有特征已永久保存至: {output_csv}")
   