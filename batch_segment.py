import os
import glob
import time
from totalsegmentator.python_api import totalsegmentator

def batch_process_ct(input_dir, output_base_dir):
    print("🚀 启动 GPU 满血版批量分割流水线 (三擎驱动终极版)...")
    
    search_pattern = os.path.join(input_dir, "patient_*_ct.nii.gz")
    ct_files = sorted(glob.glob(search_pattern))
    
    total_files = len(ct_files)
    if total_files == 0:
        print(f"❌ 在 {input_dir} 目录下没有找到目标 CT 文件，请检查路径！")
        return

    print(f"📦 共检测到 {total_files} 个 CT 文件，准备开始处理。\n")
    os.makedirs(output_base_dir, exist_ok=True)
    
    overall_start_time = time.time()
    success_count = 0
    fail_count = 0

    # 🌟 黄金 16 靶区 (117 个结果中只取此一瓢饮)
    keep_files = [
        # --- 肺部宏观 (5个) ---
        "lung_upper_lobe_left.nii.gz", "lung_lower_lobe_left.nii.gz", 
        "lung_upper_lobe_right.nii.gz", "lung_middle_lobe_right.nii.gz", "lung_lower_lobe_right.nii.gz", 
        # --- 肺部微观结构：血管树与支气管树 (2个) ---
        "lung_vessels.nii.gz", "lung_trachea_bronchia.nii.gz",
        # --- 大血管与气管干 (3个) ---
        "aorta.nii.gz", "pulmonary_artery.nii.gz", "trachea.nii.gz",
        # --- 整体心脏 (原生输出，用于 CAC) (1个) ---
        "heart.nii.gz",
        # --- 局部精细心血管组件 (用于 RV/LV 与心肌组学) (5个) ---
        "heart_myocardium.nii.gz", "heart_atrium_left.nii.gz", "heart_ventricle_left.nii.gz",
        "heart_atrium_right.nii.gz", "heart_ventricle_right.nii.gz"
    ]

    for idx, ct_path in enumerate(ct_files, 1):
        filename = os.path.basename(ct_path)
        patient_id = filename.split("_ct.nii.gz")[0]
        patient_output_dir = os.path.join(output_base_dir, f"{patient_id}_masks")
        
        print(f"\n[{idx}/{total_files}] 正在处理: {patient_id} ...")
        
        # 【断点续传机制】
        if os.path.exists(patient_output_dir) and len(os.listdir(patient_output_dir)) >= 16:
            print(f"   ⏭️ 检测到 {patient_id} 的 16 个靶区已齐全，安全跳过。")
            success_count += 1
            continue
            
        try:
            patient_start = time.time()
            
            # 💡 引擎 1：提取全量大器官 (肺叶、主动脉、主气管、整体心脏)
            totalsegmentator(ct_path, patient_output_dir, device="gpu")
            
            # 💡 引擎 2：提取肺部微观结构 (肺血管网、支气管树)
            print("   -> 正在追加提取肺微观树状网络...")
            totalsegmentator(ct_path, patient_output_dir, task="lung_vessels")
            
            # 💡 引擎 3：提取高精心血管结构 (心肌、四腔室、消失的肺动脉！)
            print("   -> 正在追加提取高精心血管与肺动脉...")
            totalsegmentator(ct_path, patient_output_dir, task="heartchambers_highres")
            
            # 🧹 暴力清理：移除所有无关脏器，让目录极其干净
            print("   -> 正在清理无用器官文件...")
            for file in os.listdir(patient_output_dir):
                if file.endswith(".nii.gz") and file not in keep_files:
                    os.remove(os.path.join(patient_output_dir, file))
            
            patient_time = time.time() - patient_start
            print(f"   ✅ {patient_id} 全流程解剖提取成功！耗时: {patient_time/60:.2f} 分钟")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ {patient_id} 分割失败！错误原因: {e}")
            fail_count += 1
            continue

    overall_time = (time.time() - overall_start_time) / 3600
    print("\n" + "="*50)
    print(f"🎉 批量任务彻底结束！")
    print(f"⏱️ 总耗时: {overall_time:.2f} 小时")
    print(f"📊 统计: 成功 {success_count} 例，失败 {fail_count} 例")
    print("="*50)

if __name__ == "__main__":
    # 确认路径正确后发车
    INPUT_FOLDER = r"D:\copd-radiomics\ct_source" 
    OUTPUT_FOLDER = r"D:\copd-radiomics\seg_results"
    batch_process_ct(INPUT_FOLDER, OUTPUT_FOLDER)