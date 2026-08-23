import os
import glob
import time
import shutil
from totalsegmentator.python_api import totalsegmentator

def batch_process_ct(input_dir, output_base_dir):
    print("🚀 启动 [COPD & 心血管共病] 终极满血版批量分割流水线...")
    
    # 自动搜索所有 patient_xx_ct.nii.gz 文件
    search_pattern = os.path.join(input_dir, "patient_*_ct.nii.gz")
    ct_files = sorted(glob.glob(search_pattern))
    
    total_files = len(ct_files)
    if total_files == 0:
        print(f"❌ 在 {input_dir} 目录下未找到目标 CT 文件，请检查路径！")
        return

    print(f"📦 共检测到 {total_files} 个 CT 病例，准备满血发车。\n")
    os.makedirs(output_base_dir, exist_ok=True)
    
    overall_start_time = time.time()
    success_count = 0
    fail_count = 0

    # 🌟 终极黄金 17 靶区 (117+ 个结果中只取这 17 个精华)
    keep_files = [
        # --- 1. 肺部宏观 (5个) ---
        "lung_upper_lobe_left.nii.gz", "lung_lower_lobe_left.nii.gz", 
        "lung_upper_lobe_right.nii.gz", "lung_middle_lobe_right.nii.gz", "lung_lower_lobe_right.nii.gz", 
        # --- 2. 肺部微观结构 (2个) ---
        "lung_vessels.nii.gz", "lung_trachea_bronchia.nii.gz",
        # --- 3. 大血管与主气管干 (4个) (包含回归的静脉和动脉！) ---
        "aorta.nii.gz", "pulmonary_artery.nii.gz", "pulmonary_vein.nii.gz", "trachea.nii.gz",
        # --- 4. 整体心脏 (1个) (V2 引擎原生输出，专供 CAC 钙化分析) ---
        "heart.nii.gz",
        # --- 5. 局部高精心血管组件 (5个) (专供 RV/LV 比例与心肌组学) ---
        "heart_myocardium.nii.gz", "heart_atrium_left.nii.gz", "heart_ventricle_left.nii.gz",
        "heart_atrium_right.nii.gz", "heart_ventricle_right.nii.gz"
    ]

    for idx, ct_path in enumerate(ct_files, 1):
        filename = os.path.basename(ct_path)
        patient_id = filename.split("_ct.nii.gz")[0]
        patient_output_dir = os.path.join(output_base_dir, f"{patient_id}_masks")
        
        print(f"\n[{idx}/{total_files}] ================= 正在处理: {patient_id} =================")
        
        # 💡 极其严苛的断点续传检查：17 个文件必须一个不落，缺一不可
        already_done = True
        if not os.path.exists(patient_output_dir):
            already_done = False
        else:
            for kf in keep_files:
                if not os.path.exists(os.path.join(patient_output_dir, kf)):
                    already_done = False
                    break 
        
        if already_done:
            print(f"   ⏭️ 检测到 {patient_id} 的 17 个核心靶区已完美齐全，安全跳过。")
            success_count += 1
            continue
            
        try:
            patient_start = time.time()
            
            # 🧹 战前清场：如果之前有跑到一半烂尾的文件夹，直接推平重建
            if os.path.exists(patient_output_dir):
                shutil.rmtree(patient_output_dir)
            os.makedirs(patient_output_dir, exist_ok=True)
            
            # 🚀 引擎 1：大众模型 (提取 5大肺叶、主气管、主动脉、肺静脉、整体心脏)
            print("   -> 正在全速运行 [引擎 1/3]：基础大器官与整体心脏提取...")
            totalsegmentator(ct_path, patient_output_dir)
            
            # 🚀 引擎 2：微观模型 (提取 肺血管网、支气管树)
            print("   -> 正在追加运行 [引擎 2/3]：肺微观树状网络提取 (Task 258)...")
            totalsegmentator(ct_path, patient_output_dir, task="lung_vessels")
            
            # 🚀 引擎 3：授权模型 (提取 心肌、四腔室、肺动脉)
            print("   -> 正在追加运行 [引擎 3/3]：高精心血管与肺动脉提取 (需 License)...")
            totalsegmentator(ct_path, patient_output_dir, task="heartchambers_highres")
            
            patient_time = time.time() - patient_start
            print(f"   ✅ {patient_id} 三擎提取全量成功！耗时: {patient_time/60:.2f} 分钟")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ {patient_id} 分割引擎中途崩溃: {e}")
            print("   ⚠️ (常见原因：1. 断网导致模型下载失败；2. 引擎3缺乏 License 授权秘钥)")
            fail_count += 1
            
        finally:
            # 🛡️ 强制保洁：只留下科研需要的 17 个黄金文件，其余上百个垃圾文件瞬间灰飞烟灭
            if os.path.exists(patient_output_dir):
                print("   🧹 正在执行强制清理程序...")
                cleaned = 0
                for file in os.listdir(patient_output_dir):
                    if file.endswith(".nii.gz") and file not in keep_files:
                        os.remove(os.path.join(patient_output_dir, file))
                        cleaned += 1
                if cleaned > 0:
                    print(f"      清理完毕：已焚毁 {cleaned} 个无用器官掩膜，保留 17 个核心靶区。")

    overall_time = (time.time() - overall_start_time) / 3600
    print("\n" + "="*50)
    print(f"🎉 批量任务彻底结束！总耗时: {overall_time:.2f} 小时")
    print(f"📊 战报统计: 完美生成 {success_count} 例，异常 {fail_count} 例")
    print("="*50)

if __name__ == "__main__":
    # 请确保路径指向你服务器上的 99 个病人原图位置
    INPUT_FOLDER = r"D:\copd-radiomics\ct_source" 
    OUTPUT_FOLDER = r"D:\copd-radiomics\seg_results"
    
    batch_process_ct(INPUT_FOLDER, OUTPUT_FOLDER)