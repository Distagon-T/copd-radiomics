import os
import glob
import time
import shutil
from totalsegmentator.python_api import totalsegmentator

def batch_process_ct(input_dir, output_base_dir):
    print("🚀 启动 GPU 满血版批量分割流水线 (工业级防坑自愈版)...")
    
    search_pattern = os.path.join(input_dir, "patient_*_ct.nii.gz")
    ct_files = sorted(glob.glob(search_pattern))
    
    total_files = len(ct_files)
    if total_files == 0:
        print(f"❌ 在 {input_dir} 目录下未找到 CT 文件，请检查路径！")
        return

    print(f"📦 共检测到 {total_files} 个 CT 文件，准备发车。\n")
    os.makedirs(output_base_dir, exist_ok=True)
    
    overall_start_time = time.time()
    success_count = 0
    fail_count = 0

    # 🌟 黄金 16 靶区
    keep_files = [
        "lung_upper_lobe_left.nii.gz", "lung_lower_lobe_left.nii.gz", 
        "lung_upper_lobe_right.nii.gz", "lung_middle_lobe_right.nii.gz", "lung_lower_lobe_right.nii.gz", 
        "lung_vessels.nii.gz", "lung_trachea_bronchia.nii.gz",
        "aorta.nii.gz", "pulmonary_artery.nii.gz", "trachea.nii.gz",
        "heart.nii.gz",
        "heart_myocardium.nii.gz", "heart_atrium_left.nii.gz", "heart_ventricle_left.nii.gz",
        "heart_atrium_right.nii.gz", "heart_ventricle_right.nii.gz"
    ]

    for idx, ct_path in enumerate(ct_files, 1):
        filename = os.path.basename(ct_path)
        patient_id = filename.split("_ct.nii.gz")[0]
        patient_output_dir = os.path.join(output_base_dir, f"{patient_id}_masks")
        
        print(f"\n[{idx}/{total_files}] 正在处理: {patient_id} ...")
        
        # 💡 修复 1：极其严苛的断点续传检查
        already_done = True
        if not os.path.exists(patient_output_dir):
            already_done = False
        else:
            for kf in keep_files:
                if not os.path.exists(os.path.join(patient_output_dir, kf)):
                    already_done = False
                    break # 只要缺一个，就判定没做完
        
        if already_done:
            print(f"   ⏭️ 检测到 {patient_id} 的 16 个核心靶区已完美齐全，安全跳过。")
            success_count += 1
            continue
            
        try:
            patient_start = time.time()
            
            # 如果之前有烂尾的数据，直接推平重建，防止脏数据干扰
            if os.path.exists(patient_output_dir):
                shutil.rmtree(patient_output_dir)
            os.makedirs(patient_output_dir, exist_ok=True)
            
            print("   -> 引擎 1/3：提取全量大器官...")
            totalsegmentator(ct_path, patient_output_dir)
            
            print("   -> 引擎 2/3：提取肺微观树状网络...")
            totalsegmentator(ct_path, patient_output_dir, task="lung_vessels")
            
            print("   -> 引擎 3/3：提取高精心血管与肺动脉...")
            totalsegmentator(ct_path, patient_output_dir, task="heartchambers_highres")
            
            patient_time = time.time() - patient_start
            print(f"   ✅ {patient_id} 核心器官提取成功！耗时: {patient_time/60:.2f} 分钟")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ {patient_id} 分割引擎中途报错: {e}")
            print("   ⚠️ (提示: 如果报网络错误，可能是微观/心脏子模型未下载完成)")
            fail_count += 1
            
        finally:
            # 💡 修复 2：绝对会执行的“强制保洁”
            # 无论上面是成功还是报错，只要跑了，最后统统把无关脏器删干净
            if os.path.exists(patient_output_dir):
                print("   🧹 正在执行强制清理程序...")
                cleaned = 0
                for file in os.listdir(patient_output_dir):
                    if file.endswith(".nii.gz") and file not in keep_files:
                        os.remove(os.path.join(patient_output_dir, file))
                        cleaned += 1
                if cleaned > 0:
                    print(f"      清理完毕：删除了 {cleaned} 个无用器官文件。")

    overall_time = (time.time() - overall_start_time) / 3600
    print("\n" + "="*50)
    print(f"🎉 批量任务彻底结束！总耗时: {overall_time:.2f} 小时")
    print(f"📊 统计: 完美完成 {success_count} 例，异常 {fail_count} 例")
    print("="*50)

if __name__ == "__main__":
    INPUT_FOLDER = "ct_source" 
    OUTPUT_FOLDER = "seg_results"
    batch_process_ct(INPUT_FOLDER, OUTPUT_FOLDER)