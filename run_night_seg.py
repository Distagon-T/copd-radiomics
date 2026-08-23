import os
import time
from totalsegmentator.python_api import totalsegmentator

def run_full_cpu_seg(input_ct_path, output_dir):
    print("🌙 夜间模式启动：满血版 CPU 高精度分割...")
    print("⚠️ 警告：正在处理 0.6mm 超薄层 CT，预计将消耗大量内存并耗时数小时。")
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    
    try:
        # 核心改动：
        # 1. 保留了 device="cpu" 以绕过 MX450 的 2GB 显存限制
        # 2. 彻底去掉了 fast=True，让模型以最高精度提取心脏、血管和肺部边缘
        totalsegmentator(input_ct_path, output_dir, device="cpu")
        
        end_time = time.time()
        print(f"\n🎉 满血分割大功告成！总耗时: {(end_time - start_time) / 3600:.2f} 小时")
        print(f"📁 完美的高精度 Mask 已保存至: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ 分割在半夜崩溃了。错误信息: {e}")
        print("💡 如果报错包含 'MemoryError' 或 'Killed'，说明这台电脑的物理内存撑不住 0.6mm 的高精度重建，只能去台式机上跑了。")

# ================= 运行区 =================
if __name__ == "__main__":
    # 【替换为你刚才生成的 nii.gz 绝对路径】
    TEST_CT = r"D:\copd-radiomics\patient_01_ct.nii.gz"
    
    # 建议换一个新的输出文件夹，免得覆盖了之前 fast 模式的结果，方便明天做对比
    OUTPUT_MASKS = r"D:\copd-radiomics\patient01_masks_full_res"
    
    run_full_cpu_seg(TEST_CT, OUTPUT_MASKS)