import os
import SimpleITK as sitk
import numpy as np

def merge_masks_for_visualization(mask_dir, output_file):
    print("🎨 正在生成用于 ITK-SNAP 的多标签彩色合并掩膜...")
    
    # ⚠️ 注意排序：我们把整体 heart 放在前面，精细组件放在后面
    # 这样精细组件的颜色会覆盖在整体心脏之上，展现出完美的嵌套效果！
    ordered_files = [
        "heart.nii.gz", # 标签 1
        "lung_upper_lobe_left.nii.gz", "lung_lower_lobe_left.nii.gz", # 标签 2, 3
        "lung_upper_lobe_right.nii.gz", "lung_middle_lobe_right.nii.gz", "lung_lower_lobe_right.nii.gz", # 标签 4, 5, 6
        "aorta.nii.gz",  "trachea.nii.gz", # 标签 7, 8, 9
        "heart.nii.gz"
    ]

    merged_arr = None
    ref_img = None

    for label_value, filename in enumerate(ordered_files, start=1):
        filepath = os.path.join(mask_dir, filename)
        if os.path.exists(filepath):
            img = sitk.ReadImage(filepath)
            arr = sitk.GetArrayFromImage(img)
            
            # 初始化空白的超级画布
            if merged_arr is None:
                merged_arr = np.zeros_like(arr, dtype=np.uint8)
                ref_img = img
                
            # 将当前器官的区域赋值为对应的独特标签数字 (1-14)
            merged_arr[arr > 0] = label_value
            print(f"   ✅ 已合并 {filename} -> 标签色 {label_value}")
        else:
            print(f"   ⚠️ 缺失 {filename}，已跳过。")

    if ref_img is not None:
        merged_img = sitk.GetImageFromArray(merged_arr)
        merged_img.CopyInformation(ref_img)
        sitk.WriteImage(merged_img, output_file)
        print(f"\n🎉 合并大功告成！请将此文件拖入 ITK-SNAP: {output_file}")

# ================= 运行区 =================
if __name__ == "__main__":
    # 替换为你某一个病人的掩膜文件夹路径
    MASK_FOLDER = "seg_results/patient_01_masks" 
    OUTPUT_MERGED = "seg_results/patient_01_merged_color.nii.gz"
    
    merge_masks_for_visualization(MASK_FOLDER, OUTPUT_MERGED)