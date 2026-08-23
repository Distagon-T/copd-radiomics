import os
import SimpleITK as sitk

def convert_dicom_to_nifti(dicom_dir, output_nifti_path):
    """
    将包含多个 .dcm 文件的文件夹转换为单个 3D .nii.gz 文件
    """
    print(f"🔍 正在扫描 DICOM 文件夹: {dicom_dir}")
    
    # 1. 初始化一个序列读取器
    reader = sitk.ImageSeriesReader()
    
    # 2. 自动获取该文件夹下同一序列的所有 DICOM 文件名（自动按空间位置排序）
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    
    if not dicom_names:
        print("❌ 错误：在该文件夹下没有找到有效的 DICOM 序列，请检查路径！")
        return
        
    print(f"📦 共找到 {len(dicom_names)} 张 DICOM 切片，正在拼接为 3D 矩阵...")
    reader.SetFileNames(dicom_names)
    
    # 3. 执行读取并拼接
    image = reader.Execute()
    
    # 4. 确保输出的文件夹存在
    os.makedirs(os.path.dirname(output_nifti_path), exist_ok=True)
    
    # 5. 保存为 .nii.gz 格式
    sitk.WriteImage(image, output_nifti_path)
    
    print(f"🎉 转换大功告成！\n📁 3D NIfTI 文件已保存至: {output_nifti_path}")

# ================= 运行区 =================
if __name__ == "__main__":
    # 【请在这里修改为你的真实路径】
    # 比如你的 dcm 文件都在 D 盘的 data/patient1 文件夹下
    INPUT_DICOM_FOLDER = "testDCM" 
    
    # 你希望保存的 .nii.gz 文件名和路径
    OUTPUT_NIFTI_FILE = "patient_99_ct.nii.gz"
    
    convert_dicom_to_nifti(INPUT_DICOM_FOLDER, OUTPUT_NIFTI_FILE)