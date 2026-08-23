import os
import json
import pydicom
import numpy as np
import SimpleITK as sitk
from collections import defaultdict
import argparse

# 空气的下限 (Hounsfield Unit)，用于裁剪 FOV 外区域，避免 TotalSegmentator 出错
CT_MIN_HU = -1024.0

def extract_dicom_info(dicom_dir):
    """从文件夹中提取 DICOM 信息（兼容无扩展名的文件）"""
    import glob
    info = {
        "Patient": {},
        "Study": {},
        "Series": {},
        "Instances": 0,
        "Error": None
    }
    
    if not os.path.isdir(dicom_dir):
        info['Error'] = f"目录不存在: {dicom_dir}"
        return info

    # 获取文件夹内所有文件（包括无扩展名）
    all_files = [f for f in os.listdir(dicom_dir) 
                 if os.path.isfile(os.path.join(dicom_dir, f))]
    
    if not all_files:
        info['Error'] = "文件夹为空"
        return info

    dcm_files = []
    for f in all_files:
        fpath = os.path.join(dicom_dir, f)
        try:
            # 尝试读取，看是否是 DICOM 文件
            dcm = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            dcm_files.append(fpath)
        except Exception:
            continue
    
    info['Instances'] = len(dcm_files)
    if not dcm_files:
        info['Error'] = f"文件夹内 {len(all_files)} 个文件均非有效 DICOM"
        return info

    # 读取第一个 DICOM 文件
    first_dcm = os.path.basename(dcm_files[0])
    try:
        ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
    except Exception as e:
        info['Error'] = f"读取第一个 DICOM 文件失败: {str(e)}"
        return info

    # 提取标签 (使用 try/except 避免字段缺失报错)
    def safe_tag(ds, tag_name, default=""):
        try:
            value = getattr(ds, tag_name)
            if value is None:
                return default
            return str(value)
        except Exception:
            return default

    info['Patient'] = {
        'PatientID': safe_tag(ds, 'PatientID'),
        'PatientName': safe_tag(ds, 'PatientName'),
        'PatientBirthDate': safe_tag(ds, 'PatientBirthDate'),
        'PatientSex': safe_tag(ds, 'PatientSex'),
    }
    info['Study'] = {
        'StudyInstanceUID': safe_tag(ds, 'StudyInstanceUID'),
        'StudyDate': safe_tag(ds, 'StudyDate'),
        'StudyTime': safe_tag(ds, 'StudyTime'),
        'StudyDescription': safe_tag(ds, 'StudyDescription'),
        'AccessionNumber': safe_tag(ds, 'AccessionNumber'),
    }
    info['Series'] = {
        'SeriesInstanceUID': safe_tag(ds, 'SeriesInstanceUID'),
        'SeriesNumber': safe_tag(ds, 'SeriesNumber'),
        'SeriesDescription': safe_tag(ds, 'SeriesDescription'),
        'Modality': safe_tag(ds, 'Modality'),
        'SliceThickness': safe_tag(ds, 'SliceThickness'),
        'SpacingBetweenSlices': safe_tag(ds, 'SpacingBetweenSlices'),
        'ConvolutionKernel': safe_tag(ds, 'ConvolutionKernel'),
        'Manufacturer': safe_tag(ds, 'Manufacturer'),
        'ManufacturerModelName': safe_tag(ds, 'ManufacturerModelName'),
        'KVP': safe_tag(ds, 'KVP'),
    }
    info['Rows'] = safe_tag(ds, 'Rows')
    info['Columns'] = safe_tag(ds, 'Columns')
    info['BitsStored'] = safe_tag(ds, 'BitsStored')

    return info


def clip_ct_to_hu_range(image, min_hu=CT_MIN_HU):
    """
    将图像中小于 min_hu 的体素裁剪为 min_hu（Hounsfield Unit）。
    FOV 外区域通常为极低值（< -1024 HU），强制设为空气值可避免 TotalSegmentator 出错。
    返回裁剪后的新图像；若无需裁剪则返回原图。
    """
    arr = sitk.GetArrayFromImage(image)
    n_low = int((arr < min_hu).sum())
    if n_low == 0:
        return image
    arr[arr < min_hu] = min_hu  # 等价于 MATLAB: img(img < -1024) = -1024
    clipped = sitk.GetImageFromArray(arr)
    # 保留原图的 spacing / origin / direction 等元数据
    clipped.CopyInformation(image)
    print(f"     已裁剪 {n_low} 个体素: HU < {min_hu} -> {min_hu}")
    return clipped


def convert_series_to_nifti(dicom_dir, output_nifti_path):
    """
    使用 SimpleITK 将单个序列的 DICOM 文件夹转换为 3D NIfTI 文件。
    转换后会裁剪 CT 值（< -1024 HU 强制设为 -1024），避免后续分割出错。
    """
    reader = sitk.ImageSeriesReader()
    
    # 自动获取序列中所有切片并排序
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    
    if not dicom_names:
        print(f"  ⚠️ 警告：{dicom_dir} 中没有找到有效的 DICOM 序列")
        return False
    
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # 🌟 CT 值裁剪：FOV 外区域 (HU < -1024) 强制设为空气值 -1024
    image = clip_ct_to_hu_range(image)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_nifti_path), exist_ok=True)
    
    # 保存为 .nii.gz
    sitk.WriteImage(image, output_nifti_path)
    
    # 打印序列信息
    print(f"  ✅ {os.path.basename(output_nifti_path)}")
    print(f"     切片数: {len(dicom_names)}, 图像尺寸: {image.GetSize()}")
    print(f"     像素间距: {image.GetSpacing()}")
    
    return True


def process_patient_folder(patient_folder, output_base_dir):
    """
    处理单个患者的 DICOM 文件夹。
    识别所有序列，转换为 NIfTI，并保存 DICOM 头信息。
    """
    patient_name = os.path.basename(patient_folder)
    print(f"\n{'='*60}")
    print(f"处理患者: {patient_name}")
    print(f"{'='*60}")
    
    # 创建输出目录
    patient_output_dir = os.path.join(output_base_dir, patient_name)
    os.makedirs(patient_output_dir, exist_ok=True)
    
    # 存储所有序列的信息
    all_series_info = {
        "PatientFolder": patient_name,
        "Series": []
    }
    
    # 遍历所有序列文件夹
    subdirs = [d for d in os.listdir(patient_folder) 
               if os.path.isdir(os.path.join(patient_folder, d))]
    
    for series_name in subdirs:
        series_path = os.path.join(patient_folder, series_name)
        print(f"\n📁 序列: {series_name}")
        
        # 提取 DICOM 信息
        series_info = extract_dicom_info(series_path)
        series_info["SeriesFolder"] = series_name
        all_series_info["Series"].append(series_info)
        
        # 转换为 NIfTI
        # 清理文件名，替换空格和特殊字符
        safe_series_name = series_name.replace(' ', '_').replace('.', '_')
        nifti_filename = f"{patient_name}_{safe_series_name}.nii.gz"
        nifti_path = os.path.join(patient_output_dir, nifti_filename)
        
        convert_series_to_nifti(series_path, nifti_path)
    
    # 保存 DICOM 信息为 JSON
    json_path = os.path.join(patient_output_dir, f"{patient_name}_dicom_info.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_series_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 DICOM 信息已保存至: {json_path}")
    
    return patient_output_dir


def batch_process_dicom_folders(input_base_dir, output_base_dir):
    """
    批量处理所有患者的 DICOM 文件夹。
    """
    print("🚀 启动批量 DICOM → NIfTI 转换流水线...")
    print(f"输入目录: {input_base_dir}")
    print(f"输出目录: {output_base_dir}\n")
    
    # 查找所有患者文件夹
    patient_folders = [d for d in os.listdir(input_base_dir) 
                       if os.path.isdir(os.path.join(input_base_dir, d))]
    
    if not patient_folders:
        print("❌ 未找到任何患者文件夹，请检查输入路径！")
        return
    
    print(f"📦 共找到 {len(patient_folders)} 个患者文件夹\n")
    
    success_count = 0
    for idx, patient_folder_name in enumerate(patient_folders, 1):
        patient_path = os.path.join(input_base_dir, patient_folder_name)
        
        try:
            print(f"\n{'#'*60}")
            print(f"[{idx}/{len(patient_folders)}] 正在处理...")
            process_patient_folder(patient_path, output_base_dir)
            success_count += 1
        except Exception as e:
            print(f"❌ 处理 {patient_folder_name} 时出错: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"🎉 批量转换完成！")
    print(f"📊 成功: {success_count}/{len(patient_folders)} 个患者")
    print(f"📁 输出目录: {output_base_dir}")
    print(f"{'='*60}")


# ================= 运行区 =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="批量将 DICOM 患者文件夹转换为 NIfTI 并提取扫描信息"
    )
    parser.add_argument(
        "-i", "--input",
        help="包含患者文件夹的根目录（例如 H:\\DICOM\\202607\\2026-07-heart）"
    )
    parser.add_argument(
        "-o", "--output",
        help="输出 NIfTI 文件的目标根目录（例如 H:\\DICOM\\202607\\2026-07-heart-nifti）"
    )
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    # 如果命令行未提供，则交互式询问
    if input_dir is None:
        input_dir = input("请输入患者 DICOM 文件夹所在的根目录: ").strip().strip('"')
    if output_dir is None:
        output_dir = input("请输入 NIfTI 输出目录: ").strip().strip('"')

    if not os.path.isdir(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
    else:
        batch_process_dicom_folders(input_dir, output_dir)