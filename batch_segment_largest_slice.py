import os
import glob
import json
import time
import argparse
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator

# 🌟 黄金 16 靶区 (117 个结果中只取此一瓢饮)
KEEP_FILES = [
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
    "heart_atrium_right.nii.gz", "heart_ventricle_right.nii.gz",
]

# 保留 16 个靶区 + 1 个 info json = 17 个文件即视为完成
TARGET_FILE_COUNT = len(KEEP_FILES) + 1


def find_largest_slice_nifti(patient_dir):
    """
    在患者文件夹中找到层数最多的 .nii.gz 文件。
    返回 (最佳文件路径, 层数)；找不到返回 (None, None)。
    """
    nii_files = sorted(glob.glob(os.path.join(patient_dir, "*.nii.gz")))
    if not nii_files:
        return None, None

    best_path = None
    best_slices = -1
    for f in nii_files:
        try:
            img = nib.load(f)
            shape = img.shape
            # 层数 = 第三维 (z 轴)；2D 图像取第一维
            slices = shape[2] if len(shape) >= 3 else shape[0]
            if slices > best_slices:
                best_slices = slices
                best_path = f
        except Exception as e:
            print(f"    ⚠️ 读取层数失败 {os.path.basename(f)}: {e}")
    return best_path, best_slices


def inspect_nifti(nifti_path):
    """读取 NIfTI 的体积信息，用于写入 JSON。"""
    img = nib.load(nifti_path)
    # 统一转成 Python 原生类型，避免 numpy float32 等无法被 json.dump 序列化
    zooms = [float(x) for x in img.header.get_zooms()]
    return {
        "shape": [int(x) for x in img.shape],
        "slices": int(img.shape[2]) if len(img.shape) >= 3 else int(img.shape[0]),
        "spacing": zooms,
        "pixdim": zooms,
    }


def write_segmentation_info(json_path, info):
    """将分割信息写入 JSON 文件。"""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def process_patient(patient_folder_path, patient_name, output_base_dir):
    """
    处理单个患者：找到层数最多的 NIfTI 并分割。
    返回 (success, info_dict)。
    """
    info = {
        "patient_folder": patient_name,
        "input_dir": patient_folder_path,
        "selected_nifti": None,
        "slice_count": None,
        "volume_info": None,
        "status": "pending",
        "elapsed_seconds": None,
        "error": None,
        "masks": [],
    }

    # ① 找到层数最多的 NIfTI
    nifti_path, n_slices = find_largest_slice_nifti(patient_folder_path)
    if nifti_path is None:
        info["status"] = "failed"
        info["error"] = "该文件夹下未找到任何 .nii.gz 文件"
        return False, info

    info["selected_nifti"] = os.path.basename(nifti_path)
    info["slice_count"] = n_slices
    info["volume_info"] = inspect_nifti(nifti_path)

    # ② 输出目录：与 batch_segment.py 保持一致 <患者名>_masks
    patient_output_dir = os.path.join(output_base_dir, f"{patient_name}_masks")
    info_json_path = os.path.join(patient_output_dir, f"{patient_name}_segmentation_info.json")

    # 【断点续传机制】
    # 只有 info json 标记为 success 且掩膜齐全才跳过；
    # 若 json 存在但 status=processing（中途暂停）或掩膜不足，则继续补跑。
    if os.path.exists(patient_output_dir):
        existing = [f for f in os.listdir(patient_output_dir) if f.endswith('.nii.gz')]
        if len(existing) >= len(KEEP_FILES) and os.path.exists(info_json_path):
            try:
                with open(info_json_path, encoding="utf-8") as _f:
                    _prev = json.load(_f)
                if _prev.get("status") == "success":
                    info["status"] = "skipped"
                    info["masks"] = sorted(existing)
                    print(f"   ⏭️ 检测到 {patient_name} 靶区已齐全 (已有 {len(existing)} 个掩膜 + success json)，安全跳过。")
                    return True, info
            except Exception:
                pass
            # json 存在但非 success -> 说明中途暂停，继续补跑（保留已有掩膜，
            # TotalSegmentator 会跳过已存在的目标）
            print(f"   🔄 {patient_name} 检测到未完成的分割记录 (已有 {len(existing)} 个掩膜)，继续补跑...")

    os.makedirs(patient_output_dir, exist_ok=True)
    patient_start = time.time()

    # 🔑 每完成一个引擎就写入一次 json（含当前掩膜快照），
    #    这样即使进程被暂停/中断，也能从 json 得知进度并断点续传。
    def _snapshot(status="processing"):
        snap = dict(info)
        snap["status"] = status
        snap["elapsed_seconds"] = round(time.time() - patient_start, 2)
        snap["masks"] = sorted(f for f in os.listdir(patient_output_dir)
                               if f.endswith('.nii.gz'))
        write_segmentation_info(info_json_path, snap)

    try:
        _snapshot("processing")  # 开工前先记录选中输入

        # 💡 引擎 1：提取全量大器官 (肺叶、主动脉、主气管、整体心脏)
        totalsegmentator(nifti_path, patient_output_dir, device="gpu")
        _snapshot("processing")  # 引擎1完成，记录掩膜

        # 💡 引擎 2：提取肺部微观结构 (肺血管网、支气管树)
        # 注意: 使用 LEGACY 任务以输出 lung_vessels / lung_trachea_bronchia
        # （默认 task="lung_vessels" 在 v2 中输出 lung_arteries/lung_veins/lung_airways/lung_airways_wall，
        #   与下游特征提取 extract_featureX.py 读取的文件名不匹配）
        print("   -> 正在追加提取肺微观树状网络 (LEGACY)...")
        totalsegmentator(nifti_path, patient_output_dir, task="lung_vessels_LEGACY")
        _snapshot("processing")  # 引擎2完成，记录掩膜

        # 💡 引擎 3：提取高精心血管结构 (心肌、四腔室、肺动脉)
        print("   -> 正在追加提取高精心血管与肺动脉...")
        totalsegmentator(nifti_path, patient_output_dir, task="heartchambers_highres")
        _snapshot("processing")  # 引擎3完成，记录掩膜

        # 🧹 暴力清理：移除所有无关脏器，让目录极其干净
        print("   -> 正在清理无用器官文件...")
        for file in os.listdir(patient_output_dir):
            if file.endswith(".nii.gz") and file not in KEEP_FILES:
                os.remove(os.path.join(patient_output_dir, file))

        # 记录最终掩膜
        info["masks"] = sorted(f for f in os.listdir(patient_output_dir) if f.endswith(".nii.gz"))
        info["elapsed_seconds"] = round(time.time() - patient_start, 2)
        info["status"] = "success"
        write_segmentation_info(info_json_path, info)

        print(f"   ✅ {patient_name} 分割成功！层数: {n_slices}，"
              f"掩膜数: {len(info['masks'])}，耗时: {info['elapsed_seconds']/60:.2f} 分钟")
        return True, info

    except Exception as e:
        info["status"] = "failed"
        info["error"] = str(e)
        info["elapsed_seconds"] = round(time.time() - patient_start, 2)
        info["masks"] = sorted(f for f in os.listdir(patient_output_dir)
                               if f.endswith('.nii.gz'))
        # 失败时也写入 json（含已生成的掩膜），方便排查 + 断点续传
        write_segmentation_info(info_json_path, info)
        print(f"   ❌ {patient_name} 分割失败！错误原因: {e}")
        return False, info


def batch_segment_largest_slice(input_base_dir, output_base_dir):
    print("🚀 启动「层数最多 NIfTI」批量分割流水线...")
    print(f"输入目录: {input_base_dir}")
    print(f"输出目录: {output_base_dir}\n")

    patient_folders = sorted(
        [d for d in os.listdir(input_base_dir)
         if os.path.isdir(os.path.join(input_base_dir, d))]
    )

    if not patient_folders:
        print(f"❌ 在 {input_base_dir} 目录下没有找到任何患者文件夹，请检查路径！")
        return

    total = len(patient_folders)
    print(f"📦 共检测到 {total} 个患者文件夹，准备开始处理。\n")
    os.makedirs(output_base_dir, exist_ok=True)

    overall_start = time.time()
    results = []
    success_count = 0
    fail_count = 0
    skip_count = 0

    for idx, patient_name in enumerate(patient_folders, 1):
        patient_path = os.path.join(input_base_dir, patient_name)
        print(f"\n[{idx}/{total}] 正在处理: {patient_name} ...")

        ok, info = process_patient(patient_path, patient_name, output_base_dir)
        if info["status"] == "skipped":
            skip_count += 1
        elif ok:
            success_count += 1
        else:
            fail_count += 1
        results.append(info)

    # 💾 汇总所有患者的分割信息
    summary_path = os.path.join(output_base_dir, "segmentation_summary.json")
    summary = {
        "input_base_dir": input_base_dir,
        "output_base_dir": output_base_dir,
        "total_patients": total,
        "success": success_count,
        "skipped": skip_count,
        "failed": fail_count,
        "total_elapsed_seconds": round(time.time() - overall_start, 2),
        "patients": results,
    }
    write_segmentation_info(summary_path, summary)

    overall_time = (time.time() - overall_start) / 3600
    print("\n" + "=" * 50)
    print(f"🎉 批量任务彻底结束！")
    print(f"⏱️ 总耗时: {overall_time:.2f} 小时")
    print(f"📊 统计: 成功 {success_count} 例，跳过 {skip_count} 例，失败 {fail_count} 例")
    print(f"💾 分割信息已保存至: {summary_path}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="找到每个患者文件夹中层数最多的 NIfTI 文件并进行全器官分割")
    parser.add_argument("-i", "--input", default=r"E:\DICOM\2026-05-nifti",
                        help="患者 NIfTI 文件夹所在目录（已由 dcm2nii_batch_arg.py 生成）")
    parser.add_argument("-o", "--output", default=r"D:\copd-radiomics\seg_results",
                        help="分割结果输出目录")
    args = parser.parse_args()

    batch_segment_largest_slice(args.input, args.output)
