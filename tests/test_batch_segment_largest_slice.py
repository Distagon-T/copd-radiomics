"""
单元测试脚本 —— 只测试 tests/test_data 下的前 5 位患者。

被测目标: batch_segment_largest_slice.py
  - find_largest_slice_nifti : 找到层数最多的 NIfTI（逻辑测试）
  - inspect_nifti            : 读取体积信息（逻辑测试）
  - write_segmentation_info  : 写出分割信息 JSON（逻辑测试）
  - KEEP_FILES 配置完整性     : 黄金 16 靶区（逻辑测试）
  - process_patient          : 端到端 GPU 分割（GPU 集成测试，实际调用 totalsegmentator）

GPU 集成测试：
  - 真实调用 totalsegmentator（device="gpu"），对 5 位患者逐一分割。
  - 输出写入 tests/test_output（不污染正式 seg_results 目录）。
  - 若机器无可用 CUDA GPU，会自动跳过（pytest -m "gpu"）。

运行（全部，含 GPU 分割，耗时较长）:
  cd <repo_root>
  python -m pytest tests/test_batch_segment_largest_slice.py -v

只跑 GPU 集成测试:
  python -m pytest tests/test_batch_segment_largest_slice.py -m gpu -v

只跑快速逻辑测试（跳过 GPU）:
  python -m pytest tests/test_batch_segment_largest_slice.py -m "not gpu" -v
"""
import os
import json
import sys
import nibabel as nib
import pytest

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

# 确保能导入被测脚本（脚本与被测脚本在同一项目根目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import batch_segment_largest_slice as bsl

# ── 测试数据配置：只测试这前 5 位患者 ─────────────────────────────
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "tests", "test_data")

# GPU 分割测试的输出目录（不污染正式 seg_results）
GPU_TEST_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tests", "test_output")

# 前 5 位患者文件夹名（与数据文件夹前 5 个一致）
FIRST_FIVE_PATIENTS = [
    "20130130_Anonymous_CT_1.2.840.113619.2.55.3.1678396440.5613.1359500545.33501186",
    "20130319_Anonymous_CT_1.2.840.113619.2.55.3.1678396440.5821.1363647971.2402916",
    "20130406_Anonymous_CT_1.2.840.113619.2.55.3.1678396440.5704.1365203438.49346934",
    "20130417_Anonymous_CT_1.2.840.113619.2.55.3.1678396440.5794.1366156304.50851136",
    "20130506_Anonymous_CT_1.2.840.113619.186.3512539255108.20130506182920614.20102811",
]


def _patient_dir(patient_name):
    """返回某位患者的测试数据目录，若不存在则抛错（防止悄悄漏测）。"""
    pdir = os.path.join(TEST_DATA_DIR, patient_name)
    if not os.path.isdir(pdir):
        raise FileNotFoundError(f"测试数据缺失: {pdir}")
    return pdir


def _all_nifti_slices(patient_dir):
    """返回 {nifti 文件名: 层数}，供测试对比期望值。"""
    result = {}
    for f in os.listdir(patient_dir):
        if f.endswith(".nii.gz"):
            img = nib.load(os.path.join(patient_dir, f))
            shape = img.shape
            result[f] = int(shape[2]) if len(shape) >= 3 else int(shape[0])
    return result


# ── 测试用例 ─────────────────────────────────────────────────────

@pytest.mark.parametrize("patient_name", FIRST_FIVE_PATIENTS)
def test_find_largest_slice_selects_max(patient_name):
    """对每位患者，find_largest_slice_nifti 应选中层数最多的那个 NIfTI。"""
    pdir = _patient_dir(patient_name)
    best_path, best_slices = bsl.find_largest_slice_nifti(pdir)

    assert best_path is not None, f"{patient_name}: 未找到任何 NIfTI"
    assert best_slices > 0, f"{patient_name}: 层数应大于 0"

    # 期望值 = 所有 NIfTI 中的最大层数
    slices_map = _all_nifti_slices(pdir)
    expected = max(slices_map.values())

    assert best_slices == expected, (
        f"{patient_name}: 返回层数 {best_slices} != 期望 {expected}"
    )
    assert os.path.basename(best_path) in slices_map
    assert slices_map[os.path.basename(best_path)] == expected, (
        f"{patient_name}: 选中的文件不是层数最多的那个"
    )


@pytest.mark.parametrize("patient_name", FIRST_FIVE_PATIENTS)
def test_find_largest_slice_returns_real_file(patient_name):
    """选中的文件必须真实存在且是 .nii.gz。"""
    pdir = _patient_dir(patient_name)
    best_path, best_slices = bsl.find_largest_slice_nifti(pdir)
    assert best_path is not None
    assert os.path.isfile(best_path)
    assert best_path.endswith(".nii.gz")
    assert best_slices == bsl.inspect_nifti(best_path)["slices"]


@pytest.mark.parametrize("patient_name", FIRST_FIVE_PATIENTS)
def test_inspect_nifti_fields(patient_name):
    """inspect_nifti 返回的字段应完整且与 nibabel 读取一致。"""
    pdir = _patient_dir(patient_name)
    best_path, _ = bsl.find_largest_slice_nifti(pdir)
    info = bsl.inspect_nifti(best_path)

    img = nib.load(best_path)
    assert info["shape"] == list(img.shape)
    assert info["slices"] == (int(img.shape[2]) if len(img.shape) >= 3 else int(img.shape[0]))
    assert len(info["spacing"]) == len(img.header.get_zooms())
    assert info["pixdim"] == [float(x) for x in img.header.get_zooms()]
    # 必须可直接 json 序列化（防止 numpy float32 等报错）
    json.dumps(info)


def test_write_segmentation_info_roundtrip():
    """write_segmentation_info 应写出合法 JSON，且内容可完整读回。"""
    tmp_json = os.path.join(TEST_DATA_DIR, "_tmp_test_info.json")
    sample_info = {
        "patient_folder": "test_patient",
        "status": "success",
        "slice_count": 100,
        "masks": ["a.nii.gz", "b.nii.gz"],
    }
    try:
        bsl.write_segmentation_info(tmp_json, sample_info)
        assert os.path.isfile(tmp_json)
        with open(tmp_json, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == sample_info
    finally:
        if os.path.exists(tmp_json):
            os.remove(tmp_json)


def test_keep_files_complete():
    """黄金 16 靶区配置必须完整且唯一。"""
    assert len(bsl.KEEP_FILES) == 16
    assert len(set(bsl.KEEP_FILES)) == 16
    for name in bsl.KEEP_FILES:
        assert name.endswith(".nii.gz")
    # 关键靶区必须存在
    required = {"lung_upper_lobe_left.nii.gz", "lung_lower_lobe_right.nii.gz",
                "lung_vessels.nii.gz", "lung_trachea_bronchia.nii.gz",
                "aorta.nii.gz", "pulmonary_artery.nii.gz", "trachea.nii.gz",
                "heart.nii.gz", "heart_myocardium.nii.gz",
                "heart_atrium_left.nii.gz", "heart_ventricle_right.nii.gz"}
    assert required.issubset(set(bsl.KEEP_FILES))


def test_all_five_patients_have_dicom_info_json():
    """每位患者的测试数据目录都应包含对应的 dicom_info.json。"""
    for patient_name in FIRST_FIVE_PATIENTS:
        pdir = _patient_dir(patient_name)
        info_json = os.path.join(pdir, f"{patient_name}_dicom_info.json")
        assert os.path.isfile(info_json), f"{patient_name}: 缺少 dicom_info.json"
        with open(info_json, encoding="utf-8") as f:
            data = json.load(f)
        assert "Series" in data and len(data["Series"]) > 0


# ═══ GPU 端到端分割集成测试 ═════════════════════════════════════

@pytest.mark.gpu
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="当前机器无可用 CUDA GPU，跳过 GPU 分割测试")
@pytest.mark.parametrize("patient_name", FIRST_FIVE_PATIENTS)
def test_gpu_segmentation_runs_successfully(patient_name):
    """真实调用 totalsegmentator（GPU）对患者进行分割，验证全流程成功。"""
    pdir = _patient_dir(patient_name)
    os.makedirs(GPU_TEST_OUTPUT_DIR, exist_ok=True)

    ok, info = bsl.process_patient(pdir, patient_name, GPU_TEST_OUTPUT_DIR)

    assert ok, f"{patient_name}: process_patient 返回失败: {info.get('error')}"
    assert info["status"] in ("success", "skipped"), \
        f"{patient_name}: 状态异常: {info['status']}"

    patient_output_dir = os.path.join(GPU_TEST_OUTPUT_DIR, f"{patient_name}_masks")
    assert os.path.isdir(patient_output_dir), f"{patient_name}: 输出目录未创建"

    # 目标掩膜应齐全
    existing_masks = set(info.get("masks", []))
    missing = set(bsl.KEEP_FILES) - existing_masks
    assert not missing, f"{patient_name}: 缺失掩膜 {missing}"

    # info json 应已写出且可读回
    info_json_path = os.path.join(patient_output_dir, f"{patient_name}_segmentation_info.json")
    assert os.path.isfile(info_json_path), f"{patient_name}: 缺少 segmentation_info.json"
    with open(info_json_path, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["patient_folder"] == patient_name
    assert saved["selected_nifti"] is not None
    assert saved["slice_count"] is not None and saved["slice_count"] > 0
    assert saved["status"] in ("success", "skipped")


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="当前机器无可用 CUDA GPU，跳过 GPU 分割测试")
def test_gpu_segmentation_writes_summary_json():
    """GPU 批量分割结束后，应生成汇总 JSON（segmentation_summary.json）。"""
    # 只对 5 位患者做汇总，输出到测试输出目录
    results = []
    for patient_name in FIRST_FIVE_PATIENTS:
        pdir = _patient_dir(patient_name)
        ok, info = bsl.process_patient(pdir, patient_name, GPU_TEST_OUTPUT_DIR)
        results.append(info)

    summary_path = os.path.join(GPU_TEST_OUTPUT_DIR, "segmentation_summary.json")
    summary = {
        "input_base_dir": TEST_DATA_DIR,
        "output_base_dir": GPU_TEST_OUTPUT_DIR,
        "total_patients": len(FIRST_FIVE_PATIENTS),
        "patients": results,
    }
    bsl.write_segmentation_info(summary_path, summary)

    assert os.path.isfile(summary_path)
    with open(summary_path, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["total_patients"] == len(FIRST_FIVE_PATIENTS)
    assert len(saved["patients"]) == len(FIRST_FIVE_PATIENTS)
    for p in saved["patients"]:
        assert p["status"] in ("success", "skipped")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
