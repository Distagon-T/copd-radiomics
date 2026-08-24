#!/usr/bin/env bash
# =========================================================================
# Mac 统一流水线一键安装脚本（TotalSeg 分割 + Fast PyRadiomics 特征提取）
# =========================================================================
# pyradiomics 是纯 CPU 库（SimpleITK + numpy），无 CUDA 依赖，Mac 原生支持。
# TotalSegmentator 在 Mac 上走 CPU（较慢但可用；只用已有掩膜可 --radiomics-only）。
# 用法:  chmod +x setup_mac.sh && ./setup_mac.sh
set -e

ENV_NAME="mac-pyradiomics"
PYTHON_VERSION="3.10"

echo "==== 1. 创建 conda 环境: $ENV_NAME ===="
if command -v conda >/dev/null 2>&1; then
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
else
    echo "[warn] 未检测到 conda，尝试用 python venv ..."
    python3 -m venv "$ENV_NAME" && source "$ENV_NAME/bin/activate"
fi

echo "==== 2. 安装依赖（纯 CPU，无需 CUDA） ===="
python -m pip install --upgrade pip
python -m pip install -r requirements_mac.txt

echo "==== 3. 验证导入 ===="
python - <<'PY'
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy, SimpleITK, nibabel, pandas, scipy
import radiomics
print('numpy', numpy.__version__)
print('SimpleITK', SimpleITK.__version__)
print('pyradiomics', radiomics.__version__)
print('--- 全部导入成功 (纯 CPU，无 GPU 依赖) ---')
PY

echo ""
echo "==== 完成！使用方式 ===="
echo "conda activate $ENV_NAME"
echo "# ① 完整流程（TotalSeg 分割 + Fast 特征提取）:"
echo "python run_pipeline.py -i /path/CT -o /path/out --workers 2"
echo "# ② 只做特征提取（复用已有掩膜，Mac 上推荐，快）:"
echo "python run_pipeline.py -i /path/CT -o /path/out --radiomics-only --workers 2"
echo "# ③ 旧入口（纯特征提取）:"
echo "python run_radiomics.py --nifti-dir /path/CT --seg-dir /path/seg --workers 2"
echo "# 指定患者: 追加 --patients 患者A,患者B"
echo "# ④ Lite 模式（全 ROI shape+firstorder，更快）:"
echo "python run_radiomics_lite.py --nifti-dir /path/CT --seg-dir /path/seg --workers 2"
