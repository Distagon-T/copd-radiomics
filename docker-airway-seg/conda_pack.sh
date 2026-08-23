#!/usr/bin/env bash
# =========================================================================
# conda_pack.sh — 用 conda-pack 打包 Python 环境 (无需 Docker 的备用方案)
# 在「可联网的 Linux 机器」上运行一次，产出:
#   - airway-seg_env.tar.gz   (conda 环境)
#   - airway-seg_data.tar.gz  (代码 + 权重)
# 用法: ./conda_pack.sh
# =========================================================================
set -euo pipefail

ENV_NAME="airway-seg"
PYTHON_VER="3.10"

echo "=============================================="
echo "创建 conda 环境: ${ENV_NAME}"
echo "=============================================="

# 1) 创建环境（若已存在则跳过）
if ! conda env list | grep -q "${ENV_NAME}"; then
    conda create -n "${ENV_NAME}" python=${PYTHON_VER} -y
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# 2) 安装依赖（锁版本，与本地验证一致）
pip install torch==2.6.0 monai==1.4.0 SimpleITK==2.4.1 nibabel==5.3.2 TotalSegmentator==2.13.0
pip install conda-pack

# 3) 打包环境
echo "打包 conda 环境 -> airway-seg_env.tar.gz"
conda pack -n "${ENV_NAME}" -o airway-seg_env.tar.gz

# 4) 打包代码 + 权重
echo "打包代码 + 权重 -> airway-seg_data.tar.gz"
tar -czf airway-seg_data.tar.gz \
    run_segmentation.py \
    code/airway_seg \
    weights/nnunet \
    scripts

SIZE=$(du -h airway-seg_env.tar.gz | cut -f1)
SIZE2=$(du -h airway-seg_data.tar.gz | cut -f1)
echo "=============================================="
echo "✅ 完成!"
echo "   airway-seg_env.tar.gz  (${SIZE})"
echo "   airway-seg_data.tar.gz (${SIZE2})"
echo ""
echo "   医院端解压运行:"
echo "     tar -xzf airway-seg_env.tar.gz -C ~/airway-seg-env && cd ~/airway-seg-env"
echo "     source bin/activate && conda-unpack"
echo "     tar -xzf airway-seg_data.tar.gz"
echo "     export TOTALSEG_WEIGHTS_FOLDER=\$(pwd)/weights/nnunet"
echo "     python run_segmentation.py -i /输入 -o /输出 -d cuda:0"
echo "=============================================="
