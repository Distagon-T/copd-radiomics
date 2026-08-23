#!/usr/bin/env bash
# =========================================================================
# build_and_export.sh — 在 Linux 服务器上构建 Docker 镜像并导出离线 tar
# 用法:
#   ./build_and_export.sh [image_name] [tag]
# 默认: airway-seg:latest
#
# 产物: airway-seg_<tag>.tar  (拷到医院服务器, docker load < xxx.tar)
# =========================================================================
set -euo pipefail

IMG_NAME="${1:-airway-seg}"
IMG_TAG="${2:-latest}"
FULL="${IMG_NAME}:${IMG_TAG}"
TAR="${IMG_NAME}_${IMG_TAG}.tar"

echo "=============================================="
echo "构建镜像: ${FULL}"
echo "=============================================="

# 1) 构建镜像 (可选 --no-cache 强制重装)
docker build -t "${FULL}" .

# 2) 导出镜像为 tar
echo "导出镜像 -> ${TAR}"
docker save -o "${TAR}" "${FULL}"

# 3) 显示大小
SIZE=$(du -h "${TAR}" | cut -f1)
echo "=============================================="
echo "✅ 完成! 镜像包: ${TAR} (${SIZE})"
echo "   拷贝到医院服务器后执行:"
echo "     docker load < ${TAR}"
echo "     docker run --gpus all --shm-size=16g \\"
echo "       -v /医院输入路径:/data/input -v /医院输出路径:/data/output \\"
echo "       ${FULL} --input-dir /data/input --output-dir /data/output"
echo "=============================================="
