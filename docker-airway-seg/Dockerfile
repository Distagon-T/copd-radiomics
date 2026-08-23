# =========================================================================
# Dockerfile — 气道 + 全器官分割 (TotalSegmentator + Connectivity-Aware Airway)
# 基础镜像: PyTorch 官方 (Ubuntu 22.04 + CUDA 12.4 + cuDNN 9)
# 构建:  docker build -t airway-seg:latest .
# 运行:  docker run --gpus all --shm-size=16g \
#          -v /path/input:/data/input -v /path/output:/data/output \
#          airway-seg:latest --input-dir /data/input --output-dir /data/output
# =========================================================================
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TOTALSEG_HOME_DIR=/app/weights \
    KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /app

# 系统依赖 (itk 等编译需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
        && rm -rf /var/lib/apt/lists/*

# 复制 requirements 并离线安装
COPY requirements.txt /app/requirements.txt
# 说明: 若服务器可联网, 用下面这行直接装 (推荐)
RUN pip install --no-cache-dir -r /app/requirements.txt
# 若完全离线(无网), 把下方这行取消注释, 并把 whl 包放到 ./wheelhouse:
# RUN pip install --no-cache-dir --find-links=/app/wheelhouse -r /app/requirements.txt

# 复制代码与权重 (由 prepare_package.py 预先生成)
COPY run_segmentation.py /app/run_segmentation.py
COPY code/airway_seg /app/airway_seg
COPY weights/nnunet /app/weights/nnunet
COPY scripts /app/scripts

# 确保 airway 模型找到权重 (相对路径, 由 config 指定)
RUN ls -la /app/airway_seg/checkpoints/ && \
    ls /app/weights/nnunet/results/ && \
    chmod +x /app/run_segmentation.py

# 验证安装
RUN python -c "import torch, monai, totalsegmentator, SimpleITK, nibabel; \
    print('torch', torch.__version__, '| cuda available', torch.cuda.is_available()); \
    print('monai', monai.__version__, '| TotalSegmentator', totalsegmentator.__version__)" \
    || echo "WARN: 部分导入失败，请检查依赖"

WORKDIR /app
ENTRYPOINT ["python", "/app/run_segmentation.py"]
