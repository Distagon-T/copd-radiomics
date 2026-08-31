# =========================================================================
# Dockerfile — TotalSegmentator 分割 + PyRadiomics 特征提取
# 基础镜像: PyTorch 官方 (Ubuntu 22.04 + CUDA 12.4 + cuDNN 9)
# 目标: 医院 Ubuntu 22 + 双 4090 服务器，离线部署
#
# 构建:  docker build -t radiomics-seg:latest .
# 运行:  docker run --gpus all --shm-size=16g \
#          -v /path/input:/data/input -v /path/output:/data/output \
#          radiomics-seg:latest --input-dir /data/input --output-dir /data/output
#
# 说明: 若服务器可联网, 用 RUN pip install 那行直接装;
#       若完全离线, 用 wheelhouse 方案 (见 prepare_package.py)
# =========================================================================
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TOTALSEG_HOME_DIR=/app/weights \
    KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /app

# 系统依赖 (itk / SimpleITK 运行需要 libgl, libglib)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
        && rm -rf /var/lib/apt/lists/*

# 复制依赖清单并安装
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
# 完全离线时取消注释下面这行（先运行 prepare_package.py 生成 wheelhouse）:
# RUN pip install --no-cache-dir --find-links=/app/wheelhouse -r /app/requirements.txt

# 复制代码
COPY run_pipeline.py /app/run_pipeline.py
COPY radiomics_extract.py /app/radiomics_extract.py
COPY declared_features_lib.py /app/declared_features_lib.py
COPY merge_radiomics.py /app/merge_radiomics.py
COPY weights /app/weights

# 验证
RUN ls /app/weights/nnunet/results/ && \
    chmod +x /app/run_pipeline.py && \
    python -c "import torch, monai, totalsegmentator, SimpleITK, nibabel, radiomics; \
        print('torch', torch.__version__, '| cuda', torch.cuda.is_available()); \
        print('TotalSeg', totalsegmentator.__version__, '| pyradiomics', radiomics.__version__)" \
    || echo "WARN: 部分导入失败，请检查依赖"

ENTRYPOINT ["python", "/app/run_pipeline.py"]
