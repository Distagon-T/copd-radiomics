# 医院服务器离线分割部署包

把 `TotalSegmentator` 和 `Connectivity-Aware-Airway-Segmentation` 打包，拷贝到医院服务器
（Ubuntu 22.04 + 双 NVIDIA RTX 4090）跑分割任务。**完全离线**，无需联网。

## 目录结构

```
docker-airway-seg/
├── Dockerfile                 # 镜像构建文件
├── requirements.txt           # Python 依赖
├── run_segmentation.py        # 统一分割入口 (TotalSeg + Airway)
├── prepare_package.py         # (Windows 端) 准备代码+权重
├── build_and_export.sh        # (Linux 端) 构建 + 导出 tar
├── conda_pack.sh              # (Linux 端) conda-pack 备用方案
├── code/airway_seg/           # airway 分割代码 + airway_model.pth 权重
├── weights/nnunet/            # TotalSegmentator 全部 nnUNet 权重 (~2.2GB)
├── scripts/                   # 批量分割脚本参考
└── wheelhouse/                # (可选) 离线 pip 安装包
```

---

## 方案一：Docker 镜像（推荐）

### 1. 在医院服务器上安装 Docker + NVIDIA 容器工具包（一次性）
> 医院服务器需要联网安装 Docker 本身（Docker 引擎安装一次即可，之后跑分割完全离线）。
> 如果 Docker 也无法在线装，请用下面的"方案二：conda-pack"。

```bash
# 1) 安装 Docker
curl -fsSL https://get.docker.com | sudo sh

# 2) 安装 nvidia-container-toolkit（让容器用上 4090）
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 3) 验证
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 2. 在任意 Linux 机器上构建镜像并导出（需装 Docker，可联网装依赖）
```bash
# 在 docker-airway-seg 目录下
chmod +x build_and_export.sh
./build_and_export.sh airway-seg latest
# 产物: airway-seg_latest.tar (约 8-10 GB)
```

### 3. 拷贝到医院服务器并加载
用移动硬盘/U盘把 `airway-seg_latest.tar` 拷过去：
```bash
docker load < airway-seg_latest.tar
docker images | grep airway-seg   # 确认已加载
```

### 4. 运行分割
```bash
# 输入: /data/input 下每个子文件夹 = 一个患者
#   - 支持 <患者>_dicom_info.json (自动选增强/最大层数序列)
#   - 支持单个/多个 .nii.gz (自动识别)
# 输出: /data/output 下生成 <患者>_masks/ 和 segmentation_summary.json
docker run --gpus all --shm-size=16g \
  -v /医院输入路径:/data/input \
  -v /医院输出路径:/data/output \
  airway-seg:latest \
  --input-dir /data/input --output-dir /data/output --device cuda:0
```

**双 4090 并行跑两个队列**（分两批输入目录）：
```bash
# GPU0 处理队列 A
docker run --gpus '"device=0"' --shm-size=16g \
  -v /data/queueA:/data/input -v /data/outA:/data/output \
  airway-seg:latest -i /data/input -o /data/output -d cuda:0 &

# GPU1 处理队列 B
docker run --gpus '"device=1"' --shm-size=16g \
  -v /data/queueB:/data/input -v /data/outB:/data/output \
  airway-seg:latest -i /data/input -o /data/output -d cuda:1 &
```

**常用参数**：
| 参数 | 说明 |
|---|---|
| `--airway-only` | 只跑气道分割（跳过 TotalSeg） |
| `--totalseg-only` | 只跑 TotalSegmentator（跳过气道） |
| `--device cuda:0` | 指定 GPU |

---

## 方案二：conda-pack（无需 Docker）

> 如果医院无法安装 Docker，可用 conda-pack 打包 Python 环境，直接解压运行。

### 1. 在本地构建 conda 环境并打包
```bash
# 在 Linux 机器上
conda create -n airway-seg python=3.10 -y
conda activate airway-seg
pip install torch==2.6.0 monai==1.4.0 SimpleITK==2.4.1 nibabel==5.3.2 TotalSegmentator==2.13.0
pip install conda-pack
conda pack -n airway-seg -o airway-seg_env.tar.gz
```
把 `airway-seg_env.tar.gz` + `code/` + `weights/` + `run_segmentation.py` 一起拷到医院。

### 2. 医院端解压并运行
```bash
# 解压环境 (任选目录, 例如 ~/airway-seg-env)
mkdir -p ~/airway-seg-env && tar -xzf airway-seg_env.tar.gz -C ~/airway-seg-env
source ~/airway-seg-env/bin/activate
conda-unpack   # conda-pack 专用，修正路径

# 运行
export TOTALSEG_HOME_DIR=/绝对路径/weights
python run_segmentation.py --input-dir /医院输入路径 --output-dir /医院输出路径 --device cuda:0
```

---

## 输入数据约定

每个患者一个子文件夹，支持三种形态（自动识别）：

| 形态 | 说明 |
|---|---|
| 带 `dicom_info.json` | 自动选 `+C` 增强序列；无增强则选层数最多序列 |
| 单个 `.nii.gz` | 直接分割 |
| 多个 `.nii.gz` | 选 z 轴层数最多者 |

---

## 输出说明

每个患者输出到 `<输出>/<患者>_masks/`：

- **TotalSeg 黄金 16 靶区**：肺叶 ×5、肺血管/支气管树、主动脉、肺动脉、气管、心脏 + 心肌/四腔（对比剂 CT 用 `batch_segment_contrast_ct.py` 逻辑）
- **气道树**：`<患者>_airway.nii.gz`（connectivity-aware 精细气道，命名与本机一致）
- `<患者>_segmentation_info.json`（该患者信息）
- `segmentation_summary.json`（队列汇总）

> ⚠️ TotalSeg 的 `lung_vessels` 任务使用 `lung_vessels_LEGACY` 以输出 `lung_vessels.nii.gz` +
> `lung_trachea_bronchia.nii.gz`（与下游特征提取命名一致），与本地测试一致。
