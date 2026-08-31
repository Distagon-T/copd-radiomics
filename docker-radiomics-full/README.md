# docker-radiomics-full — TotalSegmentator 分割 + PyRadiomics 完整特征 (离线 Docker)

面向 **医院 Ubuntu 22 + 双 4090 服务器** 的**完整版**部署包：
完成 **TotalSegmentator 分割 → PyRadiomics 全部耗时特征 → 合并 CSV** 全流程。

> **与 docker-radiomics-seg（快速版）的区别**
> - 本包（full）：计算**全部耗时特征**，单患者约 30~40 分钟
>   - 所有 ROI：shape + firstorder + **全部纹理类**（GLCM/GLRLM/GLSZM/GLDM/NGTDM）
>   - 肺叶：额外 **LoG 斑点滤波** (sigma=[1.0, 3.0])
>   - 心肌：额外 **Wavelet 高频子带**
>   - `lung_vessels`：**保留完整 shape 特征**（最耗时，约 35 分钟）
> - `docker-radiomics-seg`（快速版）：单患者约 2~3 分钟
>   - 仅 shape + firstorder；`lung_vessels` 只算 firstorder

## 目录结构

```
docker-radiomics-full/
├── Dockerfile              # Ubuntu22 + CUDA12.4 + PyTorch2.6
├── requirements.txt        # Python 依赖（含 pyradiomics / TotalSegmentator）
├── run_pipeline.py         # 统一入口：分割 + 特征提取 + 合并
├── radiomics_extract.py    # PyRadiomics 完整特征核心（全部耗时特征）
├── declared_features_lib.py  # 申报清单补算特征（CAC/心包脂肪/FAI/主动脉/心胸比/血管CSA）
├── merge_radiomics.py      # 合并 JSON -> CSV
├── prepare_package.py      # 离线打包工具（收集权重 / 生成 wheelhouse）
├── weights/nnunet/         # TotalSegmentator 权重（已就位，2230MB）
└── wheelhouse/             # （可选）离线 pip wheel
```

## 功能

对每个患者：
1. **TotalSegmentator 三引擎分割**：全器官 → `lung_vessels_LEGACY`（肺血管/支气管）→ `heartchambers_highres`（高精心血管），最后清理保留 **16 个黄金靶区**。
2. **PyRadiomics 完整特征**（每个掩膜）：
   - 所有 ROI：shape + firstorder + 纹理类（GLCM/GLRLM/GLSZM/GLDM/NGTDM）
   - 肺叶：额外 LoG 滤波 (sigma=[1.0, 3.0])
   - 心肌：额外 Wavelet
   - lung_vessels：完整 shape（不省略）
3. **四类 COPD 表型指标**：
   - 肺叶级肺气肿（LAA-950% / Perc15 / 体积 / 占全肺比）
   - 气道-肺叶耦合（气道在各肺叶占比）
   - 心肺共病（PA/Ao 直径比 / RV/LV 容积比 / CAC 钙化体积）
   - 膈肌形态（肺底轮廓填充比）
4. **申报清单补算特征**（`declared_features_lib.py`）：CAC Agatston/MS · 心包脂肪 · FAI ·
   主动脉直径/壁厚 · 心胸比 · 血管体积/CSA · 主肺动脉等效直径
   （与 Windows 单脚本双输出一致；BronchoArtery_Ratio 由 AirQuant MATLAB 计算）
5. **合并 CSV**：全部患者 → `radiomics_all_patients.csv`

**断点续传**：掩膜已齐全的患者跳过分割；`radiomics.json` 已存在的患者完全跳过（`--force` 重算）。

## 构建（开发机）

```bash
# 权重已就位（weights/nnunet, 2230MB），直接构建
# 若需完全离线，先执行: python prepare_package.py --wheels
docker build -t radiomics-full:latest .
```

## 运行（医院 4090 服务器）

```bash
# 单卡
docker run --gpus all --shm-size=16g \
  -v /path/to/input:/data/input \
  -v /path/to/output:/data/output \
  radiomics-full:latest \
  --input-dir /data/input --output-dir /data/output --device cuda:0

# 双 4090：把患者分成两半，用 cuda:0 和 cuda:1 并行跑两份
docker run --gpus '"device=0"' ... --device cuda:0 --input-dir /data/input_0 --output-dir /data/output_0
docker run --gpus '"device=1"' ... --device cuda:1 --input-dir /data/input_1 --output-dir /data/output_1
```

### 可选参数
| 参数 | 说明 |
|------|------|
| `--seg-only` | 只分割，不做特征提取 |
| `--radiomics-only` | 只做特征提取（用已有掩膜） |
| `--force` | radiomics json 已存在也重算 |
| `--device` | cuda:0 / cuda:1 / cpu |

### 输出
```
<output>/
├── <患者>_masks/                 # 16 个靶区掩膜
│   └── <患者>_segmentation_info.json
├── <患者>_radiomics.json         # 每个患者的特征（约 500+ 维）
├── radiomics_all_patients.csv    # 合并表（一行一患者）
└── segmentation_summary.json
```

## 输入约定
- `--input-dir` 下每个子文件夹 = 一个患者（含 `.nii.gz` 或 NIfTI 序列）
- 自动选择输入：优先 `+C` 增强序列 → 层数最多序列 → 唯一文件
- 支持 `<患者>_dicom_info.json` 辅助选择

## 已知注意点
- **本包为完整版**，计算全部耗时特征（lung_vessels 完整 shape、肺叶 LoG/纹理、
  心肌 Wavelet）。单患者约 30~40 分钟，16 患者全量约 8~10 小时。
- 若需要快速版（~2-3 分钟/患者，lung_vessels 只算 firstorder），请用
  `docker-radiomics-seg` 快速版包。
- 内存：单患者峰值约 3~6 GB（Wavelet/LoG 滤波 + 纹理矩阵），双卡并行建议 `--shm-size=16g`。

---

## 方案二：conda-pack（无需 Docker）

> 如果医院无法安装 Docker，可用 conda-pack 打包 Python 环境，直接解压运行。

### 1. 在本地构建 conda 环境并打包
```bash
# 在任意 Linux x86_64 机器上（需可联网装依赖）
conda create -n radiomics-full python=3.10 -y
conda activate radiomics-full
pip install torch==2.6.0 monai==1.4.0 SimpleITK==2.4.1 nibabel==5.3.2 \
            numpy pandas scipy pyradiomics==3.0.1 TotalSegmentator==2.13.0
pip install conda-pack
conda pack -n radiomics-full -o radiomics-full_env.tar.gz
```
把 `radiomics-full_env.tar.gz` + 本目录的代码（`run_pipeline.py`、
`radiomics_extract.py`、`declared_features_lib.py`、`merge_radiomics.py`）
+ `weights/` 一起拷到医院。

### 2. 医院端解压并运行
```bash
# 解压环境 (任选目录, 例如 ~/radiomics-env)
mkdir -p ~/radiomics-env && tar -xzf radiomics-full_env.tar.gz -C ~/radiomics-env
source ~/radiomics-env/bin/activate
conda-unpack   # conda-pack 专用，修正路径

# 运行（注意权重路径指向医院端的 weights/nnunet）
export TOTALSEG_HOME_DIR=/绝对路径/weights
python run_pipeline.py --input-dir /医院输入路径 --output-dir /医院输出路径 --device cuda:0
```

> ⚠️ **权重路径说明**：TotalSegmentator 通过 `TOTALSEG_HOME_DIR` 环境变量定位权重
> （容器内默认 `/app/weights`；conda-pack 方式需手动 export 到医院端 weights 目录，
> 该目录内需含 `nnunet/results/`）。
