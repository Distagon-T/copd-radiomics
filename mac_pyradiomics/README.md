# mac_pyradiomics — Mac 统一流水线（TotalSeg 分割 + Fast PyRadiomics，纯 CPU）

Mac 端**唯一**的影像流水线包：可做 TotalSegmentator 分割，也可只做 Fast 特征提取。
特征策略与 Windows `compute_patient_radiomics_fast.py` **完全一致**，跨端列名统一。

## 🔑 核心特性

- **纯 CPU**：pyradiomics 基于 SimpleITK + numpy，**无 CUDA 依赖**，Mac 原生运行
- **Fast 特征策略**（与 Windows/Docker 完全一致）：
  - 肺叶：shape + firstorder + 纹理(GLCM/GLRLM/GLSZM) + LoG(σ=1.0)
  - 心肌：全特征 + Wavelet
  - `lung_vessels`：只算 firstorder（由 11 个 `Vessel_*` 高级特征替代慢速 shape）
  - 其余 ROI：shape + firstorder
- **黄金 16 靶区**：只计算 16 个心肺靶区掩膜（KEEP_FILES 过滤）
- **掩膜级多进程并行** + **单患者超时保护**（挂死患者自动跳过，整批不卡死）
- 四类 COPD 表型指标 + 11 个 Vessel/BV 高级特征

## 目录结构

```
mac_pyradiomics/
├── setup_mac.sh          # 一键安装（建环境 + 装依赖 + 验证）
├── requirements_mac.txt  # 依赖（全纯 CPU，TotalSegmentator 可选）
├── run_pipeline.py       # 主入口：分割 + Fast 特征提取（--radiomics-only 只算特征）
├── run_radiomics.py      # Fast 纯特征提取（复用已有掩膜）
├── run_radiomics_lite.py # Lite 纯特征提取（全 ROI shape+firstorder，更快）
├── radiomics_extract.py  # 特征核心（Fast/Lite 策略 + Vessel/BV + KEEP_FILES）
└── merge_radiomics.py    # 合并 JSON -> CSV
```

## 安装（在 Mac 上）

```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

## 运行

**① 完整流程**（TotalSeg 分割 + Fast 特征提取）：
```bash
conda activate mac-pyradiomics
python run_pipeline.py -i /path/CT -o /path/out --workers 2
```

**② 只做特征提取**（复用已有掩膜，Mac 上推荐，快）：
```bash
python run_pipeline.py -i /path/CT -o /path/out --radiomics-only --workers 2
```

**③ Fast 纯特征提取**（复用已有掩膜）：
```bash
python run_radiomics.py --nifti-dir /path/CT --seg-dir /path/seg --workers 2
```

**④ Lite 纯特征提取**（全 ROI shape+firstorder，更快）：
```bash
python run_radiomics_lite.py --nifti-dir /path/CT --seg-dir /path/seg --workers 2
```

**指定患者**（逗号分隔，按文件夹名精确匹配）：
```bash
python run_pipeline.py -i /path/CT -o /path/out --patients "患者A,患者B" --radiomics-only
```

### 可选参数（run_pipeline.py）
| 参数 | 说明 |
|------|------|
| `--radiomics-only` | 只做特征提取（复用已有掩膜） |
| `--seg-only` | 只分割，不做特征提取 |
| `--patients id1,id2` | 只处理指定患者 |
| `--workers N` | 掩膜并行进程数（Mac 建议 2~3，内存不足用 1） |
| `--timeout 秒` | 单患者超时（默认 2400s=40min；超时自动跳过继续） |
| `--force` | json 已存在也重算 |

> **Lite vs Fast**：Lite 用 `run_radiomics_lite.py`（所有掩膜 shape+firstorder，无 LoG/无 Wavelet/无纹理，更快）；
> Fast 用 `run_pipeline.py` / `run_radiomics.py`（肺叶 LoG+纹理、心肌 Wavelet）。两者特征列不同，**同一队列只能用一种模式**。

## 输出
```
out/ (或 seg-dir/)
├── <患者>_masks/                    # 16 靶区掩膜 + segmentation_info.json
├── <患者>_radiomics.json            # 每个患者的特征
└── radiomics_all_patients.csv       # 合并表（一行一患者）
```

## 特征组成
- **PyRadiomics**：16 掩膜（肺叶带 LoG+纹理、心肌带 Wavelet、血管 firstorder）
- **Vessel/BV 高级特征（11 个）**：Vessel_Fractal_Dim、Vessel_BV5/BV10_pct、
  Vessel_Skeleton_Voxels/Length、Branch/Junction/Endpoint_Count、
  Branching_Density、Tortuosity_Mean/Max
- **肺叶气肿**：5 肺叶 × (LAA-950% / Perc15 / 体积 / 占全肺比) + 全肺汇总
- **心肺共病**：PA/Ao 直径比 + 体积比、RV/LV 容积比、CAC 钙化体积
- **气道耦合**：气道总量 + 各肺叶气道占比
- **膈肌形态**：左右肺底轮廓填充比
