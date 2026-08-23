# COPD 影像表型分析管线（d:\copd-radiomics）

COPD / 支气管扩张（BCOS）/ 咯血 影像表型研究：从 CT 分割 → Radiomics / AirQuant 气道特征 /
肺血管高级特征 → 建模判别 → 报告。

---

## 1. 环境

- Python: conda 环境 `copd-radiomics`（Python 3.10）
  - `C:\ProgramData\miniconda3\envs\copd-radiomics\python.exe`
  - 依赖: pyRadiomics 3.0.1, SimpleITK, scipy, scikit-image, edt, sklearn, pandas, numpy, matplotlib
- MATLAB: R2026a（AirQuant 气道量化用）`D:\MATLAB\R2026a\bin\matlab.exe`

> Windows 下 `conda activate` 可能失败，直接用全路径：`& 'C:\ProgramData\miniconda3\envs\copd-radiomics\python.exe' script.py`

## 2. 数据位置（2026-05 队列）

| 内容 | 路径 |
|---|---|
| 原始 CT (NIfTI) | `E:\DICOM\2026-05-nifti\<患者>\` |
| 分割掩膜（16 靶区） | `E:\DICOM\2026-05-seg\<患者>_masks\` |
| 单患者 radiomics json | `E:\DICOM\2026-05-seg\<患者>_radiomics.json` |
| 合并 radiomics CSV（698 例） | `E:\DICOM\2026-05-seg\radiomics_2026_05_features.csv` |
| **+11 个肺血管特征** | `E:\DICOM\2026-05-seg\radiomics_2026_05_features_vessel.csv` |
| 全部 nifti 患者的血管特征 | `E:\DICOM\2026-05-seg\vessel_feats_2026_05_all.csv` |
| AirQuant 气道指标 | `E:\DICOM\2026-05-Airway_metrics_tmp\<患者>_airquant\` |
| AirQuant 特征 CSV | `E:\DICOM\2026-05-Airway_features\airway_features_all.csv` |
| AirQuant 聚合 | `E:\DICOM\2026-05-seg\airquant_2026_05_aggregated.csv` |
| 临床/标签 | `E:\DICOM\2026-05\info-2026-05.csv`、`E:\DICOM\2026-05\2026-5-9-overlap.xlsx` |
| 报告 | `E:\DICOM\2026-05-seg\report_*.md/.html` + `figs\` |

## 3. 管线总览

```
CT (2026-05-nifti, 1106例)
   │  batch_segment_largest_slice.py / batch_segment*.py（TotalSegmentator 3 引擎 → 16 掩膜）
   ▼
16 掩膜 (2026-05-seg\<患者>_masks)
   │  compute_patient_radiomics_{full,fast,lite}.py  → <患者>_radiomics.json
   │  + 四类 COPD 表型指标（肺叶气肿/心肺血管/气道耦合/膈肌）
   │  + 肺血管高级特征 Vessel_*（见 §5）
   ▼
radiomics CSV  +  AirQuant 气道特征（MATLAB，见 §4）
   │  train_fusion_model.py / run_*_2026_05.py（分层 5 折 CV LR）
   ▼
报告 report_*.md/.html（含 ROC/单变量/一致性/bootstrap 精简模型）
```

## 4. AirQuant 气道特征（MATLAB）

- 主脚本：`AirQuant\compute_airway_features.m`（R2026a）
  - 配置 `METRICS_DIR=E:\DICOM\2026-05-Airway_metrics_tmp`、`FEATURES_DIR=E:\DICOM\2026-05-Airway_features`
  - 支持 `AQ_MAX_PATIENTS` 环境变量限量
- 特征：形态(WA%/T-D/内径/外径/壁厚) + 拓扑 + **T/D 急剧变化**(TD_ratio_std/CV/slope/outlier) +
  **FWHM 边界模糊**(blur_*) + **FWHM 法 T/D**(TD_fwhm_*) + PCA 异质性
- 合并到 Python：`merge_airway_features_2026_05.py`（新列前缀 `aq_`）
- 全量运行日志：`E:\DICOM\2026-05-Airway_features\airway_features_run.log`（1106 例，~18-37h）

## 5. 肺血管高级特征（Vessel_*，替代慢速 shape）

> 原理：`lung_vessels` 的 pyRadiomics shape（Maximum3DDiameter/SurfaceArea 等）在百万体素上耗时 ~35min
> 且无临床特异性，故只算 firstorder，改由以下极速、可解释的特征替代。

| 特征 | 含义 |
|---|---|
| `Vessel_Fractal_Dim` | 3D 计盒分形维度（血管网复杂度；COPD pruning 下降 / 增生上升） |
| `Vessel_BV5_pct` / `Vessel_BV10_pct` | 截面积 <5/10mm² 小血管血容量占比（EDT 半径阈值 1.26/1.78mm） |
| `Vessel_Skeleton_Voxels` / `_Length_mm` | 中心线体素数 / 长度 |
| `Vessel_Branch_Count` / `Junction_Count` / `Endpoint_Count` | 分支数 / 分叉点数 / 端点数 |
| `Vessel_Branching_Density_per_mm` | 分支点密度 |
| `Vessel_Tortuosity_Mean` / `_Max` | 迂曲度（弧长/直线距离） |

- 实现：`compute_patient_radiomics_{fast,lite}.py` 内 `vessel_advanced_features()`
  （skeletonize + 26 邻域计数 + 去分叉点分段 + PCA 端点求迂曲度；BV5 用 `edt` 包 float32 省内存）
- 批量计算：`compute_2026_05_vessel_features.py`（多进程 + 增量缓存 `vessel_feats_2026_05.json`）
- 分类能力评估：`eval_vessel_features_2026_05.py`（单变量 AUC/d/p）
- 加入建模对比：`run_vessel_boost_2026_05.py`（top8 ± 血管特征 bootstrap 对比）

**关键结果**：`Vessel_Branching_Density_per_mm` 是 BCOS 表型最强单特征（AUC 0.673, p=0.0002）；
血管特征为咯血/BCOS 任务提供稳定化 bootstrap 增益。

## 6. 建模 / 标签任务

| 脚本 | 任务 / 标签 |
|---|---|
| `run_nsfc_2026_05.py` | NSFC 急慢（关键词） |
| `run_copd_acute_2026_05.py` | 纯 COPD 急性加重(308) vs 单纯慢阻肺(118) |
| `run_bcos_2026_05.py` | BCOS 队列 AECOPD vs SCOPD |
| `run_bcos_phenotype_2026_05.py` | BCOS 表型：医生标注 COPD合并支扩=1 vs PureCOPD |
| `run_bronch_hemoptysis_2026_05.py` | 支扩咯血 vs 无咯血（610 例支扩） |
| `run_copd_ae_cause_2026_05.py` | 急性 COPD 感染型 vs 非感染型 |
| `run_{...}_topmodel_2026_05.py` | Top 显著特征精简模型 + bootstrap |

统一建模：`train_fusion_model.py`（`load_and_join`/`select_features`/`univariate_summary`）。
统一报告图：`plot_consistency_2026_05.py`（bootstrap 一致性森林图）。

## 7. 关键坑（经验）

- **skimage 0.25 已移除 `skeletonize_3d`** → 用 `skeletonize`（3D 自动分派）。
- **scipy `ndi.convolve` 3D 会分配 `(3,Nz,Ny,Nx)` 缓冲** → 大容积 OOM；26 邻域计数改用逐偏移求和 + `*skel`。
- **内存大户**：EDT 用 `edt` 包（float32）而非 scipy float64；分形 boxcount 用 int32。
- **标签泄漏**：`select_features` 的 label 排除必须大小写不敏感（否则 uppercase 标签列漏入特征 → AUC=1.0 假象）。
- **`_to_jsonable`**：nibabel/radiomics 的 numpy 类型要先转 Python 原生再 json.dump。
- 报告融合脚本用 str.replace 前先数出现次数，避免重复插入（曾出现总结章节插两次）。

## 8. 备注

- 报告所用 radiomics = **Lite 特征集**（shape + firstorder + 自定义表型，无纹理/滤波）。
  如需全量纹理/滤波特征，用 `compute_patient_radiomics_fast.py` 或 `docker-radiomics-full`。
- Mac 打包版 `mac_pyradiomics` 与 Lite 同策略。
- 2026-07 测试集存在 lite/full json 混合（3002 列），合并时注意特征集版本一致。
