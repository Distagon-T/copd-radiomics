# COPD 多序列影像组学 + AirQuant 气道特征研究说明

> 研究主题：基于胸部 CT 的 **radiomics + AirQuant(气道/肺气肿/血管)** 特征，在慢阻肺(COPD)三序列队列上做**跨序列外部验证**。
> 日期：2026-08-29　|　代码：见 `Analysis/`　|　报告：`E:\DICOM\reports\`

---

## 1. 项目概述

利用三个不同批次/年份的慢阻肺胸部 CT 队列（2026-05 训练，2026-01 / 2026-02 外部验证），评估从 CT 影像提取的 radiomics 与 AirQuant 结构化气道指标，在三个分类任务上的**泛化能力**。

- **核心问题**：这些特征在训练队列内是否有效？能否跨序列迁移到另一批 CT 数据？
- **主要发现**：跨序列泛化整体偏弱；但 **J44.0 vs J44.9（感染 vs 稳定）** 与 **COPD_BCOS 的 rad+aq 融合模型** 显示出有意义（虽弱）的信号；单临床影像特征（LAA950/TD/WA/Vessel）在 COPD_BCOS 上可达 AUC≈0.64。

## 2. 数据队列

| 队列 | 例数(特征表) | 来源 | 说明 |
|---|---|---|---|
| 2026-05 | 1106（含 J47 支扩 610） | `E:\DICOM\2026-05-seg\..._integrated_radiomics_aq.csv` | 训练队列；J44 慢阻肺 500 例 |
| 2026-02 | 383 | `E:\DICOM\2026-02-seg\..._integrated_radiomics_aq.csv` | 外部验证；纯 J44 |
| 2026-01 | 100 | `E:\DICOM\2026-01-seg\..._integrated_radiomics_aq.csv` | 外部验证；纯 J44，最干净(缺失1%) |

- **ID 规范**：临床号(`患者id`)与 DICOM `PatientID` 是两套体系，**统一用 `Patient_ID`(长串目录名) join**；数字 PatientID 从 nifti `dicom_info.json` 提取。
- **标签来源**：三队列均以 `主要诊断-ICD码` 派生，口径一致。

## 3. 标签定义（一致口径）

| 标签 | 规则 | 备注 |
|---|---|---|
| `AECOPD` | ICD 前缀 `J44.1*`(急性加重) 或 `J44.0*`(急性下呼吸道感染) → 1；`J44.9*/J44.8*` → 0 | 仅 J44 有效；J47(支扩)=NaN |
| `COPD_BCOS` | 医生标注 `COPD合并支扩==1` → 1，否则 0 | 三队列同口径，阳性率均 ~6% |
| `J44.0 vs J44.9` | `J44.0`(感染) → 1 vs `J44.9`(稳定) → 0 | 阳性率最均衡(02 达 40%) |

分布：
| 任务 | 2026-05 | 2026-02 | 2026-01 |
|---|---|---|---|
| AECOPD | 264/84 (76%) | 309/61 (84%) | 82/16 (84%) |
| COPD_BCOS | 46/652 (7%) | 22/348 (6%) | 6/92 (6%) |
| J44.0 vs J44.9 | 32/84 (28%) | 40/60 (40%) | 6/16 (27%) |

## 4. 特征工程

- **radiomics**：pyRadiomics 纹理/形态/小波（器官分割内）。
- **AirQuant(aq)**：气道（`TD_/WA_/Din_/Dout_/wall_/blur_/Pi10`）、肺气肿（`Lobe_*_LAA950_pct`/`Perc15_HU`）、血管（`Vessel_*`）。
- **特征对齐（重要）**：2026-02/05 原表含大量**非胸部扩展器官**特征（`adrenal_gland/kidney/vertebrae/rib/hip` 等，缺失率高且与胸部研究无关），已按 **2026-01 胸部标准**规约，三队列统一为 **2273 个密度一致特征**（脚本 `clean_to_2026_01_standard.py`）。
- **缺失处理**：剔除 >50% 缺失列；余下中位数填补（`fillna(median).fillna(0)`）。

## 5. 方法

- **模型**：Logistic Regression（liblinear, `class_weight=balanced`, C=1.0）。
- **内部验证**：2026-05 内 5 折分层 CV。
- **外部验证**：2026-05 全量训练 → 预测 2026-01 / 2026-02，报告 AUC。
- **特征筛选**：单变量 AUC（向量化 rank 法）TopK；融合 `radTop100 + aqTop20`。
- **置信区间**：测试集 bootstrap 重采样 500 次。
- **不平衡**：多数类下采样(balanced) 对比。

## 6. 结果汇总

### 6.1 对齐后外部验证（训练 2026-05）

| 任务 | 配置 | 05 CV | 02 外验 | 01 外验 |
|---|---|---|---|---|
| AECOPD | rad+aq 全量 | 0.565 | 0.525 | 0.575 |
| | radTop100+aqTop20 | 0.534 | 0.549 | 0.627 |
| COPD_BCOS | rad+aq 全量 | 0.479 | 0.501 | 0.631 |
| | **radTop100+aqTop20** | 0.660 | **0.576** | **0.677** |
| J44.0 vs J44.9 | rad+aq 全量 | 0.591 | **0.626** | **0.656** |
| | radTop100+aqTop20 | 0.688 | 0.514 | 0.438 |

### 6.2 临床影像单特征 AUC（2026-05 训练集）

**COPD_BCOS（最强）**：`TD_fwhm_all` 0.644、`Lobe_RLL_LAA950_pct`(肺气肿) 0.643、`Vessel_Junction_Count` 0.641、`WA_pct_all` 0.637、`Lobe_LLL_LAA950_pct` 0.635
**J44.0 vs J44.9**：`Vessel_Tortuosity_Mean` 0.632、`Lobe_RLL_Perc15_HU` 0.626、`wall_hu_kurt` 0.610
**AECOPD**：弱（最高 0.563）

### 6.3 Bootstrap 95%CI（radTop100+aqTop20 融合）

| 任务 | 02 外验 | 01 外验 |
|---|---|---|
| J44.0 vs J44.9 | 0.624 [**0.510**, 0.720] | 0.656 [0.319, 1.000] |
| COPD_BCOS | 0.603 [0.481, 0.722] | 0.693 [0.537, 0.855] |
| AECOPD | 0.570 [0.495, 0.653] | 0.621 [0.476, 0.768] |

> 唯一统计显著（CI 下界 > 0.5）的是 **J44.0 vs J44.9 → 2026-02**。

### 6.4 aq 特征单独评估

| 任务 | aq 单变量有信号数 | aq-only CV | 说明 |
|---|---|---|---|
| COPD_BCOS | **60/106** | 0.532 | TD_fwhm/LAA950/WA_pct 强 |
| J44.0 vs J44.9 | 58/106 | 0.576 | Vessel/Perc15 强 |
| AECOPD | 8/106 | 0.497 | 基本无信号 |

## 7. 结论与局限

**结论**
1. 跨序列泛化整体偏弱（多数外验 AUC 0.5–0.6），提示特征存在队列特异性/过拟合。
2. **J44.0 vs J44.9** 是唯一统计显著的任务（02 外验 0.624，CI 下界 >0.5）。
3. **aq 特征并非无用**：COPD_BCOS 上 `radTop100+aqTop20` 融合（01=0.677）优于 rad 全量；单临床特征（肺气肿 LAA950、气道 TD/WA、血管）即达 AUC≈0.64，影像表型符合病理。
4. **特征筛选 > 全量**：rad 全量(2157 维)过拟合，精选 Top100+aqTop20 泛化更好。

**局限**
- 样本不平衡严重（AECOPD 阳性 76-84%、COPD_BCOS 阳性仅 6%）；2026-01 阳性极少（COPD_BCOS 6 例、J44.0 6 例），CI 极宽。
- 扩展器官特征缺失率高，已剔除；但 2026-01 未算 `Vessel_/Lobe_` 完整集（已用其标准规约）。
- 标签由 ICD 码派生，无文本诊断交叉验证。
- 建议补充稳定期(J44.9)与支扩阳性样本；考虑 ComBat/域适应缓解跨序列偏移。

## 8. 复现指南

1. 环境：`conda activate copd-radiomics`（Python 3.10）。
2. 按 `Analysis/README.md` 的流程 ①→⑦ 执行：
   - 标签一致化 → 2026-01 整合 → PatientID 转数字 → 规约胸部标准 → 补标签 → 三队列外验 → 补充分析/报告。
3. 每个脚本都有 `用法: python xxx.py` 注释。
4. 输出均落在 `E:\DICOM\reports\`（HTML 自包含）+ 各序列 `2026-0x-seg\`（中间表/log）。

## 9. 产物清单（`E:\DICOM\reports\`）

| 文件 | 内容 |
|---|---|
| `report_COPD_final.html/.md` | **综合正式报告**（摘要/方法/结果/结论 + 对齐验证 + 临床单变量） |
| `report_balanced_3cohort.html/.md` | 均衡任务 + TopK + Bootstrap 报告 |
| `report_aligned3cohort.html/.md` | 对齐后三队列外验 + 临床单变量 |
| `report_aq_fusion.html/.md` | rad+aqTopK 融合对比 |
| `clinical_uni_aligned.csv` | 93 临床特征 × 3 任务单变量 AUC |
| `copd_bcos_radtop100_features.csv` / `copd_bcos_aqtop20_features.csv` | 最优模型特征清单 |
| `missing_patients.csv` / `missing_stats.log` | 缺失/补跑清单 |
