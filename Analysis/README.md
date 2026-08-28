# COPD 多序列影像组学 + AirQuant 气道特征分析

> 📖 研究说明/方法论/结果详见 **`../docs/README.md`**。

本目录存放 **2026-01 / 2026-02 / 2026-05** 三个慢阻肺(COPD) CT 序列的**跨序列外部验证**分析脚本。

## 0. 环境与数据

- **Python**: `C:\ProgramData\miniconda3\envs\copd-radiomics\python.exe`（Python 3.10；pyRadiomics/sklearn/scipy/pandas/matplotlib）
- **数据路径**（绝对路径，Windows）：
  - 特征表：
    - `E:\DICOM\2026-05-seg\2026-05-integrated_radiomics_aq.csv`（1106 行 × 4964 列）
    - `E:\DICOM\2026-02-seg\2026-02-integrated_radiomics_aq.csv`（383 行 × 4580 列）
    - `E:\DICOM\2026-01-seg\2026-01-integrated_radiomics_aq.csv`（100 行 × 2280 列，`integrate_2026_01.py` 生成）
  - 标签源（ICD 码）：
    - 2026-05：`E:\DICOM\2026-05\2026-5-9-overlap.xlsx`
    - 2026-02：`E:\DICOM\2026-02-seg\2026-2提取.xlsx`
    - 2026-01：`E:\DICOM\2026-01\2026-1标注信息.xlsx`
  - 数字 PatientID：各序列 `*-nifti\*\*_dicom_info.json` 的 `Series[].Patient.PatientID`
- **输出目录**：`E:\DICOM\reports\`（HTML/MD 报告 + `figs\`）、`E:\DICOM\2026-0x-seg\`（中间表/log）
- **join 键**：一律用 `Patient_ID`（长串目录名）；**不要**用 `PatientID`（临床号与 DICOM 号是两套体系）

## 1. 处理流程（按序执行）

```
① 标签一致化        build_bcos_labels_consistent.py      → labels_ae_bcos_2026_05/02.csv
② 2026-01 整合      integrate_2026_01.py                 → 2026-01-integrated_radiomics_aq.csv
③ PatientID 转数字  convert_pid_to_numeric_v2.py <02|05>  → *_numid.csv + <tag>_pid_map.csv
④ 规约到胸部标准    clean_to_2026_01_standard.py         → *_aligned01.csv（剔除肾上腺/肾/椎体等非胸部器官）
⑤ 补标签列          add_labels_to_aligned.py             → *_aligned01_labeled.csv
⑥ 三队列外部验证    run_3cohort_aligned.py               → 外验 AUC + 临床单变量 AUC
⑦ 补充分析/报告     run_aq_only / run_fusion_aqtopk / run_fusion_boot_ci / run_bootstrap_balance /
                    run_topk_generalization / missing_stats / run_report_balanced / run_final_report
```

## 2. 脚本清单

| 脚本 | 用法 | 输入 | 输出 |
|---|---|---|---|
| `build_bcos_labels_consistent.py` | `python build_bcos_labels_consistent.py` | 三序列 xlsx/overlap | `labels_ae_bcos_*_csv`（含 `ICD/AECOPD/COPD_BCOS`） |
| `integrate_2026_01.py` | `python integrate_2026_01.py` | 2026-01 radiomics JSON + airway CSV + 标签 | `2026-01-integrated_radiomics_aq.csv` |
| `convert_pid_to_numeric_v2.py` | `python convert_pid_to_numeric_v2.py <02\|05>` | integrated CSV + nifti json | `*_numid.csv`、`<tag>_pid_map.csv` |
| `clean_to_2026_01_standard.py` | `python clean_to_2026_01_standard.py` | 02/05 integrated | `*_aligned01.csv`（2273 胸部特征） |
| `add_labels_to_aligned.py` | `python add_labels_to_aligned.py` | aligned01 + labels | `*_aligned01_labeled.csv` |
| `run_3cohort_aligned.py` | `python run_3cohort_aligned.py` | 三份 aligned01_labeled | `report_aligned3cohort.html/.md`、`clinical_uni_aligned.csv` |
| `run_final_report.py` | `python run_final_report.py` | 各结果 CSV | `report_COPD_final.html/.md`（综合报告）+ 特征清单 |
| `run_aq_only.py` | `python run_aq_only.py` | 三队列 + labels | `aq_only.log`（aq 单独评估） |
| `run_fusion_aqtopk.py` | `python run_fusion_aqtopk.py` | 三队列 + labels | `report_aq_fusion.html/.md`、`aq_fusion_results.csv` |
| `run_fusion_boot_ci.py` | `python run_fusion_boot_ci.py` | 融合模型 | `fusion_boot_ci.csv` + 森林图 |
| `run_bootstrap_balance.py` | `python run_bootstrap_balance.py` | 三队列 | `bootstrap_balance_results.csv` + 森林图 |
| `run_topk_generalization.py` | `python run_topk_generalization.py` | 三队列 | `topk_gen.log`（TopK 外验） |
| `missing_stats.py` | `python missing_stats.py` | 三特征表 | `missing_stats.log`、`missing_patients.csv` |
| `run_report_balanced.py` | `python run_report_balanced.py` | 各结果 | `report_balanced_3cohort.html/.md` |

### 2.1 补充/早期脚本（同一项目）

| 脚本 | 说明 |
|---|---|
| `run_bcos_validate_consistent.py` | 一致 ICD 标签下 2026-05→02 外部验证（早期版，被 `run_3cohort_aligned.py` 取代） |
| `run_bcos_validate_j44.py` | COPD_BCOS 任务仅限 J44 的 CV+外验 |
| `run_3cohort_validate.py` | 早期三队列外验（被 `run_3cohort_aligned.py` 取代） |
| `run_bcos_study_2026_05(_v2).py` / `run_bcos_finish.py` | COPD+BCOS 特征筛选研究（Top100、CV 0.698 来源） |
| `rerun_2026_05_internal.py` / `rerun_2026_05_bcos_task.py` | 2026-05 内部任务重现（验证旧效果是否还在） |

## 3. 标签口径（一致）

| 标签 | 定义 | 备注 |
|---|---|---|
| `AECOPD` | 主要诊断-ICD码 前缀 `J44.1*`(急性加重) 或 `J44.0*`(急性下呼吸道感染) → 1；`J44.9*/J44.8*` → 0 | 仅 J44(COPD) 有效；J47(支扩) 为 NaN |
| `COPD_BCOS` | 医生标注 `COPD合并支扩==1` → 1，否则 0 | 三序列同口径，但患病率差异大(6%) |
| `J44.0 vs J44.9` | `J44.0` → 1 vs `J44.9` → 0（最均衡的候选任务） | 仅 J44 内 |

## 4. 关键结论（截至 2026-08-29）

1. **跨序列泛化总体偏弱**（多数外验 AUC 0.5–0.6）；`J44.0 vs J44.9 → 2026-02` 外验 0.624，95%CI **[0.510,0.720] 下界 >0.5**，是唯一统计显著的任务。
2. **aq(AirQuant) 特征可用且与 rad 互补**：COPD_BCOS 上 `radTop100+aqTop20` 融合外验 02=0.576 / 01=0.677，优于 rad 全量。
3. **单临床影像特征分辨力**：COPD_BCOS 上 `LAA950`(肺气肿 0.64)、`TD_fwhm`(0.64)、`Vessel_Junction`(0.64)、`WA_pct`(0.64) 单个即可达 AUC≈0.64。
4. **2026-02/05 原表扩展器官特征(肾上腺/肾/椎体/肋骨等)缺失率高**，已按 2026-01 胸部标准规约（2273 密集特征）。
5. 建议补充稳定期(J44.9)与支扩阳性样本缓解不平衡。

## 5. 注意

- 本机有"长时程重 DataFrame 操作会被静默 kill"的问题 → 大表读写建议用分块(chunksize)；避免逐特征 scipy 循环（用向量化 rank AUC 或 sklearn）。
- 2026-05 的 AECOPD 标签只在 J44 内定义（758 例 J47 为 NaN），建模前先过滤。
- 报告输出到 `E:\DICOM\reports\`（HTML 自包含，图已 base64 内嵌）。
