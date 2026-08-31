# -*- coding: utf-8 -*-
"""
申报清单补算特征 —— 保护区特征工程降维（共享实现）
================================================
把 9 个原始声明特征压缩成 5 个临床域 z-score 复合分，作为保护区进入建模，
解决「一次性强制塞入 9 个原始特征 → 冗余(CAC 对) + 噪声(FAI/EpiFat_Mean_HU)
稀释线性 SVM 判别信号」的问题。

复合分 = 等权 z-score 和（固定权重，非数据驱动 → 无特征选择泄露）：
  PAH_vascular   = z(Vessel_CSA_mean_mm2) + z(PA_Equivalent_Diameter_mm)
  CAC_score      = z(CAC_Agatston)                 # Mass 与 Agatston r≈0.99，去掉
  FatInflam_score= z(EpiFat_Volume_mm3) + z(FAI_pericoronary_HU) + z(EpiFat_Mean_HU)
  CTR_score      = z(CardioThoracic_Ratio)
  BAR_score      = z(BronchoArtery_Ratio)

保护区（强制进模型）= 5 复合分 + Pi10 + Vessel_BV5_pct + Vessel_BV10_pct = 8 项
（原始 9 项从特征池移除，避免重复计数）。

z-score 用全队列 mean/std（固定单调变换，特征构造阶段的常规预处理）；
建模管线内仍有 per-fold StandardScaler/RobustScaler 做正式标准化。
"""
import numpy as np
import pandas as pd

RAW_DECLARED = [
    "CAC_Agatston", "CAC_Mass_mg",
    "EpiFat_Volume_mm3", "EpiFat_Mean_HU", "FAI_pericoronary_HU",
    "CardioThoracic_Ratio", "BronchoArtery_Ratio",
    "Vessel_CSA_mean_mm2", "PA_Equivalent_Diameter_mm",
]

# 复合分名 -> 组成（原始声明特征）
ENG_SPEC = {
    "PAH_vascular":    ["Vessel_CSA_mean_mm2", "PA_Equivalent_Diameter_mm"],
    "CAC_score":       ["CAC_Agatston"],
    "FatInflam_score": ["EpiFat_Volume_mm3", "FAI_pericoronary_HU", "EpiFat_Mean_HU"],
    "CTR_score":       ["CardioThoracic_Ratio"],
    "BAR_score":       ["BronchoArtery_Ratio"],
}

# 工程化后的保护区（8 项）
PROTECTED_ENG = ["PAH_vascular", "CAC_score", "FatInflam_score", "CTR_score",
                 "BAR_score", "Pi10", "Vessel_BV5_pct", "Vessel_BV10_pct"]


def build_eng(work):
    """在 work(DataFrame) 上就地新增 z 列与复合分列；返回 work。"""
    for raw in RAW_DECLARED:
        x = pd.to_numeric(work[raw], errors="coerce")
        m = x.mean()
        s = x.std()
        work[raw + "_z"] = (x - m) / s if s and np.isfinite(s) and s > 0 else x * np.nan
    for name, raws in ENG_SPEC.items():
        cols = [r + "_z" for r in raws]
        work[name] = work[cols].sum(axis=1, min_count=1)
    return work


def engineered_feature_names(raw_feature_names):
    """从原始特征名列表移除被复合分替代的 9 项，加入复合分列名。"""
    return [f for f in raw_feature_names if f not in RAW_DECLARED] + list(ENG_SPEC.keys())
