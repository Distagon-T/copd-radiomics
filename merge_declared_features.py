# -*- coding: utf-8 -*-
"""
把 compute_declared_features.py 补算的特征合并进建模表
=====================================================
输入：E:\\DICOM\\results\\declared_features_computed.csv
目标：合并进（就地更新，自动 .bak 备份，只新增不存在的列）
  - E:\\DICOM\\results\\patients_feature_label.csv
  - E:\\DICOM\\results\\ordinal_risk_all_patients_feature_label.csv（若存在）
join 键：PatientID（数字串，去掉尾部 .0 与前导 0，避免两套口径不一致）

用法：
  python merge_declared_features.py
说明：合并后，下游建模（lasso_svm_binary / lasso_svm_nested 等）通过 is_feature()
      白名单自动纳入这些新列（EpiFat_/FAI_/Aorta_/BronchoArtery_/CardioThoracic_ 等前缀已加入）。
"""
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULTS = Path(r"E:\DICOM\results")
DECLARED = RESULTS / "declared_features_computed.csv"
TARGETS = [
    RESULTS / "patients_feature_label.csv",
    RESULTS / "ordinal_risk_all_patients_feature_label.csv",
]
# 合并时要排除的非特征列
NON_FEAT = {"PatientID", "Patient_ID", "cohort", "label", "Label", "_err", "info_match"}
# 新特征列（来自 compute_declared_features.py）
NEW_FEAT_COLS = [
    "Vessel_Volume_mm3", "Vessel_CSA_mean_mm2",
    "PA_Equivalent_Diameter_mm", "BronchoArtery_Ratio",
    "CAC_Agatston", "CAC_Mass_mg", "CAC_Volume_mm3",
    "EpiFat_Volume_mm3", "EpiFat_Mean_HU", "FAI_pericoronary_HU",
    "Aorta_Outer_Mean_Diameter_mm", "Aorta_Wall_Fraction",
    "Aorta_Wall_Thickness_mm_approx", "CardioThoracic_Ratio",
]


def norm_pid(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = re.sub(r"\.0+$", "", s)
    s = s.lstrip("0") or "0"
    return s


def find_pid_col(df):
    for c in df.columns:
        if str(c).lower() in ("patientid", "patient_id", "patient id", "pid"):
            return c
    return None


def main():
    if not DECLARED.exists():
        print(f"[ERR] 未找到 {DECLARED}，请先运行 compute_declared_features.py")
        return
    d = pd.read_csv(DECLARED, low_memory=False, dtype={"PatientID": str})
    d = d[d["_err"].astype(str) == ""].copy() if "_err" in d.columns else d.copy()
    d["_pid"] = d["PatientID"].map(norm_pid)
    print(f"补算表: {len(d)} 例，新特征列 {len(NEW_FEAT_COLS)} 个")

    for tgt in TARGETS:
        if not tgt.exists():
            print(f"[skip] 目标不存在: {tgt.name}")
            continue
        t = pd.read_csv(tgt, low_memory=False)
        pidc = find_pid_col(t)
        if pidc is None:
            print(f"[skip] {tgt.name}: 找不到 PatientID 列")
            continue
        t["_pid"] = t[pidc].map(norm_pid)
        feats = [c for c in NEW_FEAT_COLS if c not in t.columns]
        if not feats:
            print(f"[skip] {tgt.name}: 新特征列已全部存在")
            continue
        # 合并（只取补算表中的新列）
        add = d[["_pid"] + feats].dropna(subset=["_pid"])
        merged = t.merge(add, on="_pid", how="left")
        n_match = int(merged[feats[0]].notna().sum()) if len(merged) else 0
        # 备份
        bak = tgt.with_suffix(tgt.suffix + ".bak")
        shutil.copy2(tgt, bak)
        merged = merged.drop(columns=["_pid"])
        merged.to_csv(tgt, index=False, encoding="utf-8-sig")
        print(f"[OK] {tgt.name}: 行 {len(t)} -> {len(merged)}，新增列 {feats}，匹配 {n_match} 例（备份 {bak.name}）")


if __name__ == "__main__":
    main()
