# -*- coding: utf-8 -*-
"""
气道重塑定量特征聚合（Python 版，逻辑与 compute_airway_features.m 一致）
=====================================================================
从 batch_airway_quant.m 生成的 _full_metrics.csv（AirQuant 拓扑+几何）聚合三大类特征：
  一、形态学分级：各代 WA% 均值、T/D (WT/OD)、内径/外径、分支数、最大代数
  二、拓扑网络：分叉角(parent/sibling)、终端分支数/修剪率、迂曲度
  三、预留：管壁密度/纹理（需重读 CT 做 FWHM，见 MATLAB 脚本 compute_airway_features.m）

输出：
  - 每患者一行: <输出>/<患者>_airway_features.csv
  - 汇总表:     <输出>/airway_features_all.csv
用法：
  python compute_airway_features.py
"""
import os
import glob
import json
import numpy as np
import pandas as pd

# =========================================================================
# 路径配置
# =========================================================================
METRICS_DIR = r"E:\DICOM\2026-04-Airway_metrics"   # batch_airway_quant 输出
FEATURES_DIR = r"E:\DICOM\2026-04-Airway_features"  # 本脚本输出
os.makedirs(FEATURES_DIR, exist_ok=True)

GEN_TARGETS = [3, 4, 5]


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(x.mean()) if len(x) else np.nan


def main():
    csv_files = sorted(
        glob.glob(os.path.join(METRICS_DIR, "*_airquant", "*_full_metrics.csv"))
    )
    print(f"找到 {len(csv_files)} 个患者定量结果 CSV")
    if not csv_files:
        print("未找到任何 _full_metrics.csv，请检查 METRICS_DIR。")
        return

    rows = []
    for csv_file in csv_files:
        patient_dir = os.path.dirname(csv_file)
        patient = os.path.basename(csv_file).replace("_full_metrics.csv", "")

        T = pd.read_csv(csv_file)
        if T.empty:
            continue

        # 读取 info JSON（Pi10）
        Pi10 = np.nan
        info_file = os.path.join(patient_dir, f"{patient}_airquant_info.json")
        if os.path.exists(info_file):
            with open(info_file, "r", encoding="utf-8") as f:
                info = json.load(f)
            Pi10 = info.get("Pi10", np.nan)

        gen = T["generation"].to_numpy(dtype=float)
        wa = T["WA_pct"].to_numpy(dtype=float)
        wt = T["Wall_Thickness_mm"].to_numpy(dtype=float)
        od = T["Outer_Diameter_mm"].to_numpy(dtype=float)
        din = T["Inner_Diameter_mm"].to_numpy(dtype=float)
        la = T["LumenArea_mm2"].to_numpy(dtype=float)
        waa = T["WallArea_mm2"].to_numpy(dtype=float)
        pi = T["Pi_Perimeter_mm"].to_numpy(dtype=float)
        tort = T["stats_tortuosity"].to_numpy(dtype=float)
        pdeg = T["stats_parent_deg"].to_numpy(dtype=float)
        sdeg = T["stats_sibling_deg"].to_numpy(dtype=float)

        # 终端分支（无子代）
        has_child = np.zeros(len(T), dtype=bool)
        if "children_1" in T.columns:
            has_child |= ~T["children_1"].isna().to_numpy()
        if "children_2" in T.columns:
            has_child |= ~T["children_2"].isna().to_numpy()
        terminal = ~has_child

        tdr = wt / od  # T/D ratio

        f = {
            "patient_folder": patient,
            "Pi10": Pi10,
            # ---- 一、形态学分级 ----
            "n_branches": len(T),
            "Din_mean_all": safe_mean(din),
            "Din_mean_gen3": safe_mean(din[gen == 3]),
            "Din_mean_gen4": safe_mean(din[gen == 4]),
            "Din_mean_gen5": safe_mean(din[gen == 5]),
            "Dout_mean_all": safe_mean(od),
            "WA_pct_gen3": safe_mean(wa[gen == 3]),
            "WA_pct_gen4": safe_mean(wa[gen == 4]),
            "WA_pct_gen5": safe_mean(wa[gen == 5]),
            "WA_pct_gen3to6": safe_mean(wa[(gen >= 3) & (gen <= 6)]),
            "WA_pct_all": safe_mean(wa),
            "TD_ratio_all": safe_mean(tdr),
            "TD_ratio_gen3": safe_mean(tdr[gen == 3]),
            "TD_ratio_gen4": safe_mean(tdr[gen == 4]),
            "TD_ratio_gen5": safe_mean(tdr[gen == 5]),
            "LA_mean_all": safe_mean(la),
            "WA_mean_all": safe_mean(waa),
            "Pi_mean_all": safe_mean(pi),
            # ---- 二、拓扑网络 ----
            "max_generation": int(np.nanmax(gen)) if len(gen) else np.nan,
            "mean_tortuosity": safe_mean(tort),
            "std_tortuosity": float(np.nanstd(tort)) if np.any(~np.isnan(tort)) else np.nan,
            "mean_parent_angle": safe_mean(pdeg),
            "mean_sibling_angle": safe_mean(sdeg),
            "mean_parent_angle_gen3": safe_mean(pdeg[gen == 3]),
            "mean_parent_angle_gen4": safe_mean(pdeg[gen == 4]),
            "n_terminal_total": int(np.sum(terminal)),
            "n_terminal_gen5plus": int(np.sum((gen >= 5) & terminal)),
            "n_terminal_gen6plus": int(np.sum((gen >= 6) & terminal)),
            "pruning_ratio_gen5": float(np.sum((gen >= 5) & terminal) / max(1, len(T))),
            "pruning_ratio_gen6": float(np.sum((gen >= 6) & terminal) / max(1, len(T))),
            "mean_WA_pct_terminal": safe_mean(wa[terminal]),
            # ---- 三、管壁密度（预留，MATLAB 脚本计算） ----
            "wall_hu_mean": np.nan, "wall_hu_std": np.nan,
            "wall_hu_skew": np.nan, "wall_hu_kurt": np.nan,
            "wall_hu_mean_gen3": np.nan, "wall_hu_mean_gen4": np.nan,
            "wall_hu_mean_gen5": np.nan,
            "pca_explained_1": np.nan, "pca_explained_2": np.nan,
            "pca_explained_3": np.nan, "pca_first_pc_std": np.nan,
        }
        rows.append(f)

    df = pd.DataFrame(rows)
    # 每患者单独文件
    for _, r in df.iterrows():
        patient = r["patient_folder"]
        r.to_frame().T.to_csv(
            os.path.join(FEATURES_DIR, f"{patient}_airway_features.csv"), index=False
        )
    # 汇总
    summary_path = os.path.join(FEATURES_DIR, "airway_features_all.csv")
    df.to_csv(summary_path, index=False)
    print(f"\n完成！共 {len(df)} 个患者。")
    print(f"汇总: {summary_path}")
    print("\n特征表预览：")
    show_cols = ["patient_folder", "n_branches", "WA_pct_gen3", "WA_pct_gen4",
                 "WA_pct_gen5", "TD_ratio_all", "mean_tortuosity",
                 "mean_parent_angle", "mean_sibling_angle", "n_terminal_total",
                 "max_generation", "Pi10"]
    print(df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
