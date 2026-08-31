# -*- coding: utf-8 -*-
"""
补算申报清单中缺失的 9 类特征（基于已有分割掩膜 + 原始 CT，无需新分割模型）
==============================================================================
数据：建模表 patients_feature_label.csv（含患者列表、队列、source_CT_Series、
      已有气道指标 Din_mean_all 等）+ 原始 CT（E:\\DICOM\\<队列>-nifti\\<患者>\\）
      + 分割掩膜（E:\\DICOM\\<队列>-seg\\<患者>_masks\\，16 个 ROI，与 CT 同空间）
输出：每患者一行，新列如下

  1) Vessel_Volume_mm3            肺血管体积（lung_vessels 掩膜体素 × 体素体积）
  2) Vessel_CSA_mean_mm2          肺血管平均横截面积（体积 / 含血管的轴向层数）
  3) BronchoArtery_Ratio          支气管-血管比（代理：气道 Din_mean_all / 主肺动脉等效直径）
  4) CAC_Agatston                 冠脉钙化 Agatston 积分（heart 掩膜内 HU>=130，按层连接域面积×HU权重）
  5) CAC_Mass_mg                  钙化质量积分（ΣHU×体素体积/110）
  6) EpiFat_Volume_mm3            心包脂肪体积（heart 掩膜膨胀 8mm 内 HU∈[-190,-30]）
  7) EpiFat_Mean_HU               心包脂肪平均密度
  8) FAI_pericoronary_HU          冠周脂肪衰减指数（代理：heart 掩膜膨胀 3mm 内脂肪平均 HU）
  9) Aorta_Outer_Mean_Diameter_mm 主动脉平均外径（2*sqrt(V/(π·L))，按层平均）
 10) Aorta_Wall_Fraction          主动脉壁占比（1 - 内缩2mm后体积/原始体积）
 11) Aorta_Wall_Thickness_mm_approx 主动脉壁厚近似（wall_fraction × 外径/2，注意：非增强CT无管腔分割，
     本值仅为形态学代理，需用专用血管壁分割验证）
 12) CardioThoracic_Ratio         心胸比（心脏最大横径 / 同层肺轮廓内横径，取最大值）

说明：原始 CT 直接读（dcm2nii 已应用 slope/intercept → 值为 HU）；掩膜为二值。
     脂/钙化阈值基于 HU；FAI/壁厚为明示的近似代理，用于申报量级评估。
用法：
  python compute_declared_features.py [--csv E:\\DICOM\\results\\patients_feature_label.csv]
      [--out E:\\DICOM\\results\\declared_features_computed.csv] [--limit N] [--patients "id1,id2"]
断点续传：已存在于 --out 的患者跳过。
"""
import argparse
import os
import re
import sys
import datetime
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_CSV = r"E:\DICOM\results\patients_feature_label.csv"
DEFAULT_OUT = r"E:\DICOM\results\declared_features_computed.csv"
COHORT_MAP = {"Jan-26": "2026-01", "Feb-26": "2026-02", "Apr-26": "2026-04", "May-26": "2026-05"}
FAT_LO, FAT_HI = -190.0, -30.0
CAL_TH = 130.0
MASK_NAMES = ["heart", "aorta", "lung_vessels", "pulmonary_artery",
              "lung_upper_lobe_left", "lung_upper_lobe_right",
              "lung_middle_lobe_right", "lung_lower_lobe_left", "lung_lower_lobe_right"]


def arr(img):
    return sitk.GetArrayFromImage(img)  # [z,y,x]


def crop_slice(mask, margin=5):
    """返回 mask 三维包围盒（含 margin）的切片对象；mask 为空返回 None。"""
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return None
    z0, z1 = int(idx[:, 0].min()), int(idx[:, 0].max())
    y0, y1 = int(idx[:, 1].min()), int(idx[:, 1].max())
    x0, x1 = int(idx[:, 2].min()), int(idx[:, 2].max())
    z0 = max(0, z0 - margin); z1 = min(mask.shape[0] - 1, z1 + margin)
    y0 = max(0, y0 - margin); y1 = min(mask.shape[1] - 1, y1 + margin)
    x0 = max(0, x0 - margin); x1 = min(mask.shape[2] - 1, x1 + margin)
    return (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))


def voxel_vol(spacing):
    return float(np.prod(spacing))


def equiv_diameter_from_vol(vol_mm3, n_slices):
    """体积为管状时，按层等效圆直径 = 2*sqrt(V/(π·L))。"""
    if n_slices <= 0 or not np.isfinite(vol_mm3) or vol_mm3 <= 0:
        return np.nan
    r = np.sqrt(vol_mm3 / (np.pi * n_slices))
    return 2.0 * r


def compute_patient(ct_img, masks, din_mean_all):
    """ct_img: SimpleITK image (HU)；masks: {name: SimpleITK image}；返回特征 dict。
    注意：CT 保持原始 int16 数组（HU 值在 int16 范围内），仅在掩膜区域内做运算，节省内存。
    """
    sp = ct_img.GetSpacing()  # (x,y,z)
    dx, dy, dz = sp[0], sp[1], sp[2]
    vv = voxel_vol(sp)
    CT = arr(ct_img)  # int16 HU
    res = {}

    # ---------- 肺血管体积 / CSA ----------
    if "lung_vessels" in masks and masks["lung_vessels"] is not None:
        V = arr(masks["lung_vessels"]) > 0
        vvol = float(V.sum()) * vv
        slices_with = int(np.any(V, axis=(1, 2)).sum())
        res["Vessel_Volume_mm3"] = vvol
        res["Vessel_CSA_mean_mm2"] = vvol / slices_with if slices_with else np.nan
    else:
        res["Vessel_Volume_mm3"] = np.nan
        res["Vessel_CSA_mean_mm2"] = np.nan

    # ---------- 主肺动脉等效直径 + 支气管-血管比 ----------
    if "pulmonary_artery" in masks and masks["pulmonary_artery"] is not None:
        PA = arr(masks["pulmonary_artery"]) > 0
        pa_vol = float(PA.sum()) * vv
        pa_slices = int(np.any(PA, axis=(1, 2)).sum())
        pa_d = equiv_diameter_from_vol(pa_vol, pa_slices)
        res["PA_Equivalent_Diameter_mm"] = pa_d
        if np.isfinite(din_mean_all) and np.isfinite(pa_d) and pa_d > 0:
            res["BronchoArtery_Ratio"] = float(din_mean_all) / pa_d
        else:
            res["BronchoArtery_Ratio"] = np.nan
    else:
        res["PA_Equivalent_Diameter_mm"] = np.nan
        res["BronchoArtery_Ratio"] = np.nan

    # ---------- 冠脉钙化(Agatston+质量) + 心包脂肪/FAI：先裁剪到 heart bbox ----------
    if "heart" in masks and masks["heart"] is not None:
        Hf = arr(masks["heart"]) > 0
        sl = crop_slice(Hf, margin=12)
        if sl is not None:
            Hc = Hf[sl]
            Cc = CT[sl].copy()
            # CAC：只在 heart 掩膜内取真实 HU，掩膜外填 -2000（排除非心脏高密度组织）
            HUh = np.where(Hc, Cc.astype(np.float32), -2000.0)
            cal = (HUh >= CAL_TH)
            agatston = 0.0
            mass_mg = 0.0
            cal_vol = 0.0
            for z in range(cal.shape[0]):
                slab = cal[z]
                if not slab.any():
                    continue
                lab, n = ndimage.label(slab)
                for li in range(1, n + 1):
                    mask_li = lab == li
                    area_mm2 = float(mask_li.sum()) * dx * dy
                    peak = float(HUh[z][mask_li].max())
                    w = 1 if peak < 200 else (2 if peak < 300 else (3 if peak < 400 else 4))
                    agatston += area_mm2 * w
                mass_mg += float(HUh[z][cal[z]].sum()) * (vv / 1000.0) / 110.0
                cal_vol += float(cal[z].sum()) * vv
            res["CAC_Agatston"] = agatston
            res["CAC_Mass_mg"] = mass_mg
            res["CAC_Volume_mm3"] = cal_vol
            # --- 心包脂肪/FAI：真实 HU，限定在 heart 膨胀区域内 ---
            fat = (Cc >= FAT_LO) & (Cc <= FAT_HI)
            se8z = max(1, int(round(8 / dz))) if dz > 0 else 1
            H8 = ndimage.binary_dilation(Hc, structure=np.ones((se8z, 3, 3), dtype=np.uint8))
            H3 = ndimage.binary_dilation(Hc, structure=np.ones((1, 3, 3), dtype=np.uint8))
            f8 = fat & H8
            f3 = fat & H3
            res["EpiFat_Volume_mm3"] = float(f8.sum()) * vv
            res["EpiFat_Mean_HU"] = float(Cc[f8].mean()) if f8.any() else np.nan
            res["FAI_pericoronary_HU"] = float(Cc[f3].mean()) if f3.any() else np.nan
        else:
            res["CAC_Agatston"] = np.nan
            res["CAC_Mass_mg"] = np.nan
            res["CAC_Volume_mm3"] = np.nan
            res["EpiFat_Volume_mm3"] = np.nan
            res["EpiFat_Mean_HU"] = np.nan
            res["FAI_pericoronary_HU"] = np.nan
    else:
        res["CAC_Agatston"] = np.nan
        res["CAC_Mass_mg"] = np.nan
        res["CAC_Volume_mm3"] = np.nan
        res["EpiFat_Volume_mm3"] = np.nan
        res["EpiFat_Mean_HU"] = np.nan
        res["FAI_pericoronary_HU"] = np.nan

    # ---------- 主动脉外径 / 壁厚代理（裁剪到 bbox） ----------
    if "aorta" in masks and masks["aorta"] is not None:
        Af = arr(masks["aorta"]) > 0
        a_vol = float(Af.sum()) * vv
        a_slices = int(np.any(Af, axis=(1, 2)).sum())
        outer_d = equiv_diameter_from_vol(a_vol, a_slices)
        res["Aorta_Outer_Mean_Diameter_mm"] = outer_d
        sl = crop_slice(Af, margin=6)
        if sl is not None:
            Ac = Af[sl]
            se2z = max(1, int(round(2 / dz))) if dz > 0 else 1
            A_er = ndimage.binary_erosion(Ac, structure=np.ones((se2z, 3, 3), dtype=np.uint8))
            v_er = float(A_er.sum()) * vv
            wall_frac = 1.0 - (v_er / a_vol) if a_vol > 0 else np.nan
        else:
            wall_frac = np.nan
        res["Aorta_Wall_Fraction"] = wall_frac
        res["Aorta_Wall_Thickness_mm_approx"] = wall_frac * outer_d / 2.0 if np.isfinite(wall_frac) and np.isfinite(outer_d) else np.nan
    else:
        res["Aorta_Outer_Mean_Diameter_mm"] = np.nan
        res["Aorta_Wall_Fraction"] = np.nan
        res["Aorta_Wall_Thickness_mm_approx"] = np.nan

    # ---------- 心胸比 CTR ----------
    heart_ok = "heart" in masks and masks["heart"] is not None
    lobes = [n for n in MASK_NAMES if n.startswith("lung_") and n in masks and masks[n] is not None]
    if heart_ok and lobes:
        H = arr(masks["heart"]) > 0
        L = np.zeros_like(H, dtype=bool)
        for n in lobes:
            L |= arr(masks[n]) > 0
        ctr = 0.0
        for z in range(H.shape[0]):
            hz = H[z]
            if not hz.any():
                continue
            hw = int(hz.any(axis=0).sum()) * dx
            lw = int(L[z].any(axis=0).sum()) * dx if L[z].any() else 0
            if lw > 0:
                ctr = max(ctr, hw / lw)
        res["CardioThoracic_Ratio"] = ctr if ctr > 0 else np.nan
    else:
        res["CardioThoracic_Ratio"] = np.nan
    return res


def resolve_paths(row):
    ser = row["source_CT_Series"]
    if not isinstance(ser, str):
        return None
    folder = ser.rsplit("_", 1)[0]
    m = re.search(r"_(\d+)\.nii\.gz$", ser)
    if not m:
        return None
    raw = COHORT_MAP.get(row["cohort"])
    if not raw:
        return None
    ct = os.path.join("E:/DICOM", raw + "-nifti", folder, ser)
    md = os.path.join("E:/DICOM", raw + "-seg", folder + "_masks")
    return ct, md


def _process_one(meta):
    """多进程 worker：加载 CT+掩膜并计算一个患者的特征。meta 为 dict。"""
    import SimpleITK as _sitk
    row = {"PatientID": meta["PatientID"], "cohort": meta["cohort"]}
    try:
        paths = resolve_paths(meta)
        if paths is None:
            row["_err"] = "no_path"
            return row
        ct, md = paths
        if not os.path.exists(ct):
            row["_err"] = "no_ct"
            return row
        ct_img = _sitk.ReadImage(ct)
        masks = {}
        for n in MASK_NAMES:
            p = os.path.join(md, n + ".nii.gz")
            masks[n] = _sitk.ReadImage(p) if os.path.exists(p) else None
        din = float(meta["Din_mean_all"]) if np.isfinite(meta.get("Din_mean_all", np.nan)) else np.nan
        feats = compute_patient(ct_img, masks, din)
        row.update(feats)
        row["_err"] = ""
    except Exception as e:
        row["_err"] = str(e)[:120]
    return row


OUT_COLS = ["PatientID", "cohort", "Vessel_Volume_mm3", "Vessel_CSA_mean_mm2",
            "PA_Equivalent_Diameter_mm", "BronchoArtery_Ratio",
            "CAC_Agatston", "CAC_Mass_mg", "CAC_Volume_mm3",
            "EpiFat_Volume_mm3", "EpiFat_Mean_HU", "FAI_pericoronary_HU",
            "Aorta_Outer_Mean_Diameter_mm", "Aorta_Wall_Fraction",
            "Aorta_Wall_Thickness_mm_approx", "CardioThoracic_Ratio", "_err"]


def main():
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--patients", default=None)
    ap.add_argument("--workers", type=int, default=4, help="并行进程数（每进程约 1.5-2GB 内存）")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False, dtype={"PatientID": str})
    df["PatientID"] = df["PatientID"].astype(str).str.replace(r"\.0$", "", regex=True)
    if args.patients:
        ids = set(args.patients.split(","))
        df = df[df["PatientID"].isin(ids)]
    if args.limit:
        df = df.head(args.limit)
    df = df[df["source_CT_Series"].notna()].reset_index(drop=True)
    print(f"待处理患者: {len(df)}", flush=True)

    done_rows = []
    done_ids = set()
    if os.path.exists(args.out):
        try:
            prev = pd.read_csv(args.out, low_memory=False, dtype={"PatientID": str})
            done_rows = prev.to_dict("records")
            done_ids = set(prev["PatientID"].astype(str))
            print(f"[resume] 已有 {len(done_ids)} 例", flush=True)
        except Exception:
            pass

    todo = []
    for _, r in df.iterrows():
        pid = r["PatientID"]
        if pid in done_ids:
            continue
        todo.append({"PatientID": pid, "cohort": r["cohort"],
                     "source_CT_Series": r["source_CT_Series"],
                     "Din_mean_all": r.get("Din_mean_all", np.nan)})
    print(f"待计算: {len(todo)} 例，workers={args.workers}", flush=True)
    t0 = datetime.datetime.now()
    done_rows_out = []
    with Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_process_one, todo, chunksize=4), 1):
            done_rows_out.append({k: res.get(k, np.nan) for k in OUT_COLS})
            if i % 50 == 0 or i == len(todo):
                all_rows = done_rows + done_rows_out
                pd.DataFrame(all_rows).to_csv(args.out, index=False)
                print(f"[{i}/{len(todo)}] elapsed={datetime.datetime.now()-t0}", flush=True)
    out_df = pd.DataFrame(done_rows + done_rows_out)
    out_df.to_csv(args.out, index=False)
    n_err = int((out_df.get("_err", "").astype(str) != "").sum()) if "_err" in out_df else 0
    print(f"[done] 完成 {len(out_df)} 例，失败 {n_err} 例 -> {args.out}", flush=True)
    if "_err" in out_df:
        print(out_df.loc[out_df["_err"].astype(str) != "", ["PatientID", "_err"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
