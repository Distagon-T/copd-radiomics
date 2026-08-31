# -*- coding: utf-8 -*-
"""
申报清单补算特征 —— 共享实现
=============================
供 compute_patient_radiomics.py / lite / fast 以及 compute_declared_features.py 复用，
保证单一实现（单点维护）。

输入：ct_img（SimpleITK 图像，原始 HU）+ masks（{掩膜名: SimpleITK 图像}，与 CT 同空间）
输出：dict，含 12 个补算特征（具体列名与说明见根 README.md §5.1）

注意：
- CAC/Agatston/MS 基于整心掩膜 HU>=130（含瓣膜/主动脉根钙化，比临床冠脉 Agatston 偏大）。
- FAI、Aorta_Wall_Thickness 为形态学/区域代理（非增强 CT 无冠脉中心线/管腔分割）。
- BronchoArtery_Ratio 由 AirQuant MATLAB 脚本（compute_airway_features.m）计算
  （Din_mean_all / 主肺动脉等效直径），radiomics 侧不输出；这里仅保留 PA_Equivalent_Diameter_mm。
"""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

FAT_LO, FAT_HI = -190.0, -30.0
CAL_TH = 130.0

# 与现有 radiomics 脚本的掩膜名一致（compute_patient_radiomics*.py 用文件名作为 key）
MASK_NAMES = ["heart", "aorta", "lung_vessels", "pulmonary_artery",
              "lung_upper_lobe_left", "lung_upper_lobe_right",
              "lung_middle_lobe_right", "lung_lower_lobe_left", "lung_lower_lobe_right"]


def _arr(img):
    return sitk.GetArrayFromImage(img)  # [z,y,x]


def voxel_vol(spacing):
    return float(np.prod(spacing))


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


def equiv_diameter_from_vol(vol_mm3, n_slices):
    """体积为管状时，按层等效圆直径 = 2*sqrt(V/(π·L))。"""
    if n_slices <= 0 or not np.isfinite(vol_mm3) or vol_mm3 <= 0:
        return np.nan
    r = np.sqrt(vol_mm3 / (np.pi * n_slices))
    return 2.0 * r


def declared_features(ct_img, masks, din_mean_all=np.nan):
    """计算 12 个申报补算特征。masks: {掩膜名: sitk 图像}。"""
    sp = ct_img.GetSpacing()  # (x,y,z)
    dx, dy, dz = sp[0], sp[1], sp[2]
    vv = voxel_vol(sp)
    CT = _arr(ct_img)  # int16 HU
    res = {}

    # ---------- 肺血管体积 / CSA ----------
    V = _arr(masks["lung_vessels"]) > 0 if masks.get("lung_vessels") is not None else None
    if V is not None:
        vvol = float(V.sum()) * vv
        slices_with = int(np.any(V, axis=(1, 2)).sum())
        res["Vessel_Volume_mm3"] = vvol
        res["Vessel_CSA_mean_mm2"] = vvol / slices_with if slices_with else np.nan
    else:
        res["Vessel_Volume_mm3"] = np.nan
        res["Vessel_CSA_mean_mm2"] = np.nan

    # ---------- 主肺动脉等效直径（支气管-血管比由 AirQuant MATLAB 计算，radiomics 侧不输出） ----------
    PA = _arr(masks["pulmonary_artery"]) > 0 if masks.get("pulmonary_artery") is not None else None
    if PA is not None:
        pa_vol = float(PA.sum()) * vv
        pa_slices = int(np.any(PA, axis=(1, 2)).sum())
        res["PA_Equivalent_Diameter_mm"] = equiv_diameter_from_vol(pa_vol, pa_slices)
    else:
        res["PA_Equivalent_Diameter_mm"] = np.nan

    # ---------- 冠脉钙化(Agatston+质量) + 心包脂肪/FAI：裁剪到 heart bbox ----------
    Hf = _arr(masks["heart"]) > 0 if masks.get("heart") is not None else None
    if Hf is not None:
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
            # 心包脂肪/FAI：真实 HU，限定在 heart 膨胀区域内
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
            res["CAC_Agatston"] = res["CAC_Mass_mg"] = res["CAC_Volume_mm3"] = np.nan
            res["EpiFat_Volume_mm3"] = res["EpiFat_Mean_HU"] = res["FAI_pericoronary_HU"] = np.nan
    else:
        res["CAC_Agatston"] = res["CAC_Mass_mg"] = res["CAC_Volume_mm3"] = np.nan
        res["EpiFat_Volume_mm3"] = res["EpiFat_Mean_HU"] = res["FAI_pericoronary_HU"] = np.nan

    # ---------- 主动脉外径 / 壁厚代理（裁剪到 bbox） ----------
    Af = _arr(masks["aorta"]) > 0 if masks.get("aorta") is not None else None
    if Af is not None:
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
        res["Aorta_Outer_Mean_Diameter_mm"] = res["Aorta_Wall_Fraction"] = res["Aorta_Wall_Thickness_mm_approx"] = np.nan

    # ---------- 心胸比 CTR ----------
    heart_ok = masks.get("heart") is not None
    lobes = [n for n in MASK_NAMES if n.startswith("lung_") and masks.get(n) is not None]
    if heart_ok and lobes:
        H = _arr(masks["heart"]) > 0
        L = np.zeros_like(H, dtype=bool)
        for n in lobes:
            L |= _arr(masks[n]) > 0
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
