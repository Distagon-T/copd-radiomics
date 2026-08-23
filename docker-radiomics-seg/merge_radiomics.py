#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并全部患者的 radiomics JSON 为 CSV（Docker/Mac 版）
=====================================================
扫描输出目录下的 <患者>_radiomics.json，汇成一张表。
同时从分割 info json / 原始 CT 文件夹的 dicom_info json 提取 PatientID，
作为单独一列写入 CSV（保留 Patient_ID 文件名列）。
"""
import glob
import json
import os
import pandas as pd


def _extract_patient_id_from_seg(seg_dir, patient_name):
    """从 <患者>_masks/<患者>_segmentation_info.json 提取 PatientID。"""
    info_json = os.path.join(seg_dir, f"{patient_name}_masks",
                             f"{patient_name}_segmentation_info.json")
    if not os.path.exists(info_json):
        return None
    try:
        with open(info_json, encoding="utf-8") as f:
            info = json.load(f)
        for c in info.get("series_info", {}).get("candidates", []):
            pid = c.get("series_info", {}).get("Patient", {}).get("PatientID")
            if pid:
                return str(pid).strip()
    except Exception:
        pass
    return None


def _extract_patient_id_from_nifti(nifti_dir, patient_name):
    """回退：从原始 CT 文件夹 <患者>/<患者>_dicom_info.json 提取 PatientID。"""
    if not nifti_dir:
        return None
    pdir = os.path.join(nifti_dir, patient_name)
    if not os.path.isdir(pdir):
        return None
    for f in os.listdir(pdir):
        if f.endswith("_dicom_info.json"):
            try:
                with open(os.path.join(pdir, f), encoding="utf-8") as fh:
                    data = json.load(fh)
                for s in data.get("Series", []):
                    pid = s.get("Patient", {}).get("PatientID")
                    if pid:
                        return str(pid).strip()
            except Exception:
                pass
            break
    return None


def merge_to_csv(seg_dir, output=None, nifti_dir=None):
    """
    扫描 seg_dir 下所有 *_radiomics.json 并合并为 CSV（含 PatientID 列）。
    nifti_dir: 原始 CT 患者目录（可选），当分割 info json 提取不到 PatientID 时回退。
    返回 CSV 路径。
    """
    output = output or os.path.join(seg_dir, "radiomics_all_patients.csv")
    json_files = sorted(glob.glob(os.path.join(seg_dir, "*_radiomics.json")))
    print(f"发现 {len(json_files)} 个 radiomics json")

    if not json_files:
        print("没有找到任何 *_radiomics.json。")
        return None

    rows = []
    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        patient_name = os.path.basename(jf)[:-len("_radiomics.json")]
        pid = _extract_patient_id_from_seg(seg_dir, patient_name)
        if pid is None:
            pid = _extract_patient_id_from_nifti(nifti_dir, patient_name)
        data["PatientID"] = pid
        rows.append(data)

    df = pd.DataFrame(rows)

    # PatientID 列移到最前（紧跟 Patient_ID / CT_Series），保持字符串（保留前导零）
    if "PatientID" in df.columns:
        df["PatientID"] = df["PatientID"].astype(str)
        id_cols = [c for c in ("Patient_ID", "PatientID", "CT_Series") if c in df.columns]
        other_cols = [c for c in df.columns if c not in id_cols]
        df = df[id_cols + other_cols]

    # 数值列转 float；字符串列（Patient_ID/PatientID/CT_Series）保留
    for col in df.columns:
        if col in ("Patient_ID", "PatientID", "CT_Series"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif df[col].dtype == object:
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().sum() >= df[col].notna().sum() * 0.8:
                df[col] = coerced

    if "Patient_ID" in df.columns:
        df = df.sort_values("Patient_ID").reset_index(drop=True)

    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"合并完成！{len(df)} 个患者 × {len(df.columns)} 个特征 -> {output}")
    return output


if __name__ == "__main__":
    import sys
    seg = sys.argv[1] if len(sys.argv) > 1 else "."
    nifti = sys.argv[2] if len(sys.argv) > 2 else None
    merge_to_csv(seg, nifti_dir=nifti)

