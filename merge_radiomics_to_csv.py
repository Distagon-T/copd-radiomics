# -*- coding: utf-8 -*-
"""
合并所有患者的 radiomics JSON 为一个 CSV
=========================================
扫描分割结果目录下的 <患者>_radiomics.json（由 compute_patient_radiomics.py 生成），
合并为一张表：每行一个患者，列为全部特征。

同时从每个患者的 <患者>_segmentation_info.json 中提取 PatientID（DICOM 患者 ID），
作为单独一列 PatientID 写入 CSV（保留 Patient_ID 文件名列）。

用法：
  python merge_radiomics_to_csv.py --seg-dir E:\\DICOM\\2026-07-lung-seg \\
                                   --output E:\\DICOM\\2026-07-lung-seg\\radiomics_all_patients.csv
"""
import os
import sys
import glob
import json
import argparse
import pandas as pd

# Windows 控制台 GBK 编码兼容
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser(description="合并 radiomics JSON -> CSV")
    p.add_argument("--seg-dir", "-s", required=True, help="分割结果目录（含 <患者>_radiomics.json）")
    p.add_argument("--nifti-dir", "-n", default=None,
                   help="原始 CT 患者目录（可选）：当分割 info json 里提取不到 PatientID 时，"
                        "去这里找 <患者>/<患者>_dicom_info.json 提取")
    p.add_argument("--output", "-o", default=None, help="输出 CSV 路径（默认 <seg-dir>/radiomics_all_patients.csv）")
    return p.parse_args()


def extract_patient_id_from_seg(seg_dir, patient_name):
    """
    从分割文件夹的 <患者>_segmentation_info.json 提取 DICOM PatientID。
    路径: series_info.candidates[].series_info.Patient.PatientID
    找不到返回 None。
    """
    info_json = os.path.join(seg_dir, f"{patient_name}_masks",
                             f"{patient_name}_segmentation_info.json")
    if not os.path.exists(info_json):
        return None
    try:
        with open(info_json, encoding="utf-8") as f:
            info = json.load(f)
        cands = info.get("series_info", {}).get("candidates", [])
        for c in cands:
            pid = c.get("series_info", {}).get("Patient", {}).get("PatientID")
            if pid:
                return str(pid).strip()
    except Exception:
        pass
    return None


def extract_patient_id_from_nifti(nifti_dir, patient_name):
    """
    回退：从原始 CT 文件夹的 <患者>_dicom_info.json 提取 PatientID。
    路径: Series[].Patient.PatientID（任意一个 series 有即可）
    找不到返回 None。
    """
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


def extract_patient_id(seg_dir, nifti_dir, patient_name):
    """先查分割 info json，再回退到原始 CT 的 dicom_info json。"""
    pid = extract_patient_id_from_seg(seg_dir, patient_name)
    if pid is None:
        pid = extract_patient_id_from_nifti(nifti_dir, patient_name)
    return pid


def main():
    args = parse_args()
    seg_dir = os.path.abspath(args.seg_dir)
    nifti_dir = os.path.abspath(args.nifti_dir) if args.nifti_dir else None
    output = args.output or os.path.join(seg_dir, "radiomics_all_patients.csv")

    # 收集所有 radiomics json
    json_files = sorted(glob.glob(os.path.join(seg_dir, "*_radiomics.json")))
    print(f"发现 {len(json_files)} 个 radiomics json")

    if not json_files:
        print("没有找到任何 *_radiomics.json，请先运行 compute_patient_radiomics.py")
        return

    rows = []
    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        # 从 patient 名（<患者>_radiomics.json 去掉后缀）提取 PatientID
        patient_name = os.path.basename(jf)[:-len("_radiomics.json")]
        data["PatientID"] = extract_patient_id(seg_dir, nifti_dir, patient_name)
        rows.append(data)

    df = pd.DataFrame(rows)

    # 把 PatientID 列移到最前（紧跟 Patient_ID / CT_Series），并保持字符串（保留前导零）
    if "PatientID" in df.columns:
        df["PatientID"] = df["PatientID"].astype(str)
        id_cols = [c for c in ("Patient_ID", "PatientID", "CT_Series") if c in df.columns]
        other_cols = [c for c in df.columns if c not in id_cols]
        df = df[id_cols + other_cols]

    # 清洗：仅把「数值型列」转成 float；字符串列（Patient_ID 等）保留原样
    for col in df.columns:
        if col in ("Patient_ID", "PatientID", "CT_Series"):
            continue  # 标识列始终保持字符串
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif df[col].dtype == object:
            # object 列：若多数值可转数字则转换，否则保留为字符串
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().sum() >= df[col].notna().sum() * 0.8:
                df[col] = coerced

    # 按患者排序
    if "Patient_ID" in df.columns:
        df = df.sort_values("Patient_ID").reset_index(drop=True)

    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"✅ 合并完成！{len(df)} 个患者 × {len(df.columns)} 个特征")
    print(f"输出: {output}")
    print(f"\n预览（前 5 个特征列 + PatientID）:")
    feat_cols = [c for c in df.columns if c not in ("Patient_ID", "CT_Series", "PatientID")]
    print(f"  总特征数: {len(feat_cols)}")
    preview_cols = [c for c in ("Patient_ID", "PatientID", "CT_Series") if c in df.columns]
    print(df[preview_cols + feat_cols[:5]].to_string(index=False))


if __name__ == "__main__":
    main()
