# -*- coding: utf-8 -*-
"""
整合 radiomics + AQ(气道) 特征为一张 CSV（每患者一行）
========================================================
- radiomics: 扫描 <seg-dir>/*_radiomics.json（Fast 全套特征 ~2207 键/患者）
- AQ      : 扫描 <aq-dir>/*_airway_features.csv（compute_airway_features.m 输出，每患者一行）
- 按 患者名(Patient_ID == patient_folder) 左连接 radiomics + AQ
- 输出: <output>（默认 <seg-dir>/integrated_features.csv）

用法:
  python merge_all_features_to_csv.py --seg-dir E:/DICOM/2026-02-seg \
      --aq-dir E:/DICOM/2026-02-Airway_features \
      --output E:/DICOM/2026-02-seg/integrated_radiomics_aq.csv
"""
import os
import sys
import glob
import json
import argparse
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser(description="整合 radiomics + AQ 特征 -> CSV")
    p.add_argument("--seg-dir", "-s", required=True, help="radiomics JSON 目录（含 <患者>_radiomics.json）")
    p.add_argument("--aq-dir", "-a", required=True, help="AQ 特征目录（含 <患者>_airway_features.csv）")
    p.add_argument("--nifti-dir", "-n", default=None, help="原始 CT 目录（可选，补 PatientID）")
    p.add_argument("--output", "-o", default=None, help="输出 CSV（默认 <seg-dir>/integrated_radiomics_aq.csv）")
    return p.parse_args()


def extract_patient_id(seg_dir, nifti_dir, patient_name):
    """从分割 info json 或原始 CT dicom_info json 提取 DICOM PatientID。"""
    info_json = os.path.join(seg_dir, f"{patient_name}_masks", f"{patient_name}_segmentation_info.json")
    if os.path.exists(info_json):
        try:
            with open(info_json, encoding="utf-8") as f:
                info = json.load(f)
            for c in info.get("series_info", {}).get("candidates", []):
                pid = c.get("series_info", {}).get("Patient", {}).get("PatientID")
                if pid:
                    return str(pid).strip()
        except Exception:
            pass
    if nifti_dir:
        pdir = os.path.join(nifti_dir, patient_name)
        if os.path.isdir(pdir):
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


def load_radiomics(seg_dir, nifti_dir):
    files = sorted(glob.glob(os.path.join(seg_dir, "*_radiomics.json")))
    print(f"  radiomics JSON: {len(files)} 个")
    rows = []
    for jf in files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        patient = os.path.basename(jf)[:-len("_radiomics.json")]
        data.setdefault("Patient_ID", patient)
        data["PatientID"] = extract_patient_id(seg_dir, nifti_dir, patient)
        rows.append(data)
    df = pd.DataFrame(rows)
    if "PatientID" in df.columns:
        df["PatientID"] = df["PatientID"].astype(str)
    return df


def load_aq(aq_dir):
    files = sorted(glob.glob(os.path.join(aq_dir, "*_airway_features.csv")))
    print(f"  AQ features CSV: {len(files)} 个")
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if "patient_folder" in d.columns:
                d["patient_folder"] = d["patient_folder"].astype(str)
            frames.append(d)
        except Exception as e:
            print(f"    [warn] 读取 {os.path.basename(f)} 失败: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    args = parse_args()
    seg_dir = os.path.abspath(args.seg_dir)
    aq_dir = os.path.abspath(args.aq_dir)
    nifti_dir = os.path.abspath(args.nifti_dir) if args.nifti_dir else None
    output = args.output or os.path.join(seg_dir, "integrated_radiomics_aq.csv")

    print("=== 读取 radiomics ===")
    rdf = load_radiomics(seg_dir, nifti_dir)
    print(f"  {rdf.shape[0]} 患者 x {rdf.shape[1]} 列")

    print("=== 读取 AQ ===")
    adf = load_aq(aq_dir)
    print(f"  {adf.shape[0]} 患者 x {adf.shape[1]} 列")

    if rdf.empty:
        print("无 radiomics 数据，退出。")
        return

    # 合并：radiomics 左连接 AQ
    aq_key = "patient_folder"
    if not adf.empty and aq_key in adf.columns:
        # AQ 中重名去重（保留第一个）
        adf = adf.drop_duplicates(subset=[aq_key], keep="first")
        aq_cols = [c for c in adf.columns if c != aq_key]
        merged = rdf.merge(adf[[aq_key] + aq_cols], left_on="Patient_ID", right_on=aq_key,
                           how="left")
        merged = merged.drop(columns=[aq_key])
        n_aq = merged[aq_cols].notna().any(axis=1).sum()
        print(f"  匹配到 AQ 的患者: {n_aq}/{rdf.shape[0]}（无 AQ 的患者对应列为 NaN）")
    else:
        merged = rdf.copy()
        print("  无 AQ 数据，仅输出 radiomics。")

    # 类型清洗：标识列保字符串，其余尽量转数值
    id_cols = {"Patient_ID", "PatientID", "CT_Series"}
    for col in merged.columns:
        if col in id_cols:
            continue
        if pd.api.types.is_numeric_dtype(merged[col]):
            continue
        if merged[col].dtype == object:
            coerced = pd.to_numeric(merged[col], errors="coerce")
            if coerced.notna().sum() >= merged[col].notna().sum() * 0.8:
                merged[col] = coerced

    # 排序
    if "Patient_ID" in merged.columns:
        merged = merged.sort_values("Patient_ID").reset_index(drop=True)

    merged.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\n✅ 完成！{merged.shape[0]} 患者 × {merged.shape[1]} 列")
    print(f"输出: {output}")


if __name__ == "__main__":
    main()
