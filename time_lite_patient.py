# -*- coding: utf-8 -*-
"""
time_lite_patient.py
====================
精确统计 Lite 精简版 pyRadiomics 单患者耗时构成：
  1. 加载 CT + 掩膜（读取耗时）
  2. pyRadiomics 16 掩膜并行提取（含逐掩膜耗时）
  3. 四类 COPD 指标（肺叶气肿 / 心肺血管 / 气道耦合 / 膈肌）各自耗时
  4. 单患者总耗时

用法：
  python time_lite_patient.py --nifti-dir <nifti_dir> \\
                              --seg-dir <seg_dir> \\
                              --patients <患者ID> --workers 8
"""
import argparse
import os
import sys
import time
import SimpleITK as sitk
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_patient_radiomics_lite as lite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nifti-dir", "-n", required=True)
    ap.add_argument("--seg-dir", "-s", required=True)
    ap.add_argument("--patients", default=None, help="逗号分隔的患者ID，默认全部")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    patients = lite.find_patients(args.seg_dir)
    if args.patients:
        wanted = set(args.patients.split(","))
        patients = [p for p in patients if p["patient"] in wanted]

    print(f"待测患者: {len(patients)} 个, workers={args.workers}\n")

    grand_rows = []
    for meta in patients:
        patient = meta["patient"]
        mask_dir = meta["mask_dir"]
        print("=" * 70)
        print(f"患者: {patient}")

        # --- 定位 CT ---
        t0 = time.time()
        ct_path = lite.resolve_ct_path(meta, args.nifti_dir)
        if not ct_path:
            print("  [FAIL] 找不到 CT，跳过")
            continue
        t_ct_resolve = time.time() - t0

        # --- 加载 CT ---
        t0 = time.time()
        ct_img = sitk.ReadImage(ct_path)
        ct_arr = sitk.GetArrayFromImage(ct_img)
        spacing = ct_img.GetSpacing()
        t_ct_load = time.time() - t0
        print(f"  CT: {os.path.basename(ct_path)}  形状={ct_arr.shape}  加载 {t_ct_load:.1f}s")

        # --- 掩膜列表 ---
        all_masks = sorted(f for f in os.listdir(mask_dir) if f.endswith('.nii.gz'))
        mask_files = [f for f in all_masks if f in lite.KEEP_FILES]
        print(f"  掩膜: {len(mask_files)} 个 (KEEP_FILES)")

        # --- pyRadiomics 并行（逐掩膜计时） ---
        jobs = [(ct_path, os.path.join(mask_dir, f), f[:-len('.nii.gz')]) for f in mask_files]
        per_mask = {}
        t0 = time.time()
        with Pool(processes=args.workers) as pool:
            # 串行 map（保留顺序）；worker 内部已打印单掩膜耗时
            results = pool.map(lite._worker_pyradiomics, jobs)
        t_pyr = time.time() - t0

        # --- 四类指标各自计时 ---
        t_load_masks = time.time()
        masks = {f[:-len('.nii.gz')]: sitk.ReadImage(os.path.join(mask_dir, f)) for f in mask_files}
        mask_arrays = {k: sitk.GetArrayFromImage(v) for k, v in masks.items()}
        t_masks_read = time.time() - t_load_masks

        t0 = time.time()
        f1 = lite.lobe_emphysema_features(ct_arr, spacing, mask_arrays)
        t_lobe = time.time() - t0

        t0 = time.time()
        f2 = lite.cardiopulmonary_features(ct_arr, spacing, mask_arrays)
        t_cardio = time.time() - t0

        t0 = time.time()
        f3 = lite.airway_lobe_coupling(mask_arrays, spacing)
        t_airway = time.time() - t0

        t0 = time.time()
        f4 = lite.diaphragm_flattening(ct_arr, mask_arrays)
        t_dia = time.time() - t0

        t_total = t_ct_resolve + t_ct_load + t_pyr + t_masks_read + t_lobe + t_cardio + t_airway + t_dia

        rows = {
            "患者": patient,
            "CT定位(s)": round(t_ct_resolve, 1),
            "CT加载(s)": round(t_ct_load, 1),
            "pyRadiomics并行(s)": round(t_pyr, 1),
            "掩膜读取(s)": round(t_masks_read, 1),
            "肺叶气肿(s)": round(t_lobe, 1),
            "心肺血管(s)": round(t_cardio, 1),
            "气道耦合(s)": round(t_airway, 1),
            "膈肌(s)": round(t_dia, 1),
            "总耗时(s)": round(t_total, 1),
            "总耗时(分钟)": round(t_total / 60, 2),
        }
        grand_rows.append(rows)
        print(f"\n  ---- 计时汇总 ----")
        for k, v in rows.items():
            print(f"    {k}: {v}")
        print(f"  pyRadiomics 占比: {t_pyr / max(t_total, 1e-9) * 100:.1f}%")

    # --- 汇总 ---
    print("\n" + "=" * 70)
    print("汇总（全部待测患者）:")
    for r in grand_rows:
        print(f"  {r['患者'][:50]:52s} 总 {r['总耗时(分钟)']} 分钟 "
              f"(pyRadiomics {r['pyRadiomics并行(s)']}s / 肺叶 {r['肺叶气肿(s)']}s / "
              f"心肺 {r['心肺血管(s)']}s / 气道 {r['气道耦合(s)']}s / 膈肌 {r['膈肌(s)']}s)")


if __name__ == "__main__":
    main()
