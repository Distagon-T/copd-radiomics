"""
为缺失心脏/肺叶掩膜的患者补跑 TotalSegmentator（只补缺失的黄金靶区）。

背景：2026-02 中 6 个患者因 768x768x441 巨像导致 nnU-Net 高分辨率引擎
(heartchambers_highres) 报 WinError 267 失败，缺 heart_atrium/ventricle/myocardium、
个别肺动脉；另 20190403(14掩膜)、20260204(0掩膜) 分割严重失败。

本脚本：
  1. 读 <患者>_segmentation_info.json 定位实际使用的 CT（selected_nifti）
  2. 计算缺失的黄金靶区，按任务分组（heartchambers_highres / total / lung_vessels_LEGACY）
  3. 对每个需要的任务用 roi_subset 补跑（GPU），写入原 masks 目录
  4. 清理：删除非 16 黄金靶区的无关掩膜，更新 info json 为 success

用法：
  python fill_missing_masks.py --seg-dir E:/DICOM/2026-02-seg [--only 20130902] [--no-clean]
"""
import argparse
import glob
import json
import os
import time

# 提前把 nnU-Net 临时目录指到 D 盘干净路径，规避 C:\Users\ADMINI~1\... 8.3 短路径
# 触发 WinError 267 的问题
os.environ["TMP"] = r"D:\nnunet_tmp"
os.environ["TEMP"] = r"D:\nnunet_tmp"
os.makedirs(os.environ["TMP"], exist_ok=True)

from totalsegmentator.python_api import totalsegmentator  # noqa: E402

KEEP_FILES = [
    "lung_upper_lobe_left.nii.gz", "lung_lower_lobe_left.nii.gz",
    "lung_upper_lobe_right.nii.gz", "lung_middle_lobe_right.nii.gz",
    "lung_lower_lobe_right.nii.gz",
    "lung_vessels.nii.gz", "lung_trachea_bronchia.nii.gz",
    "aorta.nii.gz", "pulmonary_artery.nii.gz", "trachea.nii.gz",
    "heart.nii.gz",
    "heart_myocardium.nii.gz", "heart_atrium_left.nii.gz",
    "heart_ventricle_left.nii.gz", "heart_atrium_right.nii.gz",
    "heart_ventricle_right.nii.gz",
]
KEEP16 = [f[:-len('.nii.gz')] for f in KEEP_FILES]

# 各任务能输出的黄金靶区（roi 名，不含 .nii.gz）
# 注意：heartchambers_highres 输出 7 个掩膜（含 aorta/pulmonary_artery）；
#       total 任务 117 类里【没有】pulmonary_artery（仅来自 heartchambers_highres）。
TASK_ROIS = {
    "heartchambers_highres": [
        "heart_myocardium", "heart_atrium_left", "heart_atrium_right",
        "heart_ventricle_left", "heart_ventricle_right",
        "aorta", "pulmonary_artery",
    ],
    "total": [
        "lung_upper_lobe_left", "lung_lower_lobe_left",
        "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
        "trachea", "heart",
    ],
    "lung_vessels_LEGACY": ["lung_vessels", "lung_trachea_bronchia"],
}


def find_largest_slice_nifti(patient_dir):
    import nibabel as nib
    nii_files = sorted(glob.glob(os.path.join(patient_dir, "*.nii.gz")))
    best, bn = None, -1
    for f in nii_files:
        try:
            n = nib.load(f).shape[2]
        except Exception:
            n = 0
        if n > bn:
            best, bn = f, n
    return best


def get_input_nifti(info_json, patient_dir):
    """优先用 info.json 的 input_dir+selected_nifti；否则找层数最多的 nii.gz。"""
    if os.path.exists(info_json):
        try:
            with open(info_json, encoding="utf-8") as f:
                d = json.load(f)
            sel = d.get("selected_nifti")
            ind = d.get("input_dir") or patient_dir
            if sel:
                for base in [ind, patient_dir]:
                    cand = os.path.join(base, sel)
                    if os.path.isfile(cand):
                        return cand
        except Exception:
            pass
    return find_largest_slice_nifti(patient_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seg-dir", "-s", required=True)
    p.add_argument("--patients", "-l", default=None,
                   help="患者名单（逗号分隔），默认=seg-dir 内所有缺掩膜的")
    p.add_argument("--only", default=None, help="只处理一个患者（测试用）")
    p.add_argument("--skip", default="", help="跳过的患者前缀（逗号分隔，如 20130805）")
    p.add_argument("--no-clean", action="store_true", help="补完不清理无关掩膜")
    args = p.parse_args()

    seg_dir = os.path.abspath(args.seg_dir)
    skip = [s.strip() for s in args.skip.split(",") if s.strip()]
    # 收集待补患者（前缀匹配，支持 --only 20130902 这类短名）
    targets = []
    for d in sorted(os.listdir(seg_dir)):
        if not d.endswith("_masks") or not os.path.isdir(os.path.join(seg_dir, d)):
            continue
        patient = d[:-len("_masks")]
        if any(patient.startswith(s) or patient == s for s in skip):
            continue
        if args.only and not patient.startswith(args.only):
            continue
        if args.patients:
            wanted = {x.strip() for x in args.patients.split(",") if x.strip()}
            if not any(patient.startswith(w) or patient == w for w in wanted):
                continue
        md = os.path.join(seg_dir, d)
        present = {f for f in os.listdir(md) if f.endswith(".nii.gz")}
        missing = [k for k in KEEP16 if k + ".nii.gz" not in present]
        if missing:
            targets.append((patient, md, missing))

    print(f"待补患者 {len(targets)} 个")
    for patient, md, missing in targets:
        print(f"  {patient[:44]:44s} 缺 {len(missing)}: {','.join(missing)}")

    for patient, md, missing in targets:
        info_json = os.path.join(md, f"{patient}_segmentation_info.json")
        nifti = get_input_nifti(info_json, os.path.dirname(md))
        if nifti is None:
            print(f"  [FAIL] {patient}: 找不到 CT")
            continue
        print(f"\n=== {patient} ===")
        print(f"  CT: {os.path.basename(nifti)}")

        # 决定需要跑哪些完整任务（roi_subset 只支持 total/total_mr，这里统一跑整任务，
        # GPU 推理很快(~30s)，多余掩膜最后由 cleanup 清理到 16 靶区）
        # nr_thr_saving=1：强制串行保存（内存内直接写，不开多进程 Pool 读临时文件），
        # 规避 Windows 上保存大体积时的 WinError 267 竞态。
        needed_tasks = set()
        for roi in missing:
            for task, rois in TASK_ROIS.items():
                if roi in rois:
                    needed_tasks.add(task)
                    break
        ok = True
        for task in sorted(needed_tasks):
            t0 = time.time()
            print(f"  补跑 {task} ...")
            try:
                totalsegmentator(nifti, md, task=task, device="gpu",
                                 force_split=True, nr_thr_saving=1, quiet=False)
                print(f"    -> 完成 {task}（{time.time()-t0:.1f}s）")
            except Exception as e:
                print(f"    [FAIL] {task} 失败: {e}")
                ok = False
                # 失败时写回 info json，保留现场
                if os.path.exists(info_json):
                    try:
                        with open(info_json, encoding="utf-8") as f:
                            d = json.load(f)
                        d["status"] = "failed"
                        d["error"] = f"fill_missing_masks {task}: {e}"
                        d["elapsed_seconds"] = round(time.time() - t0, 2)
                        d["masks"] = sorted(f for f in os.listdir(md) if f.endswith(".nii.gz"))
                        with open(info_json, "w", encoding="utf-8") as f:
                            json.dump(d, f, indent=2, ensure_ascii=False)
                    except Exception:
                        pass

        # 清理到 16 黄金靶区
        if ok and not args.no_clean:
            removed = 0
            for f in os.listdir(md):
                if f.endswith(".nii.gz") and f not in KEEP_FILES:
                    os.remove(os.path.join(md, f))
                    removed += 1
            if removed:
                print(f"  清理无关掩膜 {removed} 个 -> 保留 {len(KEEP_FILES)} 黄金靶区")

        # 更新 info json 为 success（含最终掩膜清单）
        present = sorted(f for f in os.listdir(md) if f.endswith(".nii.gz"))
        if os.path.exists(info_json):
            try:
                with open(info_json, encoding="utf-8") as f:
                    d = json.load(f)
                d["masks"] = present
                d["status"] = "success" if ok else "failed"
                if ok:
                    d["error"] = None
                with open(info_json, "w", encoding="utf-8") as f:
                    json.dump(d, f, indent=2, ensure_ascii=False)
            except Exception:
                pass
        print(f"  [{'OK' if ok else 'FAIL'}] {patient} 最终掩膜 {len(present)} 个")


if __name__ == "__main__":
    main()
