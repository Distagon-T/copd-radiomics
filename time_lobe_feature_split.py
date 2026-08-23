# -*- coding: utf-8 -*-
"""定位肺叶掩膜 pyRadiomics 最耗时特征类别：shape vs firstorder 分解计时"""
import os
import time
import SimpleITK as sitk
from radiomics import featureextractor
import logging

logging.getLogger('radiomics').setLevel(logging.WARNING)
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(0)

CT = r'<path/to/CT.nii.gz>'            # TODO: 替换为实际 CT 路径
MASK_DIR = r'<path/to/patient_masks>'  # TODO: 替换为实际掩膜目录

BASE = {'binWidth': 25, 'force2D': False, 'voxelArrayShift': 1000,
        'interpolator': sitk.sitkBSpline, 'preCrop': True}


def run(roi_name, shape_on, firstorder_on, label):
    ct = sitk.ReadImage(CT)
    mask = sitk.ReadImage(os.path.join(MASK_DIR, roi_name + '.nii.gz'))
    ext = featureextractor.RadiomicsFeatureExtractor(**BASE)
    ext.disableAllFeatures()
    if shape_on:
        ext.enableFeaturesByName(shape=[])
    if firstorder_on:
        ext.enableFeaturesByName(firstorder=[])
    t0 = time.time()
    out = ext.execute(ct, mask)
    dt = time.time() - t0
    n = len([k for k in out if not k.startswith('diagnostics_')])
    print(f"[{label}] {roi_name}: {n} 特征, 耗时 {dt:.1f}s", flush=True)
    return dt


# 对照：shape+firstorder / 仅shape / 仅firstorder，掩膜取最慢的右下肺叶
results = {}
results['shape+firstorder'] = run('lung_lower_lobe_right', True, True, 'A')
results['仅shape'] = run('lung_lower_lobe_right', True, False, 'B')
results['仅firstorder'] = run('lung_lower_lobe_right', False, True, 'C')
print('\n=== 汇总 ===')
for k, v in results.items():
    print(f"{k}: {v:.1f}s")
