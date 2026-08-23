# -*- coding: utf-8 -*-
"""对照实验：肺叶掩膜上 GLCM 等纹理特征的增量耗时"""
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
        'interpolator': sitk.sitkBSpline}


def run(label, roi_name, enable):
    ct = sitk.ReadImage(CT)
    mask = sitk.ReadImage(os.path.join(MASK_DIR, roi_name + '.nii.gz'))
    ext = featureextractor.RadiomicsFeatureExtractor(**BASE)
    ext.disableAllFeatures()
    for cls, args in enable.items():
        ext.enableFeaturesByName(**{cls: args})
    t0 = time.time()
    out = ext.execute(ct, mask)
    dt = time.time() - t0
    n = len([k for k in out if not k.startswith('diagnostics_')])
    print(f"[{label}] {roi_name}: {n} 特征, 耗时 {dt:.1f}s", flush=True)
    return dt


# 用中等大小的肺叶（middle_lobe，shape+firstorder 约 46s）
roi = 'lung_middle_lobe_right'
r = {}
r['shape+firstorder(基准)'] = run('A', roi, {'shape': [], 'firstorder': []})
r['+GLCM'] = run('B', roi, {'shape': [], 'firstorder': [], 'glcm': []})
r['+GLCM+GLRLM+GLSZM'] = run('C', roi, {'shape': [], 'firstorder': [], 'glcm': [], 'glrlm': [], 'glszm': []})
r['全纹理(glcm+glrlm+glszm+gldm+ngtdm)'] = run('D', roi, {'shape': [], 'firstorder': [], 'glcm': [], 'glrlm': [], 'glszm': [], 'gldm': [], 'ngtdm': []})

print('\n=== 汇总 ===', flush=True)
for k, v in r.items():
    print(f"{k}: {v:.1f}s", flush=True)
