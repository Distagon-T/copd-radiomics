#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单患者冒烟测试：验证 vessel_advanced_features 在真实 lung_vessels 掩膜上可用"""
import os
import sys
import time
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_patient_radiomics_fast import vessel_advanced_features

SEG = "seg_results"
patient = "20130130_Anonymous_CT_1.2.840.113619.2.55.3.1678396440.5613.1359500545.33501186"
mask_path = os.path.join(SEG, f"{patient}_masks", "lung_vessels.nii.gz")
print("mask exists:", os.path.exists(mask_path))
img = sitk.ReadImage(mask_path)
spacing = img.GetSpacing()
arr = sitk.GetArrayFromImage(img) > 0
print("shape:", arr.shape, "spacing:", spacing, "vessel voxels:", int(arr.sum()))
t0 = time.time()
feats = vessel_advanced_features(arr, spacing)
print(f"耗时 {time.time()-t0:.1f}s")
for k, v in feats.items():
    print(f"  {k} = {v}")
