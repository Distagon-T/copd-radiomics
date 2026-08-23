#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 aq_Pi10 等关键 AirQuant 特征的判别力"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv("radiomics_2026_05_features.csv")
aq = pd.read_csv("airquant_2026_05_aggregated.csv").rename(
    columns={"patient": "Patient_ID"})
lab = pd.read_csv("labels_2026_05.csv")
if "Patient_ID" in lab.columns:
    lab = lab.drop(columns=["Patient_ID"])
m = df.merge(lab.rename(columns={"patient_id": "PatientID"}), on="PatientID",
             how="inner").merge(aq, on="Patient_ID", how="left")
y = m["cvd_exacerbation_label"].values
for c in ["aq_Pi10", "aq_n_branches", "aq_stats_tortuosity_std",
          "aq_Gen3_Wall_Thickness_mm_mean", "aq_GenLe2_Wall_Thickness_mm_mean"]:
    x = pd.to_numeric(m[c], errors="coerce").values
    nn = ~np.isnan(x)
    print(f"{c}: n={int(nn.sum())} AUC={roc_auc_score(y[nn], x[nn]):.3f}")
