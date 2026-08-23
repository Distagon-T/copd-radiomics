#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_labels.py
==================
规则引擎自动生成「心血管急性加重」分类 Label（0=无, 1=有）。

规则（可调，见 RULES 配置）：
  1) 文本规则：主要诊断 + 其他诊断 + 主诉 命中"急性心血管事件"关键词
     （注意：慢性合并症词如"房颤/心功能不全"单独出现不算急性，
       必须与"急性/加重/发作/衰竭"等修饰词搭配才算）
  2) 检验规则：肌钙蛋白 I 显著升高（> TNI_THRESHOLD）
  3) 检验规则：NT-ProBNP 异常升高（按年龄分层阈值）

用法：
  python generate_labels.py \
      --xlsx Asthma.xlsx \\
      --csv  radiomics_all_patients.csv \\
      --out  patient_cvd_labels.csv \\
      --report label_diagnostics.txt
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- 可调规则配置 ----------
# 急性心血管事件关键词（强规则：直接命中即阳性）
ACUTE_KEYWORDS = [
    "急性心力衰竭", "急性心衰", "心源性休克", "急性心肌梗死", "急性心梗",
    "急性冠脉", "急性冠脉综合征", "恶性心律失常", "心肺复苏", "室颤",
    "心室颤动", "急性肺水肿", "心搏骤停", "心脏骤停",
]
# 慢性心血管词：必须搭配修饰词才算急性
CHRONIC_KEYWORDS = ["心力衰竭", "心衰", "心肌梗死", "心梗", "心功能不全",
                    "房颤", "心房颤动", "心肌缺血", "冠心病", "心绞痛"]
ACUTE_MODIFIERS = ["急性", "加重", "急性发作", "衰竭", "发作", "新发", "突发"]

# 合并症扩展词（comorbidity 模式：出现在任何诊断字段即算心血管合并症）
COMORBIDITY_EXTRA = ["心律失常", "肺源性心脏病", "肺心病", "心包积液",
                     "心肌病", "瓣膜病", "心脏瓣膜", "心内膜炎", "主动脉夹层",
                     "心肌炎", "心包炎", "二尖瓣", "主动脉瓣", "心脏扩大", "心影增大"]

# 肌钙蛋白 I 阈值 (ng/mL)：0.047 ng/mL = 47 ng/L（常见 99 百分位参考上限）
TNI_THRESHOLD = 0.047
TNI_UNIT_POSSIBLE = {"ng/mL": 1.0, "ng/L": 0.001, "μg/L": 1.0, "ug/L": 1.0}
# NT-ProBNP 按年龄分层 (pg/mL)
BNP_THRESH_AGE = [(50, 450.0), (75, 900.0), (200, 1800.0)]


def parse_float(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        t = str(v).strip().replace(",", "")
        if t in ("", "/", "无", "nan", "None", "<", ">"):
            return np.nan
        return float(re.sub(r"[<>≤≥]", "", t))
    except (TypeError, ValueError):
        return np.nan


def rule_text_hit(row):
    """文本规则：返回 (hit, 命中详情)"""
    text = " ".join(str(row.get(c, "")) for c in ["主要诊断", "其他诊断", "主诉"])
    # 强规则：急性词直接命中
    for kw in ACUTE_KEYWORDS:
        if kw in text:
            return True, f"急性关键词: {kw}"
    # 慢性词 + 修饰词
    for kw in CHRONIC_KEYWORDS:
        if kw in text:
            for mod in ACUTE_MODIFIERS:
                # 修饰词出现在慢性词前后 6 字符内
                idx = text.find(kw)
                if idx >= 0:
                    window = text[max(0, idx - 6): idx + len(kw) + 6]
                    if mod in window:
                        return True, f"慢性词+修饰: {kw}+{mod}"
    return False, ""


def rule_text_comorbidity(row, diag_cols):
    """合并症模式：任何心血管词出现在任何诊断字段即命中"""
    text = " ".join(str(row.get(c, "")) for c in diag_cols)
    terms = sorted(set(ACUTE_KEYWORDS + CHRONIC_KEYWORDS + COMORBIDITY_EXTRA),
                   key=len, reverse=True)
    for kw in terms:
        if kw in text:
            return True, f"合并症: {kw}"
    return False, ""


def rule_tni_hit(row):
    v = parse_float(row.get("肌钙蛋白ITnI测定-定量结果"))
    if np.isnan(v):
        return False, "", np.nan
    # 单位换算为 ng/mL
    unit = str(row.get("肌钙蛋白ITnI测定-单位", "")).strip()
    factor = TNI_UNIT_POSSIBLE.get(unit, 1.0)
    v_ngml = v * factor
    hit = v_ngml > TNI_THRESHOLD
    note = f"肌钙蛋白升高({v_ngml:.3f}ng/mL)" if hit else ""
    return hit, note, v_ngml


def rule_bnp_hit(row):
    v = parse_float(row.get("N端_B型钠尿肽前体NT_ProBNP测定-定量结果"))
    if np.isnan(v):
        return False, "", np.nan
    age = parse_float(row.get("年龄 (岁)"))
    thresh = 450.0
    if not np.isnan(age):
        for agelim, t in BNP_THRESH_AGE:
            if age <= agelim:
                thresh = t
                break
    hit = v > thresh
    note = f"NT-ProBNP升高({v:.0f}>{thresh:.0f}pg/mL)" if hit else ""
    return hit, note, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--mode", choices=["acute", "comorbidity"], default="acute",
                    help="acute=急性心血管事件(严格); comorbidity=心血管合并症(宽松)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    xlsx_path = Path(args.xlsx)
    csv_path = Path(args.csv)
    if not xlsx_path.exists():
        sys.exit(f"找不到 {xlsx_path}")
    if not csv_path.exists():
        sys.exit(f"找不到 {csv_path}")
    def_out = csv_path.parent / f"patient_cvd_labels_{args.mode}.csv"
    out_path = Path(args.out) if args.out else def_out
    def_rep = csv_path.parent / f"label_diagnostics_{args.mode}.txt"
    report_path = Path(args.report) if args.report else def_rep

    # ---- 读取 ----
    cache_path = csv_path.parent / "clinical_80_cache.pkl"
    if cache_path.exists():
        print(f"[使用缓存 {cache_path.name}]")
        clin = pd.read_pickle(cache_path)
    else:
        clin = pd.read_excel(xlsx_path)
        print("[缓存不存在，直接读 xlsx；建议先运行 cache_clinical_subset.py]")
    # 自动找 ID 列
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from merge_clinical_radiomics import find_id_column, normalize_id
    clin_id = find_id_column(clin.columns)
    if clin_id is None:
        sys.exit("xlsx 找不到患者 ID 列")
    clin[clin_id] = normalize_id(clin[clin_id])

    radi = pd.read_csv(csv_path, dtype={"PatientID": str})
    radi_id = find_id_column(radi.columns)
    if radi_id is None:
        sys.exit("CSV 找不到 PatientID 列")
    radi[radi_id] = normalize_id(radi[radi_id])

    # ---- 只看有 radiomics 的 80 人 ----
    sub = clin[clin[clin_id].isin(radi[radi_id])].copy()
    print(f"匹配 radiomics 患者: {len(sub)}")

    # 诊断类字段（comorbidity 模式用）
    diag_cols = [c for c in clin.columns if "诊断" in str(c)]

    # ---- 逐条规则 ----
    rows = []
    for _, r in sub.iterrows():
        if args.mode == "comorbidity":
            t_hit, t_note = rule_text_comorbidity(r, diag_cols)
        else:
            t_hit, t_note = rule_text_hit(r)
        i_hit, i_note, tni = rule_tni_hit(r)
        b_hit, b_note, bnp = rule_bnp_hit(r)
        label = 1 if (t_hit or i_hit or b_hit) else 0
        rows.append({
            "patient_id": r[clin_id],
            "cvd_exacerbation_label": label,
            "rule_text": 1 if t_hit else 0,
            "rule_tni": 1 if i_hit else 0,
            "rule_bnp": 1 if b_hit else 0,
            "hit_note": " | ".join(x for x in [t_note, i_note, b_note] if x),
            "tni_value": tni,
            "bnp_value": bnp,
            "age": parse_float(r.get("年龄 (岁)")),
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # ---- 诊断报告 ----
    mode_title = "心血管急性加重" if args.mode == "acute" else "心血管合并症"
    lines = []
    lines.append(f"=== {mode_title} Label 诊断报告 (mode={args.mode}) ===")
    lines.append(f"总患者数: {len(out)}")
    lines.append(f"阳性(1): {out['cvd_exacerbation_label'].sum()}")
    lines.append(f"阴性(0): {(out['cvd_exacerbation_label']==0).sum()}")
    lines.append("")
    lines.append("--- 各规则命中 ---")
    lines.append(f"文本规则: {out['rule_text'].sum()} 例")
    lines.append(f"肌钙蛋白规则: {out['rule_tni'].sum()} 例")
    lines.append(f"NT-ProBNP 规则: {out['rule_bnp'].sum()} 例")
    lines.append(f"多规则同时命中: {(out[['rule_text','rule_tni','rule_bnp']].sum(axis=1)>1).sum()} 例")
    lines.append("")
    lines.append("--- 阳性患者明细 ---")
    pos = out[out['cvd_exacerbation_label'] == 1]
    for _, r in pos.iterrows():
        lines.append(f"  {r['patient_id']} | {r['hit_note']} | TnI={r['tni_value']} BNP={r['bnp_value']} 年龄={r['age']}")
    lines.append("")
    lines.append("--- 检验字段缺失统计 (80 人) ---")
    lines.append(f"肌钙蛋白 缺失: {(out['tni_value'].isna()).sum()}/80")
    lines.append(f"NT-ProBNP 缺失: {(out['bnp_value'].isna()).sum()}/80")
    lines.append(f"年龄 缺失: {(out['age'].isna()).sum()}/80")
    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nLabel CSV: {out_path}")
    print(f"诊断报告: {report_path}")


if __name__ == "__main__":
    main()
