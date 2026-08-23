#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_clinical_radiomics.py
===========================
把 Asthma.xlsx 中的临床信息按 PatientID 合并到 Radiomics CSV，
并输出"临床变量可用性报告"（哪些列可作为自变量 / 应变量）。

用法：
  python merge_clinical_radiomics.py \
      --xlsx  D:\\copd-radiomics\\Asthma.xlsx \
      --csv   E:\\DICOM\\2026-04-seg-part1\\radiomics_all_patients.csv \
      --out   E:\\DICOM\\2026-04-seg-part1\\radiomics_clinical_merged.csv \
      --report E:\\DICOM\\2026-04-seg-part1\\clinical_variable_report.csv

说明：
  1. 脚本自动在 xlsx 全表中按关键字挑选临床列（无需手工列名），
     只读需要的列以加速（xlsx 有 1366 列）。
  2. 匹配键自动识别：CSV 侧自动找 PatientID 列，xlsx 侧自动找患者 ID 列
     （支持 patientid / 患者id / 住院号 / 病历号 等变体，大小写不敏感），
     再按字符串精确匹配（自动去前导零）。
  3. 数据清洗：吸烟量/饮酒量等文本数值（如 "1包/天"）解析为数字；
     检验项取 "-定量结果" 列并转为 float。
  4. 生成变量可用性报告：缺失率、唯一值数、类型、建议角色。
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================================
# 临床列挑选规则：列名命中任一关键字则入选
# =========================================================================
KEEP_PATTERNS = [
    # 主键与提取标记
    "患者id", "已提取CT",
    # 人口学
    "性别", "年龄", "身高", "体重", "血型",
    # 就诊与住院
    "入院次数", "入院日期", "出院日期", "住院天数",
    "入院病区", "出院病区", "医疗付款方式",
    # 生命体征
    "收缩压", "舒张压", "脉搏", "心率", "呼吸",
    # 诊断（应变量候选核心）
    "主要诊断", "其他诊断", "病情评估",
    # 生活习惯（自变量候选核心）
    "是否吸烟", "烟龄", "吸烟量", "戒烟时长",
    "是否饮酒", "饮酒量", "戒酒时长", "饮酒类型",
    # 护理/评估
    "特级护理天数", "一级护理天数", "二级护理天数", "三级护理天数",
    "入院日常生活能力评定表得分", "出院日常生活能力评定表得分",
    "跌倒坠床风险评估得分", "风险BRADEN量表评分", "压疮",
    # 肺功能（注意：本队列可能全缺失，但仍纳入报告）
    "FEV", "FVC", "PEF",
    # 哮喘核心检验（嗜酸粒细胞等）
    "嗜酸性粒细胞", "嗜碱性粒细胞",
    "血红蛋白", "白细胞", "淋巴细胞", "中性粒细胞", "单核细胞",
    "红细胞比积", "血小板", "C反应蛋白", "免疫球蛋白E", "IgE",
    "降钙素原", "D-二聚体", "凝血酶原时间", "纤维蛋白原",
    "血糖", "血清渗透压", "尿素", "肌酐", "尿酸",
    "谷丙转氨酶", "谷草转氨酶", "总胆红素", "白蛋白", "总蛋白",
]

# 检验类列固定使用后缀：定量结果优先
QUANT_SUFFIX = "-定量结果"

# ID 列自动识别：按优先级匹配
# 注意：CSV 里常同时有 Patient_ID（文件夹名）和 PatientID（DICOM 真 ID），
# 因此"无下划线的 patientid/患者id"优先级最高。
ID_PRIORITY_HIGH = ["patientid", "患者id", "病人id", "患者编号", "住院号", "病历号"]
ID_PRIORITY_LOW = ["patient_id", "patient id"]
ID_CONTAINS = ["patient", "患者id", "病人id", "住院号", "病历号", "编号"]


def find_id_column(columns):
    """在列名列表中自动定位患者 ID 列（大小写/空格不敏感）。
    优先级：无下划线精确名 > 带下划线变体 > 含关键字。
    返回列名；找不到返回 None。"""
    cols = [str(c) for c in columns]
    # 保留下划线的原始名（仅去空格、小写）
    raw = [c.lower().replace(" ", "") for c in cols]
    # 完全去下划线的规范化名
    norm = [r.replace("_", "") for r in raw]

    # 1) 最高优先级：原始名精确等于 patientid / 患者id 等（无下划线命名）
    for i, r in enumerate(raw):
        if r in ID_PRIORITY_HIGH:
            return cols[i]
    # 2) 次级：带下划线的 patient_id 等变体
    for i, r in enumerate(raw):
        if r in ID_PRIORITY_LOW:
            return cols[i]
    # 3) 兜底：规范化后包含关键字
    for i, n in enumerate(norm):
        if any(k in n for k in ID_CONTAINS):
            return cols[i]
    return None


def normalize_id(series: pd.Series) -> pd.Series:
    """ID 统一化：转字符串、去空白、去前导零、去掉数值转字符串产生的 .0。"""
    return (series.astype(str).str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.lstrip("0"))


def pick_xlsx_columns(xlsx_path: Path) -> list:
    """只读表头，按关键字（大小写不敏感）挑出需要的列名（保持原顺序）。"""
    header = pd.read_excel(xlsx_path, nrows=0)
    picked = []
    for c in header.columns:
        cstr = str(c).lower().replace(" ", "")
        if any(p.lower().replace(" ", "") in cstr for p in KEEP_PATTERNS):
            picked.append(c)
    return picked


def parse_text_amount(s) -> float:
    """解析 '1包/天' / '2两/日' / '3/日' / '/' -> 数值；无法解析返回 NaN。"""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    t = str(s).strip()
    if t in ("/", "", "无", "None", "nan"):
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1))
    return np.nan


def to_float(v):
    """通用转 float；非数值返回 NaN。"""
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        t = str(v).strip()
        if t in ("", "/", "无", "nan", "None", "<", ">"):
            return np.nan
        t = re.sub(r"[<>]", "", t)
        return float(t)
    except (TypeError, ValueError):
        return np.nan


def clean_clinical(df: pd.DataFrame) -> pd.DataFrame:
    """清洗临床子表：解析数值、简化命名。"""
    out = df.copy()

    # 吸烟量 / 饮酒量：文本 -> 数字（新列 _num）
    for col, new in [("吸烟量", "吸烟量_包每日"), ("饮酒量", "饮酒量_两每日")]:
        if col in out.columns:
            out[new] = out[col].apply(parse_text_amount)

    # 所有 "-定量结果" 列强制转 float
    for c in out.columns:
        if c.endswith(QUANT_SUFFIX):
            out[c] = out[c].apply(to_float)

    # 年龄 / 住院天数 / 护理天数 转数值
    for c in ["年龄 (岁)", "住院天数", "入院次数",
              "特级护理天数", "一级护理天数", "二级护理天数", "三级护理天数"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def classify_role(nonnull, nunique, dtype_kind, colname, id_colname=None):
    """
    依据缺失率/唯一值数/类型，给出变量角色建议。
    返回 (role, note)
    """
    missing_rate = 1 - nonnull
    if missing_rate > 0.5:
        return "排除", f"缺失率 {missing_rate:.0%} > 50%"
    if missing_rate > 0.2:
        return "谨慎使用", f"缺失率 {missing_rate:.0%} (20%-50%)"
    if colname == id_colname:
        return "主键", "匹配键，不参与建模"
    if nunique == 1:
        return "排除", "唯一值=1（常数）"
    if dtype_kind in "if" and nunique >= 5:
        return "连续自变量", "数值型"
    if dtype_kind in "if" and nunique < 5:
        return "有序/分类自变量", f"数值但唯一值仅 {nunique}"
    if dtype_kind == "O":
        if nunique == 2:
            return "二分类变量", "可作自变量或应变量"
        if nunique <= 10:
            return "分类自变量", f"唯一值 {nunique}，需 one-hot"
        return "文本变量", "需人工归类或词向量化"
    if dtype_kind == "M":
        return "日期变量", "可衍生住院季节/时段"
    return "待定", ""


def main():
    ap = argparse.ArgumentParser(description="合并 Asthma.xlsx 临床信息到 Radiomics CSV 并输出变量报告")
    ap.add_argument("--xlsx", required=True, help="Asthma.xlsx 路径")
    ap.add_argument("--csv", required=True, help="Radiomics 汇总 CSV 路径")
    ap.add_argument("--out", default=None, help="合并结果输出 CSV（默认 <csv同目录>/radiomics_clinical_merged.csv）")
    ap.add_argument("--report", default=None, help="变量可用性报告 CSV（默认 <csv同目录>/clinical_variable_report.csv）")
    args = ap.parse_args()

    xlsx_path = Path(args.xlsx)
    csv_path = Path(args.csv)
    if not xlsx_path.exists():
        sys.exit(f"[错误] 找不到 {xlsx_path}")
    if not csv_path.exists():
        sys.exit(f"[错误] 找不到 {csv_path}")

    out_path = Path(args.out) if args.out else csv_path.parent / "radiomics_clinical_merged.csv"
    report_path = Path(args.report) if args.report else csv_path.parent / "clinical_variable_report.csv"

    # ---------- 1. 读取 radiomics CSV（自动识别 PatientID 列） ----------
    radi = pd.read_csv(csv_path)
    radi_id_col = find_id_column(radi.columns)
    if radi_id_col is None:
        sys.exit(f"[错误] CSV 中未找到 PatientID 列，现有列前10个: {list(radi.columns)[:10]}")
    radi[radi_id_col] = normalize_id(radi[radi_id_col])
    print(f"[1] Radiomics CSV: {len(radi)} 行, {len(radi.columns)} 列, ID 列=[{radi_id_col}]")

    # ---------- 2. 挑选并读取 xlsx 临床列（自动识别患者 ID 列） ----------
    usecols = pick_xlsx_columns(xlsx_path)
    print(f"[2] xlsx 命中 {len(usecols)} 个临床列")
    clin = pd.read_excel(xlsx_path, usecols=usecols)
    clin_id_col = find_id_column(clin.columns)
    if clin_id_col is None:
        sys.exit(f"[错误] xlsx 中未找到患者 ID 列，现有列前10个: {list(clin.columns)[:10]}")
    clin[clin_id_col] = normalize_id(clin[clin_id_col])
    print(f"    xlsx 全表: {len(clin)} 行, ID 列=[{clin_id_col}]")

    # ---------- 3. 按 PatientID 匹配 ----------
    matched = clin[clin[clin_id_col].isin(radi[radi_id_col])].copy()
    print(f"[3] xlsx 中匹配到 {len(matched)} 行 (Radiomics 共 {len(radi)} 行)")
    unmatched_radi = radi[~radi[radi_id_col].isin(clin[clin_id_col])]
    if len(unmatched_radi):
        print(f"    [警告] Radiomics 有 {len(unmatched_radi)} 行在 xlsx 中未匹配:")
        print("          ", unmatched_radi[radi_id_col].tolist()[:10])

    # ---------- 4. 清洗 ----------
    clin_clean = clean_clinical(matched)

    # ---------- 5. 合并 ----------
    merged = radi.merge(clin_clean, left_on=radi_id_col, right_on=clin_id_col, how="left")
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[5] 合并结果已写出: {out_path}  ({len(merged)} 行 x {len(merged.columns)} 列)")

    # ---------- 6. 变量可用性报告 ----------
    rows = []
    for c in clin_clean.columns:
        s = clin_clean[c]
        nonnull = int(s.notna().sum())
        nunique = int(s.nunique(dropna=True))
        role, note = classify_role(nonnull, nunique, s.dtype.kind, c,
                                   id_colname=clin_id_col)
        rows.append({
            "列名": c,
            "非空数": nonnull,
            "总行数": len(clin_clean),
            "缺失率": round(1 - nonnull / len(clin_clean), 3),
            "唯一值数": nunique,
            "类型": str(s.dtype),
            "建议角色": role,
            "备注": note,
        })
    report = pd.DataFrame(rows)
    # 建议角色排序：把可用的放前面
    role_order = {"主键": 0, "二分类变量": 1, "连续自变量": 2, "有序/分类自变量": 3,
                  "分类自变量": 4, "文本变量": 5, "日期变量": 6, "谨慎使用": 7, "排除": 8, "待定": 9}
    report["_order"] = report["建议角色"].map(role_order).fillna(9)
    report = report.sort_values(["_order", "缺失率"]).drop(columns="_order")
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"[6] 变量报告已写出: {report_path}")

    # ---------- 7. 终端摘要 ----------
    print("\n================ 变量可用性摘要 ================")
    usable = report[~report["建议角色"].isin(["排除"])]
    print(usable.to_string(index=False, max_colwidth=30))
    print("\n================ 建模建议 ================")
    print("【应变量候选】(哮喘队列, J45.x) 建议优先:")
    print("  1. 哮喘急性发作二分类: 由 '主要诊断' 文本派生 (含'急性发作'=1, 否则=0)")
    print("  2. 哮喘严重度分组: 主要诊断-ICD码 (J45.903 vs 其他)")
    print("  3. 连续终点: 住院天数 / 二级护理天数")
    print("【自变量候选】: 年龄, 性别, 吸烟量(包/日), 饮酒量(两/日),")
    print("  收缩压/舒张压/脉搏, 嗜酸性粒细胞(计数/百分比), 血红蛋白, 白细胞, 淋巴细胞比例等")
    print("  注: 肺功能(FEV1/PEF)在本队列 80 人中基本全缺失, 不可用;")
    print("      Radiomics 特征本身为自变量主体, 临床变量作为协变量。")
    print("=================================================")


if __name__ == "__main__":
    main()
