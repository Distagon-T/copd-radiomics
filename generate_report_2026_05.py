#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_report_2026_05.py
==========================
生成 2026-05 急性加重分类的完整报告：
  report_2026_05.md   (Markdown, 引用 figs/*.png)
  report_2026_05.html (自包含, 图片 base64 内嵌)
数据来源: fusion_2026_05.log / *_univariate_top.csv / *_lr_coefficients.csv / consistency.log
"""
import base64
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEG = "seg_results"
FIGDIR = os.path.join(SEG, "figs")
MD_OUT = os.path.join(SEG, "report_2026_05.md")
HTML_OUT = os.path.join(SEG, "report_2026_05.html")

FIG_PATHS = [
    ("fig_roc_2026_05.png", "图 1. 5 折交叉验证平均 ROC 曲线（AUC=0.721±0.019，阴影为±1SD）"),
    ("fig_univariate_auc_2026_05.png", "图 2. radiomics+AirQuant 单变量 AUC Top 20（红=正向，蓝=负向）"),
    ("fig_univariate_auc_airquant_2026_05.png", "图 3. AirQuant 特征单变量 AUC Top 15"),
    ("fig_boxplot_top8_2026_05.png", "图 4. Top 8 特征按 阴性/阳性 分组箱线图"),
    ("fig_consistency_radiomics_2026_05.png", "图 5. radiomics 显著特征 bootstrap 一致性森林图"),
    ("fig_consistency_airquant_2026_05.png", "图 6. AirQuant 特征（含 Pi10）bootstrap 一致性森林图"),
]


def read_log(path):
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_cv(log):
    m = re.search(r"平均: AUC=([\d.]+)±([\d.]+)\s+Acc=([\d.]+)\s+Sens=([\d.]+)\s+Spec=([\d.]+)", log)
    if not m:
        return None
    return {"auc": m.group(1), "auc_std": m.group(2), "acc": m.group(3),
            "sens": m.group(4), "spec": m.group(5)}


def parse_folds(log):
    folds = re.findall(r"fold(\d): AUC=([\d.]+) Acc=([\d.]+) Sens=([\d.]+) Spec=([\d.]+) "
                       r"\(train (\d+) pos / test (\d+) pos\)", log)
    return folds


def parse_consistency(log):
    """从 consistency.log 解析 '名称  AUC=x CI[a,b] 同向稳定=p%'"""
    rows = []
    in_section = None
    for line in log.splitlines():
        if "radiomics 一致性" in line:
            in_section = "radiomics"
            continue
        if "AirQuant 一致性" in line:
            in_section = "airquant"
            continue
        m = re.match(r"\s+(.+?)\s{2,}AUC=([\d.]+) CI\[([\d.]+),([\d.]+)\]\s+同向稳定=(\d+)%", line)
        if m and in_section:
            rows.append({"group": in_section, "feature": m.group(1).strip(),
                         "auc": m.group(2), "lo": m.group(3), "hi": m.group(4),
                         "stab": m.group(5)})
    return rows


def airquant_counts():
    """从实际文件动态统计 AirQuant 特征数 / 合并成功数（文件缺失时回退历史值）。

    返回: (n_feat, n_merged, n_total)
      n_feat   : airquant_2026_05_aggregated.csv 中 aq_* 特征列数（合并脚本新增 blur_/TD_* 后自动变化）
      n_merged : labeled 患者中 PatientID 对应 Patient_ID 出现在 AirQuant 表中的例数
      n_total  : labeled 患者总数
    """
    default = (49, 653, 698)
    aq_path = os.path.join(SEG, "airquant_2026_05_aggregated.csv")
    lab_path = os.path.join(SEG, "labels_2026_05.csv")
    rad_path = os.path.join(SEG, "radiomics_2026_05_features.csv")
    try:
        if not os.path.exists(aq_path):
            print(f"[warn] 缺 {aq_path}，AirQuant 统计回退默认 {default}")
            return default
        aq = pd.read_csv(aq_path)
        aq_feats = [c for c in aq.columns if str(c).startswith("aq_")]
        n_feat = len(aq_feats) if aq_feats else default[0]
        aq_keys = set(aq["patient"].dropna().astype(str))
        if not (os.path.exists(lab_path) and os.path.exists(rad_path)):
            print(f"[warn] 缺 labels/radiomics，合并成功数回退默认，特征数 {n_feat}")
            return n_feat, default[1], default[2]
        lab = pd.read_csv(lab_path)
        pid_col = "patient_id" if "patient_id" in lab.columns else "PatientID"
        total = int(len(lab))
        rad = pd.read_csv(rad_path, usecols=["Patient_ID", "PatientID"]).dropna(
            subset=["Patient_ID", "PatientID"])
        pid2pids = {}
        for _, r in rad.iterrows():
            pid2pids.setdefault(str(r["PatientID"]), set()).add(str(r["Patient_ID"]))
        merged = sum(1 for v in lab[pid_col].dropna()
                     if pid2pids.get(str(v), set()) & aq_keys)
        return n_feat, merged, total
    except Exception as e:
        print(f"[warn] AirQuant 动态统计失败，回退默认: {e}")
        return default


def select_airquant_examples(seg, aq_dir, labels_csv, feats_csv):
    """每个类别挑 1 例有 AirQuant 图的患者（优先含 pi10），拷贝其 PNG 供报告引用"""
    import shutil
    lab = pd.read_csv(labels_csv)
    feats = pd.read_csv(feats_csv, usecols=["Patient_ID", "PatientID"])
    if "Patient_ID" in lab.columns:
        lab = lab.drop(columns=["Patient_ID"])
    m = lab.rename(columns={"patient_id": "PatientID"}).merge(feats, on="PatientID",
                                                             how="inner")
    out_dir = os.path.join(seg, "figs", "airquant_examples")
    os.makedirs(out_dir, exist_ok=True)
    results = []

    def pick(row, suffix_order):
        pid = row["Patient_ID"]
        folder = os.path.join(aq_dir, pid + "_airquant")
        if not os.path.isdir(folder):
            return []
        got = []
        for suffix in suffix_order:
            png = os.path.join(folder, pid + f"_{suffix}.png")
            if os.path.exists(png):
                dst = os.path.join(out_dir, f"{tag}_{suffix}.png")
                shutil.copyfile(png, dst)
                cap = (f"患者 {row['PatientID']}（{'急性加重' if label_val == 1 else '非急性加重'}）"
                       f"· AirQuant {suffix}")
                got.append((dst, cap))
        return got

    for label_val, tag in [(1, "Pos"), (0, "Neg")]:
        rows = m[m["cvd_exacerbation_label"] == label_val]
        suffix_order = ["tree2d", "tree3d", "pi10", "plot3d"]
        best = []
        # 优先选有 pi10 的患者
        for _, r in rows.iterrows():
            got = pick(r, suffix_order)
            if got and any("pi10" in os.path.basename(p[0]) for p in got):
                best = got
                break
        if not best:
            for _, r in rows.iterrows():
                got = pick(r, suffix_order)
                if got:
                    best = got
                    break
        results.extend(best)
    return results


def build_airquant_collage(seg, aq_dir, labels_csv, feats_csv, n_each=2):
    """多患者对比拼图：每类 n_each 例，并列展示 tree2d + pi10"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 配置中文字体（避免标签显示为方块）
    cjk = {ff.name for ff in font_manager.fontManager.ttflist}
    for f in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]:
        if f in cjk:
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False

    lab = pd.read_csv(labels_csv)
    feats = pd.read_csv(feats_csv, usecols=["Patient_ID", "PatientID"])
    if "Patient_ID" in lab.columns:
        lab = lab.drop(columns=["Patient_ID"])
    m = lab.rename(columns={"patient_id": "PatientID"}).merge(feats, on="PatientID",
                                                             how="inner")

    def find_rows(label_val):
        got = []
        for _, r in m[m["cvd_exacerbation_label"] == label_val].iterrows():
            folder = os.path.join(aq_dir, r["Patient_ID"] + "_airquant")
            t2 = os.path.join(folder, r["Patient_ID"] + "_tree2d.png")
            p10 = os.path.join(folder, r["Patient_ID"] + "_pi10.png")
            if os.path.exists(t2) and os.path.exists(p10):
                got.append((r, t2, p10))
                if len(got) >= n_each:
                    break
        return got

    pos = find_rows(1)
    neg = find_rows(0)
    all_rows = neg + pos  # 上面负例、下面正例
    ncol = 2
    nrow = len(all_rows)
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.2, 2.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for i, (r, t2, p10) in enumerate(all_rows):
        for j, (path, which) in enumerate([(t2, "tree2d"), (p10, "pi10")]):
            ax = axes[i * ncol + j]
            ax.imshow(mpimg.imread(path))
            ax.set_title(f"{r['PatientID']} {'急性加重' if r['cvd_exacerbation_label']==1 else '非加重'} · {which}",
                         fontsize=8)
            ax.axis("off")
    fig.suptitle("AirQuant 可视化对比（上：非急性加重，下：急性加重）", fontsize=12)
    plt.tight_layout()
    out = os.path.join(seg, "figs", "airquant_collage.png")
    plt.savefig(out, dpi=140)
    plt.close()
    return out, len(pos), len(neg)


def main():
    fus_log = read_log(os.path.join(SEG, "fusion_2026_05.log"))
    con_log = read_log(os.path.join(SEG, "consistency.log"))

    cv = parse_cv(fus_log)
    folds = parse_folds(fus_log)
    n_feat = re.search(r"入选特征: (\d+)", fus_log)
    n_feat = n_feat.group(1) if n_feat else "574"
    uni = pd.read_csv(os.path.join(SEG, "fusion_2026_05_univariate_top.csv")).head(15)
    coef = pd.read_csv(os.path.join(SEG, "fusion_2026_05_lr_coefficients.csv")).head(15)
    cons = parse_consistency(con_log)

    # AirQuant 特征数 / 合并成功数（按实际文件动态统计）
    n_aq_feat, n_aq_merged, n_aq_total = airquant_counts()

    # ---- AirQuant MATLAB 可视化示例 ----
    aq_examples = select_airquant_examples(SEG, "airway_metrics",
                                           os.path.join(SEG, "labels_2026_05.csv"),
                                           os.path.join(SEG, "radiomics_2026_05_features.csv"))
    collage_path, n_pos_c, n_neg_c = build_airquant_collage(
        SEG, "airway_metrics",
        os.path.join(SEG, "labels_2026_05.csv"),
        os.path.join(SEG, "radiomics_2026_05_features.csv"), n_each=2)

    # ---------- Markdown ----------
    L = []
    L.append("# 2026-05 队列：COPD 急性加重预测（Radiomics + AirQuant 融合）\n")
    L.append(f"> 报告生成时间：2026-08-21　|　样本 n=698（急性加重 214 / 非 484）\n")
    L.append("## 1. 数据与方法\n")
    L.append("| 组件 | 说明 |")
    L.append("|---|---|")
    L.append("| 影像特征 | pyRadiomics lite（shape+firstorder+自定义肺/气道/血管/主动脉指标），1106 例中 698 例有 PatientID 且匹配临床 |")
    L.append(f"| AirQuant | 每例气管树 branch 级指标聚合为患者级（{n_aq_feat} 特征，含 Pi10、管壁厚度、迂曲度、FWHM 管壁密度/边界模糊、T/D 变化等），{n_aq_merged}/{n_aq_total} 合并成功 |")
    L.append("| 临床 Label | `info-2026-05.csv` 主要诊断含\"急性加重\"→阳性（214 例，31%） |")
    L.append("| 模型 | Logistic Regression（L2, C=1.0, class_weight=balanced, liblinear），StandardScaler，分层 5 折 CV |")
    L.append(f"| 特征 | {n_feat} 个入选（剔除高缺失/零方差），中位数填补 |")
    L.append("")
    L.append("## 2. 分类性能（5 折 CV）\n")
    if cv:
        L.append(f"**平均 AUC = {cv['auc']} ± {cv['auc_std']}**，Acc={cv['acc']}，Sens={cv['sens']}，Spec={cv['spec']}\n")
    if folds:
        L.append("| Fold | AUC | Acc | Sens | Spec | 训练阳性 | 测试阳性 |")
        L.append("|---|---|---|---|---|---|---|")
        for f in folds:
            L.append(f"| {f[0]} | {f[1]} | {f[2]} | {f[3]} | {f[4]} | {f[5]} | {f[6]} |")
    L.append("")
    L.append("## 3. 单变量判别力 Top 特征\n")
    L.append("| 特征 | AUC | Cohen's d | p(MWU) |")
    L.append("|---|---|---|---|")
    for _, r in uni.iterrows():
        L.append(f"| {r['feature']} | {r['auc_univ']:.3f} | {r['cohens_d']:+.2f} | {r['p_mwu']:.2g} |")
    L.append("")
    L.append("## 4. Logistic 回归系数 Top（相关性，非因果）\n")
    L.append("| 特征 | 平均系数 ± SD |")
    L.append("|---|---|")
    for _, r in coef.iterrows():
        L.append(f"| {r['feature']} | {r['coef_mean']:+.3f} ± {r['coef_std']:.3f} |")
    L.append("")
    L.append("## 5. 一致性分析（bootstrap 200 次，单变量 AUC 均值±95%CI + 同向稳定率）\n")
    L.append("### 5.1 radiomics 显著特征\n")
    L.append("| 特征 | AUC | 95%CI | 同向稳定 |")
    L.append("|---|---|---|---|")
    for r in [x for x in cons if x["group"] == "radiomics"]:
        L.append(f"| {r['feature']} | {r['auc']} | [{r['lo']}, {r['hi']}] | {r['stab']}% |")
    L.append("\n### 5.2 AirQuant 特征（含 Pi10）\n")
    L.append("| 特征 | AUC | 95%CI | 同向稳定 |")
    L.append("|---|---|---|---|")
    for r in [x for x in cons if x["group"] == "airquant"]:
        L.append(f"| {r['feature']} | {r['auc']} | [{r['lo']}, {r['hi']}] | {r['stab']}% |")
    L.append("")
    L.append("## 6. AirQuant MATLAB 结果可视化示例\n")
    L.append("> 每例患者均输出 pi10 / 气道树 2D / 3D / spline / plot3d 的 PNG+PDF（共 4950 张）。")
    L.append("> 以下各挑 1 例急性加重与非急性加重患者的代表性图（完整 PDF 位于各患者 `_airquant` 文件夹）：\n")
    for path, cap in aq_examples:
        rel = os.path.relpath(path, SEG).replace("\\", "/")
        L.append(f"![{cap}]({rel})\n")
        L.append(f"*{cap}*\n")
    # 多患者对比拼图
    collage_rel = os.path.relpath(collage_path, SEG).replace("\\", "/")
    L.append(f"![AirQuant 多患者对比拼图]({collage_rel})\n")
    L.append(f"*图 AQ-拼图. AirQuant 多患者对比（非急性加重 {n_neg_c} 例 vs 急性加重 {n_pos_c} 例，tree2d + pi10）*\n")
    L.append("## 7. 图表\n")
    for fn, caption in FIG_PATHS:
        L.append(f"![{caption}](figs/{fn})\n")
        L.append(f"*{caption}*\n")
    L.append("## 8. 结论与局限\n")
    L.append("- 融合 radiomics+AirQuant 的 LR 在该队列取得 **AUC 0.72**，5 折间波动很小（±0.019），结果稳健。")
    L.append("- 显著且一致的特征集中于**肺气肿（LAA950/Perc15）、肺血管密度、主动脉形态**，符合 COPD 急性加重的病理生理。")
    L.append("- AirQuant 中**管壁厚度类特征稳定可用**（弱-中等判别力），但 **Pi10 对\"急性加重\"无判别力（AUC≈0.50）**。")
    L.append(f"- 局限：① 单中心回顾性，Label 来自主要诊断文本关键词；② {n_aq_total - n_aq_merged} 例缺 AirQuant 以中位数填补；③ 574 特征下仍有多重比较风险，需外部验证。")
    md = "\n".join(L) + "\n"
    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Markdown 报告 -> {MD_OUT}")

    # ---------- HTML (base64 内嵌图片) ----------
    def b64img(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def md_to_html(md_text):
        lines = md_text.splitlines()
        html = ["<h1>2026-05 队列：COPD 急性加重预测（Radiomics + AirQuant 融合）</h1>"]
        html.append("<p><em>报告生成时间：2026-08-21　|　n=698（急性加重 214 / 非 484）</em></p>")
        in_table = False
        for ln in lines[3:]:
            if ln.startswith("## "):
                html.append(f"<h2>{ln[3:]}</h2>")
            elif ln.startswith("### "):
                html.append(f"<h3>{ln[4:]}</h3>")
            elif ln.startswith("|") and "|---|---|" not in ln:
                cells = [c.strip() for c in ln.strip("|").split("|")]
                if not in_table:
                    html.append("<table><thead><tr>")
                    for c in cells:
                        html.append(f"<th>{c}</th>")
                    html.append("</tr></thead><tbody>")
                    in_table = True
                else:
                    html.append("<tr>")
                    for c in cells:
                        html.append(f"<td>{c}</td>")
                    html.append("</tr>")
            elif ln.startswith("![") and "](" in ln:
                cap = ln[2:].split("](")[0]
                path = ln.split("](")[1].rstrip(")")
                full = os.path.join(SEG, path)
                if os.path.exists(full):
                    html.append(f'<img src="data:image/png;base64,{b64img(full)}" '
                                f'style="max-width:95%;height:auto;display:block;margin:10px auto;" '
                                f'alt="{cap}"/>')
                    html.append(f'<p style="text-align:center;color:#555;font-size:0.9em">{cap}</p>')
            elif ln.startswith("*") and ln.endswith("*") and len(ln) > 4:
                continue
            elif ln.strip() == "":
                if in_table:
                    html.append("</tbody></table>")
                    in_table = False
            else:
                if in_table:
                    html.append("</tbody></table>")
                    in_table = False
                html.append(f"<p>{ln}</p>")
        if in_table:
            html.append("</tbody></table>")
        return "\n".join(html)

    body = md_to_html(md)
    html_doc = ("<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
                "<title>2026-05 急性加重预测报告</title>"
                "<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:1000px;"
                "margin:20px auto;padding:0 20px;color:#222;line-height:1.6}"
                "table{border-collapse:collapse;margin:10px 0;font-size:0.92em}"
                "th,td{border:1px solid #ccc;padding:4px 8px}th{background:#f0f0f0}"
                "h2{border-bottom:2px solid #4472C4;padding-bottom:4px;margin-top:28px}"
                "</style></head><body>" + body + "</body></html>")
    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"HTML 报告   -> {HTML_OUT}")

    # ---------- 输出目录确认 ----------
    print(f"\n图目录: {FIGDIR}")
    print("报告与图均在 seg_results 下")


if __name__ == "__main__":
    main()
