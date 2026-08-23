#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fuse_vessel_into_reports.py
===========================
把新增肺血管高级特征（Vessel_*）分析融入三个报告：
  report_bcos_phenotype / report_bronch_hemoptysis / report_copd_ae_cause
在「结论」前插入新章节（含单变量表 + 提升对比表），重生成 HTML。
"""
import base64
import os
import re

import numpy as np
import pandas as pd

SEG = "seg_results"
UNI_CSV = os.path.join(SEG, "vessel_features_univariate_ALL.csv")
BOOST_CSV = os.path.join(SEG, "vessel_boost_comparison.csv")

TASKS = [
    ("report_bcos_phenotype_2026_05.md", "BCOS 表型: BCOS vs PureCOPD",
     "BCOS（COPD合并支扩）血管网显著更复杂：分支密度/分叉点/分形维度更高，慢性炎症血管增生重构。"),
    ("report_bronch_hemoptysis_2026_05.md", "支扩: 咯血 vs 无咯血",
     "咯血组血管网显著简化：分支密度/分叉点/迂曲度/分形维度更低——与咯血型支扩「破坏性/侵蚀性」表型一致（管壁破坏同时损毁血管床）。"),
    ("report_copd_ae_cause_2026_05.md", "急性COPD: 感染型 vs 非感染型",
     "感染型加重小血管占比（BV5/BV10）更低——炎症充血使大血管占比上升。"),
]


def md_to_html(md_text, b64img):
    lines = md_text.splitlines()
    html = ["<h1>影像表型分析报告</h1>"]
    in_table = False
    for ln in lines:
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
                html.append("</tr></thead><tbody>"); in_table = True
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
        elif ln.strip() == "":
            if in_table:
                html.append("</tbody></table>"); in_table = False
        else:
            if in_table:
                html.append("</tbody></table>"); in_table = False
            html.append(f"<p>{ln}</p>")
    if in_table:
        html.append("</tbody></table>")
    return "\n".join(html)


def main():
    uni = pd.read_csv(UNI_CSV)
    boost = pd.read_csv(BOOST_CSV)
    uni["auc_dev"] = (uni["auc_univ"] - 0.5).abs()

    for fname, task, note in TASKS:
        md_path = os.path.join(SEG, fname)
        html_path = md_path.replace(".md", ".html")
        if not os.path.exists(md_path):
            print(f"[warn] 缺 {fname}，跳过"); continue

        u = uni[uni["task"] == task].sort_values("auc_dev", ascending=False).head(10)
        b = boost[boost["task"] == task]

        L = []
        L.append("## 附. 肺血管高级特征（Vessel_* 新特征）分析\n")
        L.append("> 新增 11 个肺血管高级特征（分形维度 / BV5-BV10 / 中心线迂曲度与分支密度），"
                 "基于 `lung_vessels` 掩膜计算。\n")
        L.append("### 单变量判别力（Vessel_* 特征）\n")
        L.append("| 特征 | AUC | 有效AUC | Cohen's d | p(MWU) |")
        L.append("|---|---|---|---|---|")
        for _, r in u.iterrows():
            eff = max(r["auc_univ"], 1 - r["auc_univ"])
            L.append(f"| {r['feature']} | {r['auc_univ']:.3f} | {eff:.3f} | "
                     f"{r['cohens_d']:+.2f} | {r['p_mwu']:.2g} |")
        L.append("")
        L.append("### 加入 Vessel_* 特征前后判别力对比\n")
        L.append("| 模型 | 特征数 | 5折CV AUC | bootstrap 均值 | 95%CI | 稳定性 |")
        L.append("|---|---|---|---|---|---|")
        for _, r in b.iterrows():
            L.append(f"| {r['model']} | {r['n_feat']} | {r['cv_auc']:.3f} | "
                     f"{r['boot_mean']:.3f} | {r['ci_lo']:.3f}–{r['ci_hi']:.3f} | "
                     f"{r['stability']:.0%} |")
        L.append("")
        L.append(f"- **解读**：{note}")
        L.append("")
        section = "\n".join(L)

        with open(md_path, "r", encoding="utf-8") as f:
            md = f.read()
        m = re.search(r"^## (\d+)\. 结论", md, re.M)
        if m:
            n = int(m.group(1))
            md = re.sub(r"^## \d+\. 结论", section + f"\n## {n + 1}. 结论",
                        md, count=1, flags=re.M)
            print(f"[{fname}] 在 结论(n={n}) 前插入，结论改为 {n+1}")
        else:
            md = md.rstrip() + "\n\n" + section
            print(f"[{fname}] 未找到结论，追加到末尾")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)

        def b64img(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(md_to_html(md, b64img))
        print(f"  已更新 {fname} / {os.path.basename(html_path)}")


if __name__ == "__main__":
    main()
