#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把统一可解释影像表型总结（两任务 clinical6）融合进 report_copd_ae_cause_2026_05.md/.html"""
import base64
import os
import sys

SEG = "seg_results"
MD_OUT = os.path.join(SEG, "report_copd_ae_cause_2026_05.md")
HTML_OUT = os.path.join(SEG, "report_copd_ae_cause_2026_05.html")


def md_to_html(md_text, b64img):
    lines = md_text.splitlines()
    html = ["<h1>急性期 COPD：感染型 vs 非感染型加重（Radiomics + AirQuant）</h1>"]
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
    return "\n".join(html)


SUMMARY = """## 8. 统一可解释影像表型总结（两任务 clinical6 模型）

> 两个独立临床任务验证同一结论：假说驱动的 6 特征精简模型均优于 600+ 维全特征模型，且特征方向符合病理机制。

### 8.1 模型总览

| 任务 | clinical6 特征数 | 5折CV AUC | bootstrap 均值 | 95%CI | 稳定性 |
|---|---|---|---|---|---|
| 支扩咯血（咯血 vs 无咯血） | 6 | 0.725 | 0.734 | 0.664–0.796 | 100% |
| 急性 COPD 加重病因（感染 vs 非感染） | 6 | 0.711 | 0.764 | 0.664–0.847 | 100% |

### 8.2 任务一：支扩咯血 clinical6（心肌 + 右房 + 气道结构）

| 特征 | 方向（咯血组） | 病理解读 |
|---|---|---|
| `heart_myocardium::original_shape_Sphericity` | 更低（心肌更拉长） | 慢性肺动脉高压致右心代偿/心室重构 |
| `heart_atrium_right::original_shape_MeshVolume` | 更高（右房增大） | 右心负荷↑ |
| `aq_TD_fwhm_all` | 更低（T/D 下降） | 管壁破坏/变薄 → 侵蚀出血 |
| `aq_GenLe2_Wall_Thickness_mm_mean` | 更低（管壁变薄） | 破坏性/侵蚀性支扩表型 |
| `lung_trachea_bronchia::original_firstorder_TotalEnergy` | 更高 | 气道树总体积/密度升高 |
| `PA_Ao_Diameter_Ratio` | 更低 | 肺血管代偿性改变 |

### 8.3 任务二：急性 COPD 加重病因 clinical6（肺实质 + 气道）

| 特征 | 方向（感染型） | 病理解读 |
|---|---|---|
| `lung_vessels::original_firstorder_Mean` | 更高 | 炎症充血/灌注增加 |
| `Lobe_LLL_LAA950_pct` | 更低（肺气肿轻） | 感染型以实变/浸润为主，非感染型以肺气肿为主 |
| `Lobe_RLL_Perc15_HU` | 更高（肺密度高） | 实变/浸润 |
| `aq_wall_hu_kurt` | 更高 | 气道炎症致管壁密度峰态↑ |
| `aq_blur_trans_width_std` | 更低（边界锐利） | 炎症渗出边界清晰 |
| `aq_Din_mean_gen3` | 更低（管腔窄） | 黏膜水肿致管腔变窄 |

### 8.4 共同结论

- 两个任务中 AirQuant 气道结构特征（T/D、管壁厚度、边界模糊、内径）均进入 clinical6，气道重构是两类表型分层的核心影像载体。
- 特征方向均符合临床病理机制，为"可解释影像表型"提供直接证据。

![图 9. 支扩咯血 clinical6 模型 AUC](figs/fig_hemoptysis_model_auc_2026_05.png)

*图 9. 支扩咯血 clinical6 模型 AUC（误差条 = bootstrap 95%CI）*

![图 10. 急性 COPD 加重病因 clinical6 模型 AUC](figs/fig_aecause_model_auc_2026_05.png)

*图 10. 急性 COPD 加重病因 clinical6 模型 AUC（误差条 = bootstrap 95%CI）*

"""


def main():
    with open(MD_OUT, "r", encoding="utf-8") as f:
        md = f.read()
    print(f"[dbg] read {len(md)} chars; has '精简模型验证'={('精简模型验证' in md)}; "
          f"has '## 8. 结论'={('## 8. 结论' in md)}")

    # 1) 修复精简模型章节编号（8 -> 7）
    n1 = md.count("## 8. 精简模型验证（Top 显著特征 + Bootstrap）")
    md = md.replace("## 8. 精简模型验证（Top 显著特征 + Bootstrap）",
                    "## 7. 精简模型验证（Top 显著特征 + Bootstrap）")
    print(f"[dbg] replace 精简模型 header: matched {n1}")

    # 2) 在结论前插入统一总结，并把结论改为 9
    old_concl = "## 8. 结论"
    if old_concl not in md:
        # 若已变成别的编号，兜底找"## N. 结论"
        for n in range(9, 3, -1):
            c = f"## {n}. 结论"
            if c in md:
                old_concl = c
                break
    if old_concl in md:
        md = md.replace(old_concl, SUMMARY + "## 9. 结论")
        print(f"[dbg] inserted summary before conclusion ({old_concl})")
    else:
        md = md.rstrip() + "\n\n" + SUMMARY
        print("[dbg] appended summary at end (conclusion not found)")

    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[dbg] wrote {len(md)} chars")

    def b64img(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(md_to_html(md, b64img))
    print(f"[report] {MD_OUT}\n[report] {HTML_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
