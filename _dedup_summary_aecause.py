#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复 report_copd_ae_cause_2026_05.md：删除重复插入的统一总结章节，并重生成 HTML"""
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


def main():
    with open(MD_OUT, "r", encoding="utf-8") as f:
        md = f.read()

    marker = "## 8. 统一可解释影像表型总结（两任务 clinical6 模型）"
    cnt = md.count(marker)
    print(f"[dbg] marker count = {cnt}")
    if cnt > 1:
        idx = md.find(marker)
        idx2 = md.find(marker, idx + 1)
        concl = "## 9. 结论"
        ci = md.find(concl, idx2)
        if ci != -1:
            md = md[:idx2] + md[ci:]
            print(f"[dbg] removed duplicated block [{idx2}:{ci}]")
        else:
            md = md[:idx2]
            print("[dbg] removed from 2nd occurrence to end")
    elif cnt == 0:
        print("[warn] marker not found, no change")
    else:
        print("[dbg] no duplication")

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
