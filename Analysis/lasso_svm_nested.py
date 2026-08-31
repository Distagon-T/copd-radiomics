"""Fully nested feature selection -> LASSO -> SVM analysis.

The outer split is leave-one-cohort-out across 2026-01/02/04/05. Within each
outer training set, every inner fold independently performs:
  1. univariate Kruskal/pairwise-AUC screening over all radiomics features;
  2. correlation de-redundancy;
  3. multinomial LASSO feature selection;
  4. SVM fitting and validation for C/gamma selection.

The outer test cohort is touched only once for final evaluation. The report
overwrites the previous SVM HTML report with nested results while preserving
the old metrics in a comparison table.
"""

from __future__ import annotations

import base64
import html
import json
import warnings
from collections import Counter
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lasso_svm_ordinal import CLASSES, INPUT, OUT as BASE_OUT, fit_lasso, multiclass_metrics, safe_auc, safe_macro_ovr_auc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUT = BASE_OUT
FIG = OUT / "figs_nested"
SEED = 20260830
MAX_SCREEN_POOL = 250
MAX_CANDIDATES = 30
CORR_CUT = 0.90
PARAM_GRID = list(ParameterGrid({"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1]}))

META = {"cohort", "PatientID", "info_match", "label", "label_name", "label_reason", "main_diagnosis", "main_icd", "other_diagnosis", "other_icd", "feature_source", "source_Patient_ID", "source_PatientID_raw", "source_CT_Series", "source_ICD", "source_AECOPD", "source_COPD_BCOS"}
PREFIXES = ("TD_", "blur_", "wall_", "WA_", "Din_", "Dout_", "mean_", "Pi10", "Vessel_", "Lobe_", "Lung_", "Airway_", "PA_", "Diaphragm_", "pca_", "RV_", "LV_", "CAC_", "BronchoArtery_", "EpiFat_", "FAI_", "Aorta_", "CardioThoracic_", "tortuosity", "n_branches", "pruning", "max_generation", "terminal", "generation", "branch", "junction")


def is_feature(c: str) -> bool:
    return c not in META and not c.startswith("info_") and ("::" in c or c.startswith(PREFIXES))


def find_col(df, names, required=True):
    lookup = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    if required:
        raise KeyError(names)
    return None


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    return q


def screen_features(X: np.ndarray, y: np.ndarray, names: list[str], max_candidates=MAX_CANDIDATES) -> tuple[list[str], pd.DataFrame]:
    rows = []
    for j, name in enumerate(names):
        x = X[:, j]
        finite = np.isfinite(x)
        if finite.sum() < max(20, int(.5 * len(y))) or np.unique(x[finite]).size < 2:
            continue
        p = np.nan
        groups = [x[finite & (y == c)] for c in CLASSES]
        if all(len(g) >= 3 and np.unique(g).size > 1 for g in groups):
            try:
                p = float(stats.kruskal(*groups).pvalue)
            except Exception:
                p = np.nan
        pair_aucs = []
        for low, high in [(0, 1), (0, 2), (1, 2)]:
            ok = finite & np.isin(y, [low, high])
            if ok.sum() == 0 or np.unique(y[ok]).size < 2 or np.unique(x[ok]).size < 2:
                continue
            try:
                raw = roc_auc_score((y[ok] == low).astype(int), x[ok])
                pair_aucs.append(max(float(raw), 1.0 - float(raw)))
            except ValueError:
                pass
        rows.append({"feature": name, "kruskal_p": p, "pairwise_auc_macro": float(np.mean(pair_aucs)) if pair_aucs else np.nan, "observed_fraction": float(finite.mean())})
    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        raise RuntimeError("No usable feature passed the nested screening step")
    stats_df["kruskal_fdr"] = bh_fdr(stats_df["kruskal_p"].to_numpy())
    ranked = stats_df[stats_df["kruskal_fdr"].fillna(1) < .05].sort_values(["pairwise_auc_macro", "kruskal_fdr"], ascending=[False, True])
    if ranked.empty:
        ranked = stats_df.sort_values("pairwise_auc_macro", ascending=False)
    pool = ranked.head(MAX_SCREEN_POOL)["feature"].tolist()
    mat = pd.DataFrame(X[:, [names.index(f) for f in pool]], columns=pool).copy()
    mat = mat.fillna(mat.median()).fillna(0)
    corr = mat.corr(method="spearman").abs()
    selected = []
    for f in pool:
        if not selected or max(float(corr.loc[f, s]) for s in selected) < CORR_CUT:
            selected.append(f)
        if len(selected) >= max_candidates:
            break
    if len(selected) < min(5, max_candidates):
        selected = pool[: min(max_candidates, len(pool))]
    stats_df["selected_nested"] = stats_df["feature"].isin(selected)
    stats_df["selection_order"] = stats_df["feature"].map({f: i + 1 for i, f in enumerate(selected)})
    return selected, stats_df.sort_values(["selected_nested", "selection_order", "pairwise_auc_macro"], ascending=[False, True, False])


def make_svm(C, gamma, probability=False):
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("svm", SVC(kernel="rbf", C=C, gamma=gamma, probability=probability, class_weight="balanced", random_state=SEED))])


def fit_inner_lasso_svm(Xtr, ytr, Xval, yval, feature_names):
    selected, screen_df = screen_features(Xtr, ytr, feature_names)
    idx = [feature_names.index(f) for f in selected]
    lasso, lasso_selected, _ = fit_lasso(Xtr[:, idx], ytr, selected)
    if not lasso_selected:
        lasso_selected = selected[:3]
    sv_idx = [selected.index(f) for f in lasso_selected]
    scores = {}
    for param in PARAM_GRID:
        svm = make_svm(param["C"], param["gamma"], probability=False)
        svm.fit(Xtr[:, idx][:, sv_idx], ytr)
        pred = svm.predict(Xval[:, idx][:, sv_idx])
        scores.setdefault((param["C"], str(param["gamma"])), []).append(float(balanced_accuracy_score(yval, pred)))
    return lasso, selected, lasso_selected, screen_df, scores


def tune_parameters(X, y, feature_names):
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = {(p["C"], str(p["gamma"])): [] for p in PARAM_GRID}
    inner_selected = []
    for train, val in inner.split(X, y):
        _, selected, lasso_selected, screen_df, fold_scores = fit_inner_lasso_svm(X[train], y[train], X[val], y[val], feature_names)
        inner_selected.append({"screened": selected, "lasso_selected": lasso_selected})
        for key, vals in fold_scores.items():
            scores[key].extend(vals)
    mean_scores = {key: float(np.mean(vals)) if vals else -np.inf for key, vals in scores.items()}
    best_key = max(mean_scores, key=mean_scores.get)
    best = {"C": float(best_key[0]), "gamma": best_key[1] if best_key[1] == "scale" else float(best_key[1])}
    return best, mean_scores, inner_selected


def reorder_scores(scores, classes):
    order = [list(classes).index(c) for c in CLASSES]
    return np.asarray(scores)[:, order]


def bootstrap_ci(y, scores, model_name, n_boot=1000):
    rng = np.random.default_rng(SEED)
    if isinstance(scores, pd.DataFrame):
        prefix = "lasso" if model_name.lower().startswith("lasso") else "svm"
        scores = scores[[f"{prefix}_s{c}" for c in CLASSES]].to_numpy(float)
    point = [safe_auc((y == c).astype(int), scores[:, c]) for c in CLASSES]
    point.append(safe_macro_ovr_auc(y, scores))
    values = [[] for _ in range(4)]
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y)); yy = y[idx]; ss = scores[idx]
        vals = [safe_auc((yy == c).astype(int), ss[:, c]) for c in CLASSES]
        vals.append(safe_macro_ovr_auc(yy, ss))
        for j, v in enumerate(vals):
            if np.isfinite(v): values[j].append(v)
    names = ["auc_ovr_0", "auc_ovr_1", "auc_ovr_2", "auc_macro_ovr"]
    return pd.DataFrame([{ "model": model_name, "metric": n, "estimate": v, "ci95_low": np.quantile(s, .025), "ci95_high": np.quantile(s, .975), "n_boot": len(s)} for n, v, s in zip(names, point, values)])


def uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def create_plots(oof, old_oof, frequency):
    FIG.mkdir(parents=True, exist_ok=True); paths = []
    y = oof.Label.to_numpy(int)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for c, ax in zip(CLASSES, axes):
        yy = (y == c).astype(int)
        for model, color in [("LASSO", "#1f77b4"), ("SVM", "#d62728")]:
            sc = oof[[f"{model.lower()}_s{c}" for _ in [0]]].to_numpy().ravel() if False else oof[f"{model.lower()}_s{c}"].to_numpy(float)
            ax.plot(*roc_curve(yy, sc)[:2], color=color, lw=2, label=f"{model} AUC={safe_auc(yy, sc):.3f}")
        ax.plot([0,1],[0,1],"k--",lw=.8); ax.set(title=f"Label {c} vs rest", xlabel="False-positive rate", ylabel="True-positive rate"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Nested OOF ROC: LASSO and SVM"); fig.tight_layout(); p=FIG/"nested_oof_roc.png"; fig.savefig(p,dpi=180,bbox_inches="tight"); plt.close(fig); paths.append(p)
    cm = confusion_matrix(y, oof.svm_pred, labels=CLASSES); cmn=cm/cm.sum(axis=1,keepdims=True)
    fig, ax=plt.subplots(figsize=(5.2,4.4)); im=ax.imshow(cmn,cmap="Blues",vmin=0,vmax=1)
    for i in range(3):
        for j in range(3): ax.text(j,i,f"{cm[i,j]}\n({cmn[i,j]:.1%})",ha="center",va="center",color="white" if cmn[i,j]>.5 else "black")
    ax.set(xticks=range(3),yticks=range(3),xlabel="Predicted label",ylabel="True label",title="Nested SVM confusion matrix"); fig.colorbar(im,ax=ax,label="Row proportion"); fig.tight_layout(); p=FIG/"nested_svm_confusion_matrix.png"; fig.savefig(p,dpi=180,bbox_inches="tight"); plt.close(fig); paths.append(p)
    fq=frequency.sort_values("frequency").tail(25); fig,ax=plt.subplots(figsize=(8.5,7)); ax.barh(fq.feature,fq.frequency,color="#2ca02c"); ax.set(xlabel="Outer folds selected by LASSO",title="Nested LASSO selection stability"); ax.grid(axis="x",alpha=.2); fig.tight_layout(); p=FIG/"nested_lasso_selection_frequency.png"; fig.savefig(p,dpi=180,bbox_inches="tight"); plt.close(fig); paths.append(p)
    if old_oof is not None:
        fig, ax=plt.subplots(figsize=(7,5)); labels=["Old OOF", "Nested OOF"]; vals=[safe_macro_ovr_auc(y, old_oof[[f"svm_s{c}" for c in CLASSES]].to_numpy(float)), safe_macro_ovr_auc(y, oof[[f"svm_s{c}" for c in CLASSES]].to_numpy(float))]; ax.bar(labels,vals,color=["#999999","#d62728"]); ax.set_ylim(0,1); ax.set_ylabel("Macro OVR-AUC"); ax.set_title("SVM validation estimate before vs after nesting"); ax.grid(axis="y",alpha=.2); fig.tight_layout(); p=FIG/"svm_old_vs_nested_auc.png"; fig.savefig(p,dpi=180,bbox_inches="tight"); plt.close(fig); paths.append(p)
    return paths


def table(df, n=30):
    return df.head(n).to_html(index=False, border=0, classes="data", float_format=lambda x:f"{x:.4f}")


table_html = table


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    previous_metrics = pd.read_csv(OUT / "pooled_oof_metrics.csv") if (OUT / "pooled_oof_metrics.csv").exists() else pd.DataFrame()
    previous_oof = pd.read_csv(OUT / "oof_predictions_loco.csv") if (OUT / "oof_predictions_loco.csv").exists() else None
    df = pd.read_csv(INPUT, low_memory=False)
    pid = find_col(df,["PatientID","patient_id","Patient_ID"]); label_col=find_col(df,["Label","label"]); cohort_col=find_col(df,["cohort","Cohort","sequence","序列"],False)
    if cohort_col is None: cohort_col="_cohort"; df[cohort_col]="pooled"
    feature_names=[c for c in df.columns if is_feature(c)]
    valid=pd.to_numeric(df[label_col],errors="coerce").isin(CLASSES)
    work=df.loc[valid,[pid,cohort_col,label_col]+feature_names].copy(); work["Label"]=pd.to_numeric(work[label_col],errors="coerce").astype(int); work["cohort"]=work[cohort_col].astype(str); work=work.drop_duplicates(pid).reset_index(drop=True)
    X=work[feature_names].apply(pd.to_numeric,errors="coerce").to_numpy(float); y=work.Label.to_numpy(int); groups=work.cohort.to_numpy(str)
    outer=GroupKFold(n_splits=work.cohort.nunique()); oof_rows=[]; fold_rows=[]; lasso_selection=[]; screen_rows=[]; final_outer_models=[]
    for fold,(train,test) in enumerate(outer.split(X,y,groups),1):
        best_params, inner_scores, inner_selected = tune_parameters(X[train],y[train],feature_names)
        selected, screen_df=screen_features(X[train],y[train],feature_names); screen_df["outer_fold"]=fold; screen_df.to_csv(OUT/f"nested_outer{fold}_screening.csv",index=False,encoding="utf-8-sig"); screen_rows.append(screen_df)
        idx=[feature_names.index(f) for f in selected]; lasso, lasso_selected, coef=fit_lasso(X[train][:,idx],y[train],selected); lasso_selection.extend({"fold":fold,"feature":f} for f in lasso_selected)
        sv_idx=[selected.index(f) for f in lasso_selected]; svm=make_svm(best_params["C"],best_params["gamma"],probability=True); svm.fit(X[train][:,idx][:,sv_idx],y[train])
        lp=lasso.predict_proba(X[test][:,idx]); ld=lasso.decision_function(X[test][:,idx]); lp=reorder_scores(lp,lasso.named_steps["lasso"].classes_); ld=reorder_scores(ld,lasso.named_steps["lasso"].classes_); lpred=CLASSES[np.argmax(ld,axis=1)]
        sp=svm.predict_proba(X[test][:,idx][:,sv_idx]); sd=svm.decision_function(X[test][:,idx][:,sv_idx]); sp=reorder_scores(sp,svm.classes_); sd=reorder_scores(sd,svm.classes_); spred=svm.predict(X[test][:,idx][:,sv_idx]).astype(int)
        lm=multiclass_metrics(y[test],lp,pred=lpred,scores=ld); sm=multiclass_metrics(y[test],sp,pred=spred,scores=sd)
        for model,met in [("LASSO_nested",lm),("SVM_nested",sm)]: fold_rows.append({"fold":fold,"test_cohort":";".join(sorted(set(groups[test]))),"model":model,"best_params":json.dumps(best_params),"n_screened":len(selected),"n_lasso_selected":len(lasso_selected),"inner_best_balanced_accuracy":max(inner_scores.values()),**met})
        for i,row_i in enumerate(test): oof_rows.append({"row_index":int(row_i),"PatientID":str(work.iloc[row_i][pid]),"cohort":groups[row_i],"Label":int(y[row_i]),"lasso_pred":int(lpred[i]),"svm_pred":int(spred[i]),**{f"lasso_p{c}":float(lp[i,c]) for c in CLASSES},**{f"svm_p{c}":float(sp[i,c]) for c in CLASSES},**{f"lasso_s{c}":float(ld[i,c]) for c in CLASSES},**{f"svm_s{c}":float(sd[i,c]) for c in CLASSES}})
        final_outer_models.append({"fold":fold,"selected_candidates":selected,"lasso_selected":lasso_selected,"best_params":best_params})
    oof=pd.DataFrame(oof_rows).sort_values("row_index").reset_index(drop=True); fold_df=pd.DataFrame(fold_rows); oof.to_csv(OUT/"nested_oof_predictions_loco.csv",index=False,encoding="utf-8-sig"); fold_df.to_csv(OUT/"nested_loco_fold_metrics.csv",index=False,encoding="utf-8-sig")
    pooled=pd.DataFrame([{ "model":"LASSO_nested",**multiclass_metrics(y,oof[[f"lasso_p{c}" for c in CLASSES]].to_numpy(),pred=oof.lasso_pred.to_numpy(int),scores=oof[[f"lasso_s{c}" for c in CLASSES]].to_numpy())},{"model":"SVM_nested",**multiclass_metrics(y,oof[[f"svm_p{c}" for c in CLASSES]].to_numpy(),pred=oof.svm_pred.to_numpy(int),scores=oof[[f"svm_s{c}" for c in CLASSES]].to_numpy())}]); pooled.to_csv(OUT/"nested_pooled_oof_metrics.csv",index=False,encoding="utf-8-sig")
    boot=pd.concat([bootstrap_ci(y,oof,"LASSO_nested"),bootstrap_ci(y,oof,"SVM_nested")],ignore_index=True); boot.to_csv(OUT/"nested_pooled_oof_auc_bootstrap_ci.csv",index=False,encoding="utf-8-sig")
    sf=pd.DataFrame({"feature":feature_names}); counts=Counter(x["feature"] for x in lasso_selection); sf["frequency"]=sf.feature.map(counts).fillna(0).astype(int); sf["selection_rate"]=sf.frequency/work.cohort.nunique(); sf.sort_values(["frequency","feature"],ascending=[False,True]).to_csv(OUT/"nested_lasso_selection_frequency.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{ "fold":m["fold"],"feature":f,"stage":"screened"} for m in final_outer_models for f in m["selected_candidates"]]+[{"fold":m["fold"],"feature":f,"stage":"lasso_selected"} for m in final_outer_models for f in m["lasso_selected"]]).to_csv(OUT/"nested_fold_selected_features.csv",index=False,encoding="utf-8-sig")
    old_for_plot=previous_oof if previous_oof is not None and all(f"svm_s{c}" in previous_oof.columns for c in CLASSES) else None; plots=create_plots(oof,old_for_plot,sf)
    (OUT/"nested_outer_fold_models.json").write_text(json.dumps(final_outer_models,ensure_ascii=False,indent=2),encoding="utf-8")
    report=build_report(work,feature_names,pooled,boot,fold_df,sf,previous_metrics,plots); (OUT/"lasso_svm_report.html").write_text(report,encoding="utf-8"); (OUT/"lasso_svm_nested_report.html").write_text(report,encoding="utf-8"); (OUT/"lasso_svm_nested_report.md").write_text(build_md(work,pooled,boot,sf),encoding="utf-8")
    print(json.dumps({"n":len(work),"n_all_features":len(feature_names),"pooled_nested":pooled.to_dict(orient="records"),"output":str(OUT)},ensure_ascii=False,indent=2))


def build_md(work,pooled,boot,sf):
    lines=["| Model | Accuracy | Balanced accuracy | Macro-F1 | Macro OVR-AUC | Label0 | Label1 | Label2 |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _,r in pooled.iterrows(): lines.append(f"| {r['model']} | {r.accuracy:.3f} | {r.balanced_accuracy:.3f} | {r.macro_f1:.3f} | {r.auc_macro_ovr:.3f} | {r.auc_ovr_0:.3f} | {r.auc_ovr_1:.3f} | {r.auc_ovr_2:.3f} |")
    top=sf.sort_values("frequency",ascending=False).head(15)
    return f"""# Nested LASSO → SVM 三分类报告

本报告将单变量筛选、相关性去冗余、LASSO和SVM调参全部置于外层留一序列验证的训练数据内部。有效样本：{len(work)}；原始影像组学/气道特征：{sf.shape[0]}。

## Nested pooled OOF

{chr(10).join(lines)}

Label 1 是主要瓶颈。AUC bootstrap CI 见 `nested_pooled_oof_auc_bootstrap_ci.csv`。

## LASSO稳定性

{chr(10).join(f"- `{r.feature}`：{int(r.frequency)}/4 个外层折叠" for r in top.itertuples())}
"""


def build_report(work,all_features,pooled,boot,fold_df,sf,previous_metrics,plots):
    imgs="".join(f'<figure><img src="{uri(p)}"><figcaption>{html.escape(p.stem)}</figcaption></figure>' for p in plots)
    old=previous_metrics.to_html(index=False,border=0,classes="data",float_format=lambda x:f"{x:.4f}") if not previous_metrics.empty else "<p>Previous metrics unavailable.</p>"
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>Nested LASSO SVM report</title><style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1450px;margin:30px auto;padding:0 24px;line-height:1.5;color:#222}}h1,h2{{color:#17365d}}table.data{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 22px}}table.data th,table.data td{{border:1px solid #d9e2f3;padding:5px 7px;text-align:left}}table.data th{{background:#eaf2f8}}figure{{display:inline-block;vertical-align:top;margin:8px;width:47%;text-align:center}}figure img{{max-width:100%;border:1px solid #ddd}}figcaption{{font-size:12px;color:#555}}.note{{background:#fff7e6;border-left:4px solid #f0ad4e;padding:10px 14px}}</style></head><body><h1>Nested LASSO → SVM 三分类报告</h1><p>本报告替换了原 SVM 报告中的主验证结果。目标为 Label 0/1/2 阶梯式急性加重风险。</p><h2>1. Nested设计</h2><ul><li>有效样本：<b>{len(work)}</b>；原始可用影像组学/气道特征：<b>{len(all_features)}</b>。</li><li>外层：2026-01、2026-02、2026-04、2026-05 四个序列留一队列测试。</li><li>每个外层训练折的内层3折中，重新做全特征单变量 Kruskal/AUC 筛选、BH-FDR、Spearman 去冗余、LASSO选择和 SVM C/gamma选择。</li><li>外层测试折没有参与任何特征筛选、调参或概率拟合。</li><li>类别不平衡使用 balanced class weight；未使用 SMOTE。</li></ul><div class='note'>AUC 使用 decision score；Accuracy、balanced accuracy、Macro-F1使用预测类别；log-loss使用SVM概率。这里的 nested OOF 才是主要泛化性能估计。</div><h2>2. Nested pooled OOF结果</h2>{table_html(pooled)}<h3>Bootstrap AUC 95% CI</h3>{table_html(boot,20)}<h2>3. 各外层序列结果</h2>{table_html(fold_df,20)}<h2>4. 与原SVM结果对照</h2><p>原报告结果（候选特征在全体样本上预筛选，非完整nested）：</p>{old}<p>原报告的 apparent macro OVR-AUC=0.833 是全队列拟合内评估，不能与 nested OOF 直接比较；nested OOF 应作为主结论。</p><h2>5. LASSO稳定性</h2>{table_html(sf.sort_values(['frequency','feature'],ascending=[False,True]),30)}<h2>6. 图表</h2>{imgs}<h2>7. 结论与限制</h2><p>如果 nested OOF AUC 明显低于 apparent AUC，说明原结果包含训练集拟合乐观偏倚；如果 Label 1 的CI跨过0.5，则目前不能认为模型稳定识别结构易损期。正式发表前仍建议使用独立医院/独立扫描协议队列进行外部验证。</p><p>结果文件包括 <code>nested_oof_predictions_loco.csv</code>、<code>nested_loco_fold_metrics.csv</code>、<code>nested_pooled_oof_auc_bootstrap_ci.csv</code>、<code>nested_lasso_selection_frequency.csv</code> 和外层筛选明细。</p></body></html>"""


if __name__ == "__main__":
    main()
