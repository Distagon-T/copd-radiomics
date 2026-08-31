"""LASSO -> SVM analysis for the ordinal acute-exacerbation risk cohort.

This is the Python counterpart of the former glmnet/caret/pROC workflow:
  1. L1-penalized multinomial logistic regression (LASSO-like feature selection)
  2. Class-weighted SVM on the selected features
  3. Leave-one-cohort-out validation, pooled out-of-fold ROC/AUC, and report

The initial candidate list is the non-redundant list from the preceding feature
selection report. All preprocessing is fitted inside each validation fold.
"""

from __future__ import annotations

import base64
import html
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


ROOT = Path(r"D:\copd-radiomics")
INPUT = Path(r"E:\DICOM\results\ordinal_risk_all_patients_feature_label.csv")
CANDIDATES = Path(r"E:\DICOM\reports\feature_selection_ordinal_ae\selected_nonredundant_features.csv")
OUT = Path(r"E:\DICOM\reports\feature_selection_ordinal_ae\lasso_svm")
FIG = OUT / "figs"
SEED = 20260830
CLASSES = np.array([0, 1, 2])


def find_col(df: pd.DataFrame, names: list[str], required: bool = True) -> str | None:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    if required:
        raise KeyError(f"Cannot find any of {names}; available examples: {list(df.columns[:20])}")
    return None


def fmt(x, digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, score))
    except ValueError:
        return float("nan")


def safe_macro_ovr_auc(y: np.ndarray, proba: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, proba, labels=CLASSES, multi_class="ovr", average="macro"))
    except ValueError:
        values = [safe_auc((y == c).astype(int), proba[:, int(c)]) for c in CLASSES]
        values = [x for x in values if np.isfinite(x)]
        return float(np.mean(values)) if values else float("nan")


def bootstrap_auc_table(y: np.ndarray, oof: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for model in ["lasso", "svm"]:
        scores = oof[[f"{model}_s{c}" for c in CLASSES]].to_numpy(float)
        point = [safe_auc((y == c).astype(int), scores[:, int(c)]) for c in CLASSES]
        point.append(safe_macro_ovr_auc(y, scores))
        boot = [[] for _ in range(4)]
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            yy = y[idx]
            ss = scores[idx]
            vals = [safe_auc((yy == c).astype(int), ss[:, int(c)]) for c in CLASSES]
            vals.append(safe_macro_ovr_auc(yy, ss))
            for j, value in enumerate(vals):
                if np.isfinite(value):
                    boot[j].append(value)
        names = ["auc_ovr_0", "auc_ovr_1", "auc_ovr_2", "auc_macro_ovr"]
        for name, value, samples in zip(names, point, boot):
            rows.append({"model": model.upper(), "metric": name, "estimate": value, "ci95_low": np.quantile(samples, .025), "ci95_high": np.quantile(samples, .975), "n_boot": len(samples)})
    return pd.DataFrame(rows)


def multiclass_metrics(y: np.ndarray, proba: np.ndarray, pred: np.ndarray | None = None, scores: np.ndarray | None = None) -> dict:
    if pred is None:
        pred = CLASSES[np.argmax(proba, axis=1)]
    auc_input = proba if scores is None else scores
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7), labels=CLASSES)),
        "auc_macro_ovr": safe_macro_ovr_auc(y, auc_input),
    }
    for c in CLASSES:
        out[f"auc_ovr_{c}"] = safe_auc((y == c).astype(int), auc_input[:, int(c)])
    return out


def fit_lasso(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> tuple[Pipeline, list[str], np.ndarray]:
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "lasso",
                LogisticRegressionCV(
                    Cs=12,
                    cv=inner,
                    penalty="l1",
                    solver="saga",
                    multi_class="multinomial",
                    class_weight="balanced",
                    scoring="neg_log_loss",
                    max_iter=5000,
                    n_jobs=-1,
                    random_state=SEED,
                    refit=True,
                ),
            ),
        ]
    )
    model.fit(X, y)
    # add_indicator adds columns after the original feature columns; indicators
    # are deliberately excluded from the reported biological feature list.
    coef = model.named_steps["lasso"].coef_[:, : len(feature_names)]
    keep = np.flatnonzero(np.max(np.abs(coef), axis=0) > 1e-7)
    if len(keep) == 0:
        keep = np.argsort(np.max(np.abs(coef), axis=0))[::-1][: min(3, len(feature_names))]
    return model, [feature_names[i] for i in keep], coef


def fit_svm(X: np.ndarray, y: np.ndarray, selected: list[str], all_names: list[str]) -> GridSearchCV:
    idx = [all_names.index(f) for f in selected]
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)),
        ]
    )
    grid = GridSearchCV(
        pipe,
        {"svm__C": [0.1, 1, 10, 100], "svm__gamma": ["scale", 0.01, 0.1, 1]},
        scoring="balanced_accuracy",
        cv=inner,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X[:, idx], y)
    grid.selected_indices_ = idx
    return grid


def data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def save_plots(oof: pd.DataFrame, selected_freq: pd.DataFrame, coef_df: pd.DataFrame) -> list[Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    plots: list[Path] = []
    y = oof["Label"].to_numpy(int)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for c, ax in zip(CLASSES, axes):
        yy = (y == c).astype(int)
        for model, color in [("lasso", "#1f77b4"), ("svm", "#d62728")]:
            score = oof[f"{model}_s{c}"].to_numpy(float)
            if yy.sum() and (len(yy) - yy.sum()):
                fpr, tpr, _ = roc_curve(yy, score)
                auc = safe_auc(yy, score)
                ax.plot(fpr, tpr, lw=2, label=f"{model.upper()} AUC={auc:.3f}", color=color)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(f"Label {c} vs rest")
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("True-positive rate")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.2)
    fig.suptitle("Pooled out-of-fold ROC (leave-one-cohort-out)")
    fig.tight_layout()
    p = FIG / "oof_roc_lasso_vs_svm.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    plots.append(p)

    cm = confusion_matrix(y, oof["svm_pred"].to_numpy(int), labels=CLASSES)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]}\n({cmn[i,j]:.1%})", ha="center", va="center", color="white" if cmn[i,j] > .5 else "black")
    ax.set(xticks=range(3), yticks=range(3), xlabel="Predicted label", ylabel="True label", title="SVM pooled OOF confusion matrix")
    fig.colorbar(im, ax=ax, label="Row proportion")
    fig.tight_layout()
    p = FIG / "oof_confusion_matrix.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    plots.append(p)

    sf = selected_freq.sort_values(["frequency", "feature"], ascending=[False, True]).head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.barh(sf["feature"], sf["frequency"], color="#2ca02c")
    ax.set_xlim(0, max(1, int(selected_freq["frequency"].max())))
    ax.set_xlabel("Number of outer folds selected")
    ax.set_title("LASSO feature-selection stability")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    p = FIG / "lasso_selection_frequency.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    plots.append(p)

    if not coef_df.empty:
        piv = coef_df.pivot(index="feature", columns="class", values="coefficient").fillna(0)
        piv = piv.loc[piv.abs().max(axis=1).sort_values(ascending=False).head(25).index]
        fig, ax = plt.subplots(figsize=(7.5, max(5, len(piv) * .25)))
        vmax = max(.01, float(np.abs(piv.to_numpy()).max()))
        im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="coolwarm", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
        ax.set(xticks=range(len(piv.columns)), xticklabels=[f"Label {x}" for x in piv.columns], yticks=range(len(piv)), yticklabels=piv.index, title="Final multinomial LASSO coefficients")
        fig.colorbar(im, ax=ax, label="Standardized coefficient")
        fig.tight_layout()
        p = FIG / "final_lasso_coefficients.png"
        fig.savefig(p, dpi=180, bbox_inches="tight")
        plt.close(fig)
        plots.append(p)
    return plots


def table_html(df: pd.DataFrame, n: int = 20) -> str:
    if df.empty:
        return "<p>No rows.</p>"
    return df.head(n).to_html(index=False, classes="data", border=0, float_format=lambda x: f"{x:.4f}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT, low_memory=False)
    pid_col = find_col(df, ["PatientID", "patient_id", "Patient_ID"])
    label_col = find_col(df, ["Label", "label"])
    cohort_col = find_col(df, ["cohort", "Cohort", "sequence", "序列"], required=False)
    if cohort_col is None:
        cohort_col = "_cohort"
        df[cohort_col] = "pooled"

    candidates = pd.read_csv(CANDIDATES)
    feature_col = find_col(candidates, ["feature", "Feature", "feature_name"])
    candidate_names = [str(x) for x in candidates[feature_col].dropna().tolist()]
    candidate_names = [x for x in candidate_names if x in df.columns]
    missing_candidates = sorted(set(candidates[feature_col].dropna().astype(str)) - set(candidate_names))
    if len(candidate_names) < 2:
        raise RuntimeError(f"Only {len(candidate_names)} candidate features found in merged table")

    label = pd.to_numeric(df[label_col], errors="coerce")
    valid = label.isin(CLASSES)
    work = df.loc[valid, [pid_col, cohort_col, label_col] + candidate_names].copy()
    work["Label"] = pd.to_numeric(work[label_col], errors="coerce").astype(int)
    work["cohort"] = work[cohort_col].astype(str)
    work = work.drop_duplicates(subset=[pid_col], keep="first")
    Xdf = work[candidate_names].apply(pd.to_numeric, errors="coerce")
    usable = [c for c in candidate_names if Xdf[c].notna().sum() >= max(20, int(.50 * len(Xdf))) and Xdf[c].nunique(dropna=True) > 1]
    dropped = sorted(set(candidate_names) - set(usable))
    X = Xdf[usable].to_numpy(float)
    y = work["Label"].to_numpy(int)
    groups = work["cohort"].to_numpy(str)

    group_counts = work.groupby(["cohort", "Label"]).size().unstack(fill_value=0)
    group_counts.to_csv(OUT / "cohort_label_counts.csv", encoding="utf-8-sig")

    outer = GroupKFold(n_splits=work["cohort"].nunique())
    oof_rows = []
    fold_rows = []
    selections = []
    for fold, (train, test) in enumerate(outer.split(X, y, groups), start=1):
        train_groups = sorted(set(groups[train]))
        test_groups = sorted(set(groups[test]))
        lasso, selected, coef = fit_lasso(X[train], y[train], usable)
        svm = fit_svm(X[train], y[train], selected, usable)
        lp = lasso.predict_proba(X[test])
        sp = svm.predict_proba(X[test][:, svm.selected_indices_])
        ld = lasso.decision_function(X[test])
        sd = svm.decision_function(X[test][:, svm.selected_indices_])
        # sklearn uses class order; enforce the project order 0/1/2.
        l_order = list(lasso.named_steps["lasso"].classes_)
        s_order = list(svm.classes_)
        lp = lp[:, [l_order.index(c) for c in CLASSES]]
        sp = sp[:, [s_order.index(c) for c in CLASSES]]
        ld = ld[:, [l_order.index(c) for c in CLASSES]]
        sd = sd[:, [s_order.index(c) for c in CLASSES]]
        pred_l = CLASSES[np.argmax(ld, axis=1)]
        pred_s = CLASSES[np.argmax(sd, axis=1)]
        lm = multiclass_metrics(y[test], lp, pred=pred_l, scores=ld)
        sm = multiclass_metrics(y[test], sp, pred=pred_s, scores=sd)
        for model, metrics in [("lasso", lm), ("svm", sm)]:
            fold_rows.append({"fold": fold, "train_cohorts": ";".join(train_groups), "test_cohort": ";".join(test_groups), "model": model, "best_params": json.dumps(svm.best_params_) if model == "svm" else "", **metrics, "n_selected": len(selected)})
        selections.extend({"fold": fold, "feature": f, "selected": 1} for f in selected)
        for i, row_i in enumerate(test):
            oof_rows.append({"row_index": int(row_i), "PatientID": str(work.iloc[row_i][pid_col]), "cohort": groups[row_i], "Label": int(y[row_i]), "lasso_pred": int(pred_l[i]), "svm_pred": int(pred_s[i]), **{f"lasso_p{c}": float(lp[i, c]) for c in CLASSES}, **{f"svm_p{c}": float(sp[i, c]) for c in CLASSES}, **{f"lasso_s{c}": float(ld[i, c]) for c in CLASSES}, **{f"svm_s{c}": float(sd[i, c]) for c in CLASSES}})

    oof = pd.DataFrame(oof_rows).sort_values("row_index").reset_index(drop=True)
    fold_metrics = pd.DataFrame(fold_rows)
    selection_base = pd.DataFrame({"feature": usable})
    selection_obs = pd.DataFrame(selections)
    if selection_obs.empty:
        selection_freq = selection_base.assign(frequency=0, selection_rate=0.0)
    else:
        counts = selection_obs.groupby("feature").size().rename("frequency")
        selection_freq = selection_base.join(counts, on="feature").fillna(0)
        selection_freq["frequency"] = selection_freq["frequency"].astype(int)
        selection_freq["selection_rate"] = selection_freq["frequency"] / work["cohort"].nunique()
    oof.to_csv(OUT / "oof_predictions_loco.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(OUT / "loco_fold_metrics.csv", index=False, encoding="utf-8-sig")
    selection_freq.sort_values(["frequency", "feature"], ascending=[False, True]).to_csv(OUT / "lasso_selection_frequency.csv", index=False, encoding="utf-8-sig")

    pooled_lasso = multiclass_metrics(y, oof[[f"lasso_p{c}" for c in CLASSES]].to_numpy(), pred=oof["lasso_pred"].to_numpy(int), scores=oof[[f"lasso_s{c}" for c in CLASSES]].to_numpy())
    pooled_svm = multiclass_metrics(y, oof[[f"svm_p{c}" for c in CLASSES]].to_numpy(), pred=oof["svm_pred"].to_numpy(int), scores=oof[[f"svm_s{c}" for c in CLASSES]].to_numpy())
    pooled = pd.DataFrame([{ "model": "LASSO", **pooled_lasso}, {"model": "SVM_after_LASSO", **pooled_svm}])
    pooled.to_csv(OUT / "pooled_oof_metrics.csv", index=False, encoding="utf-8-sig")
    bootstrap = bootstrap_auc_table(y, oof)
    bootstrap.to_csv(OUT / "pooled_oof_auc_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    cm = pd.DataFrame(confusion_matrix(y, oof["svm_pred"], labels=CLASSES), index=["true_0", "true_1", "true_2"], columns=["pred_0", "pred_1", "pred_2"])
    cm.to_csv(OUT / "svm_oof_confusion_matrix.csv", encoding="utf-8-sig")

    final_lasso, final_selected, final_coef = fit_lasso(X, y, usable)
    final_svm = fit_svm(X, y, final_selected, usable)
    final_proba = final_svm.predict_proba(X[:, final_svm.selected_indices_])
    final_scores = final_svm.decision_function(X[:, final_svm.selected_indices_])
    final_order = list(final_svm.classes_)
    final_proba = final_proba[:, [final_order.index(c) for c in CLASSES]]
    final_scores = final_scores[:, [final_order.index(c) for c in CLASSES]]
    final_pred = CLASSES[np.argmax(final_scores, axis=1)]
    final_app = multiclass_metrics(y, final_proba, pred=final_pred, scores=final_scores)
    coef_rows = []
    for j, c in enumerate(CLASSES):
        for i, f in enumerate(usable):
            coef_rows.append({"feature": f, "class": int(c), "coefficient": float(final_coef[j, i]), "abs_coefficient": float(abs(final_coef[j, i])), "selected": int(f in final_selected)})
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(OUT / "final_lasso_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": final_selected, "selected_by_final_lasso": 1}).to_csv(OUT / "final_lasso_selected_features.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "final_model_summary.json", "w", encoding="utf-8") as f:
        json.dump({"input": str(INPUT), "candidate_file": str(CANDIDATES), "n": len(work), "n_features_candidate": len(candidate_names), "n_features_usable": len(usable), "missing_candidate_features": missing_candidates, "dropped_features": dropped, "final_lasso_selected": final_selected, "final_svm_best_params": final_svm.best_params_, "final_apparent_metrics": final_app, "outer_validation": "GroupKFold leave-one-cohort-out", "class_weight": "balanced", "smote": False}, f, ensure_ascii=False, indent=2)
    joblib.dump({"lasso": final_lasso, "svm": final_svm, "candidate_features": usable, "selected_features": final_selected, "classes": CLASSES.tolist()}, OUT / "final_lasso_svm_models.joblib")

    plots = save_plots(oof, selection_freq, coef_df[coef_df["selected"] == 1])
    report = build_report(work, candidate_names, usable, dropped, missing_candidates, group_counts, pooled, bootstrap, fold_metrics, selection_freq, final_selected, final_svm, final_app, plots)
    (OUT / "lasso_svm_report.html").write_text(report, encoding="utf-8")
    (OUT / "lasso_svm_report.md").write_text(build_markdown(work, usable, pooled, bootstrap, selection_freq, final_selected, final_svm, final_app), encoding="utf-8")
    print(json.dumps({"n": len(work), "features_usable": len(usable), "final_selected": final_selected, "pooled_oof": pooled.to_dict(orient="records"), "output": str(OUT)}, ensure_ascii=False, indent=2))


def build_markdown(work, usable, pooled, bootstrap, selection_freq, final_selected, final_svm, final_app):
    rows = []
    for _, r in pooled.iterrows():
        rows.append(f"| {r['model']} | {fmt(r['accuracy'])} | {fmt(r['balanced_accuracy'])} | {fmt(r['macro_f1'])} | {fmt(r['auc_macro_ovr'])} | {fmt(r['auc_ovr_0'])} | {fmt(r['auc_ovr_1'])} | {fmt(r['auc_ovr_2'])} |")
    top = selection_freq.sort_values(["frequency", "feature"], ascending=[False, True]).head(15)
    top_lines = "\n".join(f"- `{r.feature}`：{int(r.frequency)}/{work['cohort'].nunique()} 个外层折叠" for r in top.itertuples())
    return f"""# LASSO → SVM 三分类分析报告

## 设计

- 有效样本：{len(work)}；特征输入：上一阶段非冗余候选特征；可用特征：{len(usable)}。
    - 主验证：按 2026-01/02/04/05 序列进行 leave-one-cohort-out（GroupKFold）。预处理、LASSO 和 SVM 调参均在训练折内完成。
- LASSO：多分类 L1 惩罚 Logistic 回归，对应 R `glmnet(..., family='multinomial', alpha=1)`。
- SVM：RBF kernel、`class_weight='balanced'`，在 LASSO 保留特征上用训练折内网格搜索 C/gamma。
- 未使用 SMOTE；对于 Label 1 较少的问题，先用类别权重避免合成样本带来的不稳定性。

## pooled OOF 结果

| 模型 | Accuracy | Balanced accuracy | Macro-F1 | Macro OVR-AUC | AUC Label0 | AUC Label1 | AUC Label2 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

注意：这里的 pooled OOF AUC 使用模型 decision score（而非每个折独立校准的概率）计算；Label 1 的性能仍需结合置信区间、每个序列的样本数和外部验证解释。

Bootstrap 95% CI 见 `pooled_oof_auc_bootstrap_ci.csv`。

## LASSO 稳定性

最终全队列 LASSO 保留 {len(final_selected)} 个特征：

{', '.join(f'`{x}`' for x in final_selected)}

外层折叠中出现频率最高的特征：

{top_lines}

## 最终全队列拟合（仅 apparent）

最终 SVM 参数：`{final_svm.best_params_}`。全队列拟合后的 apparent macro OVR-AUC 为 **{fmt(final_app['auc_macro_ovr'])}**；该数值不应作为泛化性能，应以 OOF 结果为主。

## 与既有 R workflow 的对应

`glmnet` → `LogisticRegressionCV` 的多分类 L1；`pROC` → `roc_auc_score/roc_curve`；`caret/e1071` → `SVC/GridSearchCV`。Hmisc/rms 的校准、H-L 检验和 DCA 适合在模型固定且有独立验证集后追加，本阶段暂不把它们混入特征选择结果。

## 重要限制

上一阶段候选特征列表是在全体有效样本上做过单变量筛选的，因此本分析虽然把预处理、LASSO 和 SVM 放进外层训练折，仍可能受到候选特征预筛选造成的轻度乐观偏倚。论文级结果应在每个外层训练折内重新完成 AUC/Kruskal/FDR 和去冗余筛选，再做 LASSO-SVM。
"""


def build_report(work, candidate_names, usable, dropped, missing_candidates, group_counts, pooled, bootstrap, fold_metrics, selection_freq, final_selected, final_svm, final_app, plots):
    pooled_table = table_html(pooled, 10)
    bootstrap_table = table_html(bootstrap, 20)
    fold_table = table_html(fold_metrics, 20)
    freq_table = table_html(selection_freq.sort_values(["frequency", "feature"], ascending=[False, True]), 30)
    imgs = "".join(f'<figure><img src="{data_uri(p)}"><figcaption>{html.escape(p.stem)}</figcaption></figure>' for p in plots)
    count_html = group_counts.reset_index().to_html(index=False, border=0, classes="data")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>LASSO-SVM ordinal risk report</title>
<style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1450px;margin:30px auto;padding:0 24px;color:#222;line-height:1.5}}h1,h2{{color:#17365d}}table.data{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 20px}}table.data th,table.data td{{border:1px solid #d9e2f3;padding:5px 7px;text-align:left}}table.data th{{background:#eaf2f8}}figure{{display:inline-block;vertical-align:top;margin:8px;width:47%;text-align:center}}figure img{{max-width:100%;height:auto;border:1px solid #ddd}}figcaption{{font-size:12px;color:#555}}.note{{background:#fff7e6;border-left:4px solid #f0ad4e;padding:10px 14px}}code{{background:#f3f3f3;padding:2px 4px}}</style></head><body>
<h1>LASSO → SVM 三分类特征与模型报告</h1>
<p>目标：基于阶梯式急性加重风险标签（Label 0/1/2），在上一阶段筛选的非冗余特征上进行多分类 LASSO，再用 LASSO 保留特征训练 SVM。</p>
<h2>1. 数据与方法</h2><ul><li>有效样本：<b>{len(work)}</b>；原候选特征：{len(candidate_names)}；实际可用特征：{len(usable)}。</li><li>外层验证：按序列分组的 leave-one-cohort-out；每一折内部完成中位数填补、标准化、LASSO 和 SVM 调参。</li><li>LASSO 使用多分类 L1 Logistic 回归，类别权重为 balanced；SVM 使用 RBF kernel、类别权重为 balanced。</li><li>没有使用 SMOTE。Label 1 较少时，类别权重通常比在外层验证前生成合成样本更稳妥。</li></ul>
<div class="note">该分析的主要泛化指标是 pooled out-of-fold 结果；最终全队列拟合结果只是 apparent performance，不能替代独立验证。</div>
<h3>各序列标签分布</h3>{count_html}
<h2>2. pooled OOF 性能</h2>{pooled_table}
<p>Macro OVR-AUC 是三个 one-vs-rest AUC 的宏平均。ROC/AUC 使用模型 decision score 计算；log-loss 使用每折内部校准的概率计算。Label 1 的 AUC 还应结合每个序列外层测试折的样本数解读。</p>
<h3>OOF AUC bootstrap 95% CI</h3>{bootstrap_table}
<h2>3. 外层折叠明细</h2>{fold_table}
<h2>4. LASSO 特征选择</h2><p>最终全队列 LASSO 保留 <b>{len(final_selected)}</b> 个特征：</p><p>{', '.join(f'<code>{html.escape(x)}</code>' for x in final_selected)}</p>{freq_table}
<h2>5. 图表</h2>{imgs}
<h2>6. 最终拟合模型</h2><p>SVM 最优参数：<code>{html.escape(str(final_svm.best_params_))}</code>。全队列 apparent macro OVR-AUC：<b>{fmt(final_app['auc_macro_ovr'])}</b>。</p>
<h2>7. 与旧 R 脚本的对应</h2><ul><li><code>glmnet</code> 多分类 LASSO → <code>LogisticRegressionCV(penalty='l1', solver='saga', multi_class='multinomial')</code>。</li><li><code>pROC</code> → <code>roc_curve</code> 与 <code>roc_auc_score</code>。</li><li><code>caret/e1071</code> SVM → <code>SVC</code> 与 <code>GridSearchCV</code>。</li><li>Hmisc/rms 的校准、H-L 检验、DCA 应在模型锁定并有独立验证集后进行；它们不直接负责本阶段的特征选择。</li></ul>
<h2>8. 解释限制与下一步</h2><p>候选特征来自上一阶段基于全体样本的单变量筛选，因此当前外层验证仍可能有预筛选乐观偏倚。论文级分析应将 AUC/Kruskal、FDR、相关性去冗余、LASSO 和 SVM 全部放进每个外层训练折内，形成完整 nested pipeline。建议下一步比较：线性 SVM、RBF SVM、仅 LASSO、LASSO+SVM，以及训练折内 SMOTE 的敏感性分析，并报告 bootstrap CI。</p>
<p>完整机器可读结果位于本报告同目录：OOF 预测、折叠指标、LASSO 系数、选择频率、混淆矩阵和模型文件。</p></body></html>"""


if __name__ == "__main__":
    main()
