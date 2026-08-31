"""Binary LASSO -> SVM strategies for the ordinal acute-exacerbation risk cohort.

Two strategies are implemented on top of the same fully nested pipeline
(univariate screening -> correlation de-redundancy -> L1 logistic regression
feature selection -> RBF SVM with inner C/gamma tuning), with leave-one-cohort-out
outer validation:

  * Strategy A ("aggressive early warning"): Label 0  vs  (Label 1 + Label 2).
    Any structural worsening (early or late) is pooled into a single "risk pool"
    so the classifier can be used as a high-sensitivity first-line screen.

  * Strategy B ("Drop & Predict"): Label 1 samples are completely removed from
    training; the model is trained only on the pure Label 0 vs Label 2 hyperplane.
    Held-out Label 1 samples are then scored by the trained model and we check
    whether their continuous risk probabilities concentrate in the 0.4-0.6
    "gray warning zone".

The input is the patient feature/label table `patients_feature_label.csv`.
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
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lasso_svm_nested import (
    CORR_CUT,
    MAX_CANDIDATES,
    MAX_SCREEN_POOL,
    PARAM_GRID,
    SEED,
    bh_fdr,
    find_col,
    is_feature,
)
from lasso_svm_ordinal import safe_auc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

INPUT = Path(r"E:\DICOM\results\patients_feature_label.csv")
BASE_OUT = Path(r"E:\DICOM\reports\feature_selection_ordinal_ae")
OUT_A = BASE_OUT / "strategy_A_lasso_svm"
OUT_B = BASE_OUT / "strategy_B_drop_predict"

# patients_feature_label.csv uses Jan-26/Feb-26/Apr-26/May-26; normalise for
# display consistency with the existing 2026-01/02/04/05 reports.
COHORT_MAP = {"Jan-26": "2026-01", "Feb-26": "2026-02", "Apr-26": "2026-04", "May-26": "2026-05"}
GRAY_LOW, GRAY_HIGH = 0.4, 0.6


def fmt(x, digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_frame() -> tuple[pd.DataFrame, list[str], str, str]:
    df = pd.read_csv(INPUT, low_memory=False)
    pid = find_col(df, ["PatientID", "patient_id", "Patient_ID"])
    label_col = find_col(df, ["Label", "label"])
    cohort_col = find_col(df, ["cohort", "Cohort", "sequence", "序列"], False)
    if cohort_col is None:
        cohort_col = "_cohort"
        df[cohort_col] = "pooled"
    feature_names = [c for c in df.columns if is_feature(c)]
    valid = pd.to_numeric(df[label_col], errors="coerce").isin([0, 1, 2])
    work = df.loc[valid, [pid, cohort_col, label_col] + feature_names].copy()
    work["Label"] = pd.to_numeric(work[label_col], errors="coerce").astype(int)
    work["cohort"] = work[cohort_col].astype(str).map(COHORT_MAP).fillna(work[cohort_col].astype(str))
    work = work.drop_duplicates(pid).reset_index(drop=True)
    return work, feature_names, pid, label_col


def cohort_label_counts(work: pd.DataFrame) -> pd.DataFrame:
    return work.groupby(["cohort", "Label"]).size().unstack(fill_value=0).reset_index()


# --------------------------------------------------------------------------- #
# Binary screening / feature selection
# --------------------------------------------------------------------------- #
def screen_features_binary(X, y, names, max_candidates=MAX_CANDIDATES):
    """Mann-Whitney U + BH-FDR + |AUC-0.5| ranking + Spearman de-redundancy."""
    n = len(y)
    rows = []
    for j, name in enumerate(names):
        x = X[:, j]
        finite = np.isfinite(x)
        nf = int(finite.sum())
        if nf < max(20, int(0.5 * n)) or np.unique(x[finite]).size < 2:
            continue
        pos = finite & (y == 1)
        neg = finite & (y == 0)
        if pos.sum() < 3 or neg.sum() < 3 or np.unique(x[pos]).size < 2 or np.unique(x[neg]).size < 2:
            continue
        try:
            p = float(stats.mannwhitneyu(x[pos], x[neg], alternative="two-sided").pvalue)
        except Exception:
            p = np.nan
        try:
            raw = roc_auc_score(y[finite], x[finite])
            auc = max(float(raw), 1.0 - float(raw))
        except ValueError:
            auc = np.nan
        rows.append({"feature": name, "mannwhitney_p": p, "auc": auc, "observed_fraction": float(nf / n)})
    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        raise RuntimeError("No usable feature passed the binary nested screening step")
    stats_df["fdr"] = bh_fdr(stats_df["mannwhitney_p"].to_numpy())
    ranked = stats_df[stats_df["fdr"].fillna(1) < 0.05].sort_values(["auc", "fdr"], ascending=[False, True])
    if ranked.empty:
        ranked = stats_df.sort_values("auc", ascending=False)
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
    return selected, stats_df.sort_values(["selected_nested", "selection_order", "auc"], ascending=[False, True, False])


def fit_lasso_binary(X, y, names):
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
    coef = model.named_steps["lasso"].coef_[:, : len(names)]
    keep = np.flatnonzero(np.max(np.abs(coef), axis=0) > 1e-7)
    if len(keep) == 0:
        keep = np.argsort(np.max(np.abs(coef), axis=0))[::-1][: min(3, len(names))]
    return model, [names[i] for i in keep], coef


def make_svm_binary(C, gamma, probability=False):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=C, gamma=gamma, probability=probability, class_weight="balanced", random_state=SEED)),
        ]
    )


def fit_inner_binary(Xtr, ytr, Xval, yval, feature_names):
    selected, screen_df = screen_features_binary(Xtr, ytr, feature_names)
    idx = [feature_names.index(f) for f in selected]
    lasso, lasso_selected, _ = fit_lasso_binary(Xtr[:, idx], ytr, selected)
    if not lasso_selected:
        lasso_selected = selected[:3]
    sv_idx = [selected.index(f) for f in lasso_selected]
    scores = {}
    for param in PARAM_GRID:
        svm = make_svm_binary(param["C"], param["gamma"], probability=False)
        svm.fit(Xtr[:, idx][:, sv_idx], ytr)
        pred = svm.predict(Xval[:, idx][:, sv_idx])
        scores.setdefault((param["C"], str(param["gamma"])), []).append(float(balanced_accuracy_score(yval, pred)))
    return lasso, selected, lasso_selected, screen_df, scores


def tune_binary(X, y, feature_names):
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    keys = [(p["C"], str(p["gamma"])) for p in PARAM_GRID]
    scores = {k: [] for k in keys}
    for train, val in inner.split(X, y):
        _, selected, lasso_selected, screen_df, fold_scores = fit_inner_binary(X[train], y[train], X[val], y[val], feature_names)
        for k, vals in fold_scores.items():
            scores[k].extend(vals)
    mean_scores = {k: float(np.mean(v)) if v else -np.inf for k, v in scores.items()}
    best_key = max(mean_scores, key=mean_scores.get)
    best = {"C": float(best_key[0]), "gamma": best_key[1] if best_key[1] == "scale" else float(best_key[1])}
    return best, mean_scores


def positive_proba(model, X):
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


# --------------------------------------------------------------------------- #
# Binary metrics
# --------------------------------------------------------------------------- #
def binary_metrics(y, pred, score, proba=None):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    out = {
        "n": int(len(y)),
        "n_neg": int((y == 0).sum()),
        "n_pos": int((y == 1).sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity_recall": float(sens) if np.isfinite(sens) else np.nan,
        "specificity": float(spec) if np.isfinite(spec) else np.nan,
        "ppv": float(ppv) if np.isfinite(ppv) else np.nan,
        "npv": float(npv) if np.isfinite(npv) else np.nan,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": safe_auc(y, score),
    }
    if proba is not None:
        try:
            out["average_precision"] = float(average_precision_score(y, proba))
        except ValueError:
            out["average_precision"] = np.nan
    return out


def bootstrap_binary_ci(y, score, n_boot=1000):
    rng = np.random.default_rng(SEED)
    point = safe_auc(y, score)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        v = safe_auc(y[idx], score[idx])
        if np.isfinite(v):
            vals.append(v)
    return pd.DataFrame([{"metric": "auc", "estimate": point, "ci95_low": np.quantile(vals, 0.025), "ci95_high": np.quantile(vals, 0.975), "n_boot": len(vals)}])


# --------------------------------------------------------------------------- #
# Core nested leave-one-cohort-out loop
# --------------------------------------------------------------------------- #
def nested_loco_binary(work, feature_names, X, groups, train_idx, y_train, out_dir, predict_extra=False):
    """Run the nested LASSO->SVM binary pipeline with leave-one-cohort-out.

    `train_idx` are row indices of the training population; `y_train` is the
    binary target for those rows. When `predict_extra` is True, rows outside
    `train_idx` (e.g. Label 1) that fall into a test cohort are also scored.
    """
    fig_dir = out_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    train_groups = groups[train_idx]
    outer = GroupKFold(n_splits=len(np.unique(train_groups)))
    oof_rows, fold_rows, selection_rows = [], [], []
    for fold, (tr, te) in enumerate(outer.split(X[train_idx], y_train, train_groups), 1):
        train_row_idx = train_idx[tr]
        test_row_idx = train_idx[te]
        best, inner_scores = tune_binary(X[train_row_idx], y_train[tr], feature_names)
        selected, screen_df = screen_features_binary(X[train_row_idx], y_train[tr], feature_names)
        screen_df["outer_fold"] = fold
        screen_df.to_csv(out_dir / f"nested_outer{fold}_screening.csv", index=False, encoding="utf-8-sig")

        idx = [feature_names.index(f) for f in selected]
        lasso, lasso_selected, _ = fit_lasso_binary(X[train_row_idx][:, idx], y_train[tr], selected)
        selection_rows.extend({"fold": fold, "feature": f} for f in lasso_selected)
        sv_idx = [selected.index(f) for f in lasso_selected]

        svm = make_svm_binary(best["C"], best["gamma"], probability=True)
        svm.fit(X[train_row_idx][:, idx][:, sv_idx], y_train[tr])
        lasso.fit(X[train_row_idx][:, idx], y_train[tr])

        test_cohorts = set(groups[test_row_idx])
        predict_idx = test_row_idx
        if predict_extra:
            extra = np.array([i for i in range(len(work)) if i not in set(train_idx) and groups[i] in test_cohorts])
            predict_idx = np.concatenate([test_row_idx, extra])

        lasso_pred = lasso.predict(X[predict_idx][:, idx]).astype(int)
        svm_pred = svm.predict(X[predict_idx][:, idx][:, sv_idx]).astype(int)
        lasso_score = lasso.decision_function(X[predict_idx][:, idx])
        svm_score = svm.decision_function(X[predict_idx][:, idx][:, sv_idx])
        lasso_proba = positive_proba(lasso, X[predict_idx][:, idx])
        svm_proba = positive_proba(svm, X[predict_idx][:, idx][:, sv_idx])

        # fold metrics are computed on the training-population test rows only
        pos_of = {int(r): i for i, r in enumerate(predict_idx)}
        te_pos = np.array([pos_of[int(r)] for r in test_row_idx])
        lm = binary_metrics(y_train[te], lasso_pred[te_pos], lasso_score[te_pos], lasso_proba[te_pos])
        sm = binary_metrics(y_train[te], svm_pred[te_pos], svm_score[te_pos], svm_proba[te_pos])
        for model, met in [("LASSO", lm), ("SVM", sm)]:
            fold_rows.append(
                {
                    "fold": fold,
                    "test_cohort": ";".join(sorted(test_cohorts)),
                    "model": model,
                    "best_params": json.dumps(best),
                    "n_screened": len(selected),
                    "n_lasso_selected": len(lasso_selected),
                    "inner_best_balanced_accuracy": max(inner_scores.values()),
                    **met,
                }
            )
        for i, row_i in enumerate(predict_idx):
            oof_rows.append(
                {
                    "row_index": int(row_i),
                    "PatientID": str(work.iloc[row_i]["PatientID"]),
                    "cohort": groups[row_i],
                    "Label": int(work.iloc[row_i]["Label"]),
                    "in_train": bool(row_i in set(train_idx)),
                    "lasso_pred": int(lasso_pred[i]),
                    "svm_pred": int(svm_pred[i]),
                    "lasso_score": float(lasso_score[i]),
                    "svm_score": float(svm_score[i]),
                    "lasso_proba": float(lasso_proba[i]),
                    "svm_proba": float(svm_proba[i]),
                }
            )

    oof = pd.DataFrame(oof_rows).sort_values("row_index").reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows)
    selection_obs = pd.DataFrame(selection_rows)
    sf = pd.DataFrame({"feature": feature_names})
    if selection_obs.empty:
        sf["frequency"] = 0
    else:
        counts = selection_obs.groupby("feature").size()
        sf = sf.join(counts.rename("frequency"), on="feature").fillna(0)
    sf["frequency"] = sf["frequency"].astype(int)
    sf["selection_rate"] = sf["frequency"] / len(np.unique(train_groups))
    sf.sort_values(["frequency", "feature"], ascending=[False, True], inplace=True)
    return oof, fold_df, sf


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def data_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def fig_roc(y, oof, title, fname):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for model, color in [("lasso", "#1f77b4"), ("svm", "#d62728")]:
        sc = oof[f"{model}_score"].to_numpy(float)
        fpr, tpr, _ = roc_curve(y, sc)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{model.upper()} AUC={safe_auc(y, sc):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title=title)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


def fig_pr(y, oof, title, fname):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for model, color in [("lasso", "#1f77b4"), ("svm", "#d62728")]:
        proba = oof[f"{model}_proba"].to_numpy(float)
        prec, rec, _ = precision_recall_curve(y, proba)
        ax.plot(rec, prec, lw=2, color=color, label=f"{model.upper()} AP={average_precision_score(y, proba):.3f}")
    baseline = y.mean()
    ax.axhline(baseline, ls="--", lw=0.8, color="gray", label=f"prevalence={baseline:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title=title)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def fig_confusion(y, pred, title, fname):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    labels = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.1%})", ha="center", va="center", color="white" if cmn[i, j] > 0.5 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="True", title=title)
    fig.colorbar(im, ax=ax, label="Row proportion")
    fig.tight_layout()
    return fig


def fig_frequency(sf, fname, title="LASSO feature-selection stability"):
    top = sf.sort_values("frequency", ascending=False).head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.barh(top["feature"], top["frequency"], color="#2ca02c")
    ax.set_xlim(0, max(1, int(sf["frequency"].max())))
    ax.set_xlabel("Outer folds selected by LASSO")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def fig_risk_distribution(oof, fname, proba_col="lasso_proba", title="Drop & Predict risk-score distribution"):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bins = np.linspace(0, 1, 31)
    for label, color, name in [(0, "#1f77b4", "Label 0 (stable)"), (1, "#ff7f0e", "Label 1 (held-out)"), (2, "#d62728", "Label 2 (acute)")]:
        sub = oof[oof["Label"] == label][proba_col].dropna().to_numpy(float)
        if len(sub):
            ax.hist(sub, bins=bins, alpha=0.45, color=color, label=f"{name} (n={len(sub)})", density=True)
    ax.axvspan(GRAY_LOW, GRAY_HIGH, color="gold", alpha=0.25, label=f"gray zone [{GRAY_LOW:.1f}, {GRAY_HIGH:.1f}]")
    ax.set(xlabel="Risk score (LASSO probability of Label 2)", ylabel="Density", title=title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def fig_risk_box(oof, fname, proba_col="lasso_proba", title="Risk score by true label"):
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    data = [oof[oof["Label"] == c][proba_col].dropna().to_numpy(float) for c in [0, 1, 2]]
    ax.boxplot(data, labels=["Label 0", "Label 1", "Label 2"], showmeans=True)
    ax.axhspan(GRAY_LOW, GRAY_HIGH, color="gold", alpha=0.2)
    ax.axhline(0.5, ls="--", lw=0.8, color="gray")
    ax.set(ylabel="Risk score (LASSO probability)", title=title)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def save_figs(figs, fig_dir):
    paths = []
    for fig, fname in figs:
        p = fig_dir / fname
        fig.savefig(p, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def table_html(df, n=30):
    if df is None or df.empty:
        return "<p>No rows.</p>"
    return df.head(n).to_html(index=False, border=0, classes="data", float_format=lambda x: f"{x:.4f}")


CSS = """<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1450px;margin:30px auto;padding:0 24px;color:#222;line-height:1.5}h1,h2,h3{color:#17365d}table.data{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 20px}table.data th,table.data td{border:1px solid #d9e2f3;padding:5px 7px;text-align:left}table.data th{background:#eaf2f8}figure{display:inline-block;vertical-align:top;margin:8px;width:46%;text-align:center}figure img{max-width:100%;height:auto;border:1px solid #ddd}figcaption{font-size:12px;color:#555}.note{background:#fff7e6;border-left:4px solid #f0ad4e;padding:10px 14px}.ok{background:#e8f6e8;border-left:4px solid #2e9e5b;padding:10px 14px}code{background:#f3f3f3;padding:2px 4px}</style>"""


def imgs_html(paths):
    return "".join(f'<figure><img src="{data_uri(p)}"><figcaption>{html.escape(p.stem)}</figcaption></figure>' for p in paths)


# --------------------------------------------------------------------------- #
# Strategy A
# --------------------------------------------------------------------------- #
def run_strategy_A():
    OUT_A.mkdir(parents=True, exist_ok=True)
    work, feature_names, pid, _ = load_frame()
    X = work[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    groups = work["cohort"].to_numpy(str)
    y_bin = (work["Label"].to_numpy(int) >= 1).astype(int)
    train_idx = np.arange(len(work))

    counts = work.assign(risk=pd.Series(y_bin).map({0: "Label 0 (stable)", 1: "Label 1+2 (risk)"})).groupby(["cohort", "risk"]).size().unstack(fill_value=0).reset_index()
    counts.to_csv(OUT_A / "cohort_binary_counts.csv", index=False, encoding="utf-8-sig")

    oof, fold_df, sf = nested_loco_binary(work, feature_names, X, groups, train_idx, y_bin, OUT_A)
    oof.to_csv(OUT_A / "nested_oof_predictions_loco.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(OUT_A / "nested_loco_fold_metrics.csv", index=False, encoding="utf-8-sig")
    sf.to_csv(OUT_A / "nested_lasso_selection_frequency.csv", index=False, encoding="utf-8-sig")

    y = oof["Label"].to_numpy(int)
    y_bin_all = (y >= 1).astype(int)
    pooled = pd.DataFrame(
        [
            {"model": "LASSO", **binary_metrics(y_bin_all, oof["lasso_pred"].to_numpy(int), oof["lasso_score"].to_numpy(float), oof["lasso_proba"].to_numpy(float))},
            {"model": "SVM", **binary_metrics(y_bin_all, oof["svm_pred"].to_numpy(int), oof["svm_score"].to_numpy(float), oof["svm_proba"].to_numpy(float))},
        ]
    )
    pooled.to_csv(OUT_A / "nested_pooled_oof_metrics.csv", index=False, encoding="utf-8-sig")
    boot = pd.concat(
        [
            bootstrap_binary_ci(y_bin_all, oof["lasso_score"].to_numpy(float)).assign(model="LASSO"),
            bootstrap_binary_ci(y_bin_all, oof["svm_score"].to_numpy(float)).assign(model="SVM"),
        ],
        ignore_index=True,
    )
    boot.to_csv(OUT_A / "nested_pooled_oof_auc_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    cm = pd.DataFrame(confusion_matrix(y_bin_all, oof["svm_pred"].to_numpy(int), labels=[0, 1]), index=["true_stable", "true_risk"], columns=["pred_stable", "pred_risk"])
    cm.to_csv(OUT_A / "svm_oof_confusion_matrix.csv", encoding="utf-8-sig")

    figs = [
        (fig_roc(y_bin_all, oof, "Strategy A OOF ROC (stable vs risk)", "strategy_A_oof_roc.png"), "strategy_A_oof_roc.png"),
        (fig_pr(y_bin_all, oof, "Strategy A OOF precision-recall", "strategy_A_oof_pr.png"), "strategy_A_oof_pr.png"),
        (fig_confusion(y_bin_all, oof["svm_pred"].to_numpy(int), "Strategy A SVM confusion matrix", "strategy_A_confusion.png"), "strategy_A_confusion.png"),
        (fig_frequency(sf, "strategy_A_lasso_frequency.png"), "strategy_A_lasso_frequency.png"),
    ]
    paths = save_figs(figs, OUT_A / "figs")

    summary = {
        "strategy": "A",
        "task": "Label 0 vs (Label 1 + Label 2)",
        "n": len(work),
        "n_features": len(feature_names),
        "pooled_oof": pooled.to_dict(orient="records"),
        "bootstrap_auc": boot.to_dict(orient="records"),
    }
    (OUT_A / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = build_report_A(work, counts, pooled, boot, fold_df, sf, paths)
    (OUT_A / "lasso_svm_strategy_A_report.html").write_text(report, encoding="utf-8")
    (OUT_A / "strategy_A_report.html").write_text(report, encoding="utf-8")
    print(json.dumps({"strategy": "A", "n": len(work), "pooled_oof": pooled.to_dict(orient="records"), "output": str(OUT_A)}, ensure_ascii=False, indent=2))
    return OUT_A


def build_report_A(work, counts, pooled, boot, fold_df, sf, paths):
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Strategy A - LASSO SVM early warning</title>{CSS}</head><body>
<h1>策略 A：激进预警模式 — LASSO → SVM 二分类报告</h1>
<p>任务：<b>Label 0（稳定代偿期） vs (Label 1 + Label 2)（结构易损 + 急性失代偿风险池）</b>。任何结构恶化的苗头都被并入“风险池”，作为第一道高敏感性筛查防线。</p>
<h2>1. 数据与方法</h2>
<ul>
<li>有效样本：<b>{len(work)}</b>；原始影像组学/气道特征：<b>{len(sf)}</b>。</li>
<li>二分类目标：<code>risk = 1</code> 当 Label ∈ {{1, 2}}，否则 <code>risk = 0</code>。</li>
<li>外层验证：2026-01/02/04/05 四个序列 leave-one-cohort-out；每个外层训练折内重新完成 Mann-Whitney U 单变量筛选、BH-FDR、Spearman 去冗余、L1 logistic（LASSO）特征选择与 SVM C/gamma 调参。</li>
<li>SVM：RBF kernel、<code>class_weight='balanced'</code>；未使用 SMOTE。</li>
</ul>
<h3>各序列二分类分布</h3>{table_html(counts)}
<h2>2. pooled OOF 性能（主结果）</h2>{table_html(pooled)}
<p>AUC 使用 decision score 计算；sensitivity=风险池召回率（筛查系统的核心指标），specificity=稳定组正确识别率；average_precision 使用 SVM 概率。</p>
<h3>AUC bootstrap 95% CI</h3>{table_html(boot)}
<h2>3. 外层折叠明细</h2>{table_html(fold_df, 20)}
<h2>4. LASSO 特征选择稳定性</h2>{table_html(sf.head(30))}
<h2>5. 图表</h2>{imgs_html(paths)}
<h2>6. 结论与限制</h2>
<div class="note">该策略的优化目标是<b>高敏感性</b>（宁可错杀、不可漏放）。解读时应优先关注 sensitivity 与 NPV：若 sensitivity 高而 specificity 相对较低，正符合第一道筛查防线的定位；临床可对“风险池”阳性者进一步做连续风险评估（见策略 B）。</div>
<p>限制：Label 1/2 由 ICD 与临床文本构造，属代理终点；nested OOF 才是主要泛化估计。完整机器可读结果（OOF 预测、折叠指标、bootstrap CI、选择频率、混淆矩阵）位于本报告同目录。</p>
</body></html>"""


# --------------------------------------------------------------------------- #
# Strategy B
# --------------------------------------------------------------------------- #
def run_strategy_B():
    OUT_B.mkdir(parents=True, exist_ok=True)
    work, feature_names, pid, _ = load_frame()
    X = work[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    groups = work["cohort"].to_numpy(str)
    label = work["Label"].to_numpy(int)

    train_mask = np.isin(label, [0, 2])
    train_idx = np.flatnonzero(train_mask)
    y_train = (label[train_idx] == 2).astype(int)

    counts = work.assign(cohort_label=work["Label"].map({0: "0_stable", 1: "1_structural", 2: "2_acute"})).groupby(["cohort", "cohort_label"]).size().unstack(fill_value=0).reset_index()
    counts.to_csv(OUT_B / "cohort_label_counts.csv", index=False, encoding="utf-8-sig")

    oof, fold_df, sf = nested_loco_binary(work, feature_names, X, groups, train_idx, y_train, OUT_B, predict_extra=True)
    oof.to_csv(OUT_B / "nested_oof_predictions_loco.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(OUT_B / "nested_loco_fold_metrics.csv", index=False, encoding="utf-8-sig")
    sf.to_csv(OUT_B / "nested_lasso_selection_frequency.csv", index=False, encoding="utf-8-sig")

    # binary OOF metrics on the training population (Label 0 vs 2) only
    train_oof = oof[oof["in_train"]].reset_index(drop=True)
    y02 = (train_oof["Label"] == 2).astype(int).to_numpy(int)
    pooled = pd.DataFrame(
        [
            {"model": "LASSO", **binary_metrics(y02, train_oof["lasso_pred"].to_numpy(int), train_oof["lasso_score"].to_numpy(float), train_oof["lasso_proba"].to_numpy(float))},
            {"model": "SVM", **binary_metrics(y02, train_oof["svm_pred"].to_numpy(int), train_oof["svm_score"].to_numpy(float), train_oof["svm_proba"].to_numpy(float))},
        ]
    )
    pooled.to_csv(OUT_B / "nested_pooled_oof_metrics.csv", index=False, encoding="utf-8-sig")
    boot = pd.concat(
        [
            bootstrap_binary_ci(y02, train_oof["lasso_score"].to_numpy(float)).assign(model="LASSO"),
            bootstrap_binary_ci(y02, train_oof["svm_score"].to_numpy(float)).assign(model="SVM"),
        ],
        ignore_index=True,
    )
    boot.to_csv(OUT_B / "nested_pooled_oof_auc_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    cm = pd.DataFrame(confusion_matrix(y02, train_oof["svm_pred"].to_numpy(int), labels=[0, 1]), index=["true_0", "true_2"], columns=["pred_0", "pred_2"])
    cm.to_csv(OUT_B / "svm_oof_confusion_matrix.csv", encoding="utf-8-sig")

    # gray-zone analysis on held-out Label 1 (and reference distributions)
    # Primary risk score = LASSO (L1 logistic) probability, which is monotonic
    # and well-behaved; SVM Platt probability is miscalibrated under balanced
    # class weights and is reported only as an auxiliary.
    risk_summary = risk_score_summary(oof, "lasso_proba")
    risk_summary.to_csv(OUT_B / "label_risk_score_summary.csv", index=False, encoding="utf-8-sig")
    risk_summary_svm = risk_score_summary(oof, "svm_proba")
    risk_summary_svm.to_csv(OUT_B / "label_risk_score_summary_svm_proba.csv", index=False, encoding="utf-8-sig")

    figs = [
        (fig_roc(y02, train_oof, "Strategy B OOF ROC (Label 0 vs 2)", "strategy_B_oof_roc.png"), "strategy_B_oof_roc.png"),
        (fig_confusion(y02, train_oof["svm_pred"].to_numpy(int), "Strategy B SVM confusion (0 vs 2)", "strategy_B_confusion.png"), "strategy_B_confusion.png"),
        (fig_risk_distribution(oof, "strategy_B_risk_distribution.png", "lasso_proba"), "strategy_B_risk_distribution.png"),
        (fig_risk_box(oof, "strategy_B_risk_box.png", "lasso_proba"), "strategy_B_risk_box.png"),
        (fig_frequency(sf, "strategy_B_lasso_frequency.png"), "strategy_B_lasso_frequency.png"),
    ]
    paths = save_figs(figs, OUT_B / "figs")

    # final canonical model fit on all Label 0 + 2, then score everything
    selected_final, screen_final = screen_features_binary(X[train_idx], y_train, feature_names)
    idx_f = [feature_names.index(f) for f in selected_final]
    lasso_f, lasso_sel_f, _ = fit_lasso_binary(X[train_idx][:, idx_f], y_train, selected_final)
    sv_idx_f = [selected_final.index(f) for f in lasso_sel_f]
    best_f, _ = tune_binary(X[train_idx], y_train, feature_names)
    svm_f = make_svm_binary(best_f["C"], best_f["gamma"], probability=True)
    svm_f.fit(X[train_idx][:, idx_f][:, sv_idx_f], y_train)
    final_lasso_proba = positive_proba(lasso_f, X[:, idx_f])
    final_proba = positive_proba(svm_f, X[:, idx_f][:, sv_idx_f])
    final_score = svm_f.decision_function(X[:, idx_f][:, sv_idx_f])
    final_df = pd.DataFrame({"PatientID": work["PatientID"], "cohort": groups, "Label": label, "lasso_risk_proba": final_lasso_proba, "svm_risk_proba": final_proba, "svm_score": final_score})
    final_df.to_csv(OUT_B / "final_model_risk_scores.csv", index=False, encoding="utf-8-sig")
    joblib.dump({"lasso": lasso_f, "svm": svm_f, "candidate_features": feature_names, "selected_features": lasso_sel_f, "classes": [0, 1]}, OUT_B / "final_lasso_svm_models.joblib")

    summary = {
        "strategy": "B",
        "task": "Drop & Predict: train Label 0 vs 2, score held-out Label 1",
        "n_total": len(work),
        "n_train": int(train_mask.sum()),
        "n_label1_heldout": int((~train_mask).sum()),
        "n_features": len(feature_names),
        "pooled_oof_binary": pooled.to_dict(orient="records"),
        "bootstrap_auc": boot.to_dict(orient="records"),
        "risk_score_summary_lasso_proba": risk_summary.to_dict(orient="records"),
        "risk_score_summary_svm_proba": risk_summary_svm.to_dict(orient="records"),
        "final_selected_features": lasso_sel_f,
        "final_svm_best_params": best_f,
    }
    (OUT_B / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = build_report_B(work, counts, pooled, boot, fold_df, sf, risk_summary, risk_summary_svm, paths, lasso_sel_f, best_f, n_train=int(train_mask.sum()), n_label1=int((~train_mask).sum()))
    (OUT_B / "lasso_svm_strategy_B_report.html").write_text(report, encoding="utf-8")
    (OUT_B / "strategy_B_report.html").write_text(report, encoding="utf-8")
    print(json.dumps({"strategy": "B", "n_total": len(work), "n_label1": int((~train_mask).sum()), "pooled_oof_binary": pooled.to_dict(orient="records"), "risk_summary_lasso": risk_summary.to_dict(orient="records"), "output": str(OUT_B)}, ensure_ascii=False, indent=2))
    return OUT_B


def risk_score_summary(oof, proba_col="lasso_proba"):
    rows = []
    for label in [0, 1, 2]:
        sub = oof[oof["Label"] == label][proba_col].dropna().to_numpy(float)
        if len(sub) == 0:
            continue
        rows.append(
            {
                "label": int(label),
                "n": int(len(sub)),
                "median": float(np.median(sub)),
                "mean": float(np.mean(sub)),
                "q25": float(np.quantile(sub, 0.25)),
                "q75": float(np.quantile(sub, 0.75)),
                "pct_in_gray_zone_0.4_0.6": float(np.mean((sub >= GRAY_LOW) & (sub <= GRAY_HIGH))),
                "pct_below_0.4": float(np.mean(sub < GRAY_LOW)),
                "pct_above_0.6": float(np.mean(sub > GRAY_HIGH)),
                "pct_predicted_risk_at_0.5": float(np.mean(sub >= 0.5)),
            }
        )
    return pd.DataFrame(rows)


def build_report_B(work, counts, pooled, boot, fold_df, sf, risk_summary, risk_summary_svm, paths, final_selected, best_params, n_train=None, n_label1=None):
    if n_train is None:
        n_train = 0
    if n_label1 is None:
        n_label1 = 0
    label1 = risk_summary[risk_summary["label"] == 1]
    label0 = risk_summary[risk_summary["label"] == 0]
    label2 = risk_summary[risk_summary["label"] == 2]
    gray_note = "Label 1 样本数量为 0，无法评估灰度区间。" if label1.empty else (
        f"Label 1 风险分中位数 {label1['median'].iloc[0]:.3f}（Label 0 为 {label0['median'].iloc[0]:.3f}，Label 2 为 {label2['median'].iloc[0]:.3f}），"
        f"{label1['pct_in_gray_zone_0.4_0.6'].iloc[0]:.1%} 落在 0.4–0.6 灰度预警区间，"
        f"{label1['pct_below_0.4'].iloc[0]:.1%} 落在 <0.4（偏向稳定），"
        f"{label1['pct_above_0.6'].iloc[0]:.1%} 落在 >0.6（偏向急性）。"
    )
    svm_table = risk_summary_svm.to_html(index=False, border=0, classes="data", float_format=lambda x: f"{x:.4f}") if risk_summary_svm is not None and not risk_summary_svm.empty else "<p>No rows.</p>"
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Strategy B - Drop and Predict</title>{CSS}</head><body>
<h1>策略 B：连续概率映射（Drop &amp; Predict）— LASSO → SVM 报告</h1>
<p>任务：训练阶段<b>彻底剔除 Label 1</b>，只用最纯粹的黑（Label 2 急性失代偿）与白（Label 0 稳定代偿）建立超平面；推理阶段输出 0–1 连续风险概率，再观察未参与训练的 Label 1 是否集中在 <b>0.4–0.6 灰度预警区间</b>。</p>
<h2>1. 数据与方法</h2>
<ul>
<li>全部有效样本：<b>{len(work)}</b>；其中训练集（Label 0 ∪ 2）：<b>{n_train}</b>；被剔除的 Label 1（仅用于 held-out 打分）：<b>{n_label1}</b>。</li>
<li>原始影像组学/气道特征：<b>{len(sf)}</b>。</li>
</ul>
<p>外层 leave-one-cohort-out：每个训练折只用 Label 0 与 Label 2；测试折内出现的 Label 1 样本也被打分（模型从未见过任何 Label 1），从而得到严格的 held-out 风险分。</p>
<h3>各序列标签分布</h3>{table_html(counts)}
<h2>2. Label 0 vs 2 的 pooled OOF 性能</h2>{table_html(pooled)}
<p>AUC 使用 decision score 计算；average_precision 使用概率。</p>
<h3>AUC bootstrap 95% CI</h3>{table_html(boot)}
<h2>3. Drop &amp; Predict 灰度区间分析（主风险分 = LASSO L1 logistic 概率）</h2>
<p>主风险分采用 <b>LASSO（L1 logistic 回归）</b>对“Label 2（急性）”类别的概率输出：它是严格单调、数值稳定的 0–1 连续量，可作为“稳定↔急性”风险轴上的连续定位。下表为 held-out OOF 风险分按真实标签汇总：</p>
{risk_summary.to_html(index=False, border=0, classes="data", float_format=lambda x: f"{x:.4f}")}
<div class="ok">{gray_note}</div>
<h4>附：SVM Platt 概率（辅助，注意失校准）</h4>
<p>RBF-SVM 在 <code>class_weight='balanced'</code> 下，其 Platt 校准概率会系统偏离真实患病率（本队列中甚至出现 Label 0 的中位概率高于 Label 2 的反向现象），因此<b>不用于</b>灰度区间判定，仅作参考：</p>
{svm_table}
<h2>4. 外层折叠明细（Label 0 vs 2）</h2>{table_html(fold_df, 20)}
<h2>5. LASSO 特征选择稳定性</h2>{table_html(sf.head(30))}
<h2>6. 图表</h2>{imgs_html(paths)}
<h2>7. 最终部署模型（apparent）</h2>
<p>最终模型在全体 Label 0 ∪ 2 上拟合，最优 SVM 参数 <code>{html.escape(str(best_params))}</code>；LASSO 保留特征：{', '.join(f'<code>{html.escape(x)}</code>' for x in final_selected)}。对全体样本（含 Label 1）的风险分已写入 <code>final_model_risk_scores.csv</code>。</p>
<h2>8. 结论与限制</h2>
<p>若 Label 1 的风险分确实集中在 0.4–0.6，则说明“Drop &amp; Predict”能够在不把过渡期样本混入训练的前提下，对其给出连续性风险定位——既保证黑白分类的训练纯度，又保留对结构易损期的早期预警能力。若 Label 1 分数明显偏向两端，则提示结构易损期在现有特征下更接近某一端，需进一步探索专属特征。</p>
<p>限制：Label 1/2 为 ICD/临床文本构造的代理终点；OOF 风险分由各折模型分别输出，跨折概率可能存在轻微校准差异；LASSO 概率在 balanced 权重下反映“相对风险”而非患病率校准概率；最终模型为 apparent 拟合，不应作为泛化指标。</p>
</body></html>"""


if __name__ == "__main__":
    import sys

    strategy = sys.argv[1] if len(sys.argv) > 1 else "A"
    if strategy.upper() == "A":
        run_strategy_A()
    elif strategy.upper() == "B":
        run_strategy_B()
    else:
        print("usage: lasso_svm_binary.py [A|B]")
