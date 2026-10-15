"""Builds verified validation + statistical summaries from the SAVED model.
Does not retrain. Writes JSON files to reports/ only if the saved metrics are reproduced."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "application_train.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
BAND_EDGES = [0.0, 0.10, 0.30, 0.50, 1.0000001]
BAND_LABELS = ["Low", "Moderate", "Elevated", "High"]


def clean(o):
    """Convert NumPy / pandas types into plain JSON-safe Python types."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def _fallback(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def write_json(path, obj):
    path.write_text(json.dumps(clean(obj), indent=2, default=_fallback), encoding="utf-8")


def add_ratios(df):
    inc = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    goods = df["AMT_GOODS_PRICE"].replace(0, np.nan)
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / inc
    df["ANNUITY_INCOME_RATIO"] = (df["AMT_ANNUITY"] * 12) / inc
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / goods
    for c in ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_GOODS_RATIO"]:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    return df


def thin(arrs, n=300):
    idx = np.unique(np.linspace(0, len(arrs[0]) - 1, n).astype(int))
    return [np.round(a[idx], 4) for a in arrs]


print("Loading data and model...")
df = pd.read_csv(DATA)
model = joblib.load(MODELS / "credence_risk_model.joblib")
cols = json.loads((MODELS / "feature_columns.json").read_text())
saved = json.loads((MODELS / "model_metrics.json").read_text())

df = add_ratios(df)
missing = [c for c in cols if c not in df.columns]
if missing:
    sys.exit(f"STOP: columns missing from data: {missing}")

X, y = df[cols], df["TARGET"]
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
proba = model.predict_proba(X_va)[:, 1]

roc = roc_auc_score(y_va, proba)
ap = average_precision_score(y_va, proba)
print(f"Validation rows: {len(y_va)} (saved: {saved['validation_rows']})")
print(f"ROC-AUC  recomputed {roc:.4f} | saved {saved['roc_auc']:.4f}")
print(f"Avg Prec recomputed {ap:.4f} | saved {saved['average_precision']:.4f}")
if (abs(roc - saved["roc_auc"]) > 1e-4 or abs(ap - saved["average_precision"]) > 1e-4
        or len(y_va) != saved["validation_rows"]):
    sys.exit("STOP: could not reproduce saved metrics. Nothing was written. Send me this output.")
print("Reproduction check PASSED.")

# ---------- validation summary ----------
pred = (proba >= 0.5).astype(int)
fpr, tpr, _ = roc_curve(y_va, proba)
prec, rec, _ = precision_recall_curve(y_va, proba)
fpr_t, tpr_t = thin([fpr, tpr])
rec_t, prec_t = thin([rec, prec])
frac_pos, mean_pred = calibration_curve(y_va, proba, n_bins=10, strategy="quantile")

thr_rows = []
for t in np.round(np.arange(0.1, 0.91, 0.1), 2):
    p = (proba >= t).astype(int)
    thr_rows.append({"threshold": float(t),
                     "precision": precision_score(y_va, p, zero_division=0),
                     "recall": recall_score(y_va, p, zero_division=0),
                     "f1": f1_score(y_va, p, zero_division=0),
                     "flagged_share": p.mean()})

bands = pd.cut(pd.Series(proba), bins=BAND_EDGES, labels=BAND_LABELS, right=False)
band_df = pd.DataFrame({"band": bands.astype(str).values, "y": y_va.values})
band_rows = []
for b in BAND_LABELS:
    sub = band_df.loc[band_df.band == b, "y"]
    band_rows.append({"band": b, "applicants": int(len(sub)),
                      "actual_rate": float(sub.mean()) if len(sub) else None})

validation = {
    "validation_rows": int(len(y_va)), "prevalence": float(y_va.mean()),
    "roc_auc": roc, "average_precision": ap,
    "at_threshold_0_5": {"precision": precision_score(y_va, pred),
                         "recall": recall_score(y_va, pred),
                         "f1": f1_score(y_va, pred)},
    "roc_curve": {"fpr": fpr_t, "tpr": tpr_t},
    "pr_curve": {"recall": rec_t, "precision": prec_t},
    "calibration": {"mean_predicted": mean_pred, "observed_rate": frac_pos},
    "thresholds": thr_rows, "risk_bands": band_rows,
    "mean_predicted_probability": float(proba.mean()),
}
write_json(REPORTS / "validation_summary.json", validation)

# ---------- statistical analysis (full application_train data) ----------
d = df.copy()
d["DAYS_EMPLOYED"] = d["DAYS_EMPLOYED"].replace(365243, np.nan)  # analysis copy only
d["AGE_YEARS"] = -d["DAYS_BIRTH"] / 365.25
num_cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "AGE_YEARS",
            "DAYS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "CNT_CHILDREN",
            "REGION_POPULATION_RELATIVE", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO",
            "CREDIT_GOODS_RATIO"]
cat_cols = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
            "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE", "OCCUPATION_TYPE"]

print("Computing statistical tests...")
numeric = []
for c in num_cols:
    a, b = d.loc[d.TARGET == 1, c].dropna(), d.loc[d.TARGET == 0, c].dropna()
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    auc = u.statistic / (len(a) * len(b))
    numeric.append({"feature": c, "missing_pct": d[c].isna().mean() * 100,
                    "mean_target1": a.mean(), "mean_target0": b.mean(),
                    "median_target1": a.median(), "median_target0": b.median(),
                    "mann_whitney_p": u.pvalue, "rank_biserial": 2 * auc - 1,
                    "univariate_auc": auc,
                    "spearman_with_target": d[[c, "TARGET"]].corr(method="spearman").iloc[0, 1]})

categorical = []
for c in cat_cols:
    s = d[c].fillna("Missing").astype(str)
    ct = pd.crosstab(s, d.TARGET)
    chi2, p, _, _ = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    rates = [{"category": str(k), "applicants": int(ct.loc[k].sum()),
              "default_rate": ct.loc[k, 1] / ct.loc[k].sum()} for k in ct.index]
    categorical.append({"feature": c, "chi2_p": p, "cramers_v": v, "rates": rates})

corr = d[num_cols + ["TARGET"]].corr(method="spearman")
miss = (d[cols].isna().mean() * 100).sort_values(ascending=False).head(20)

statistical = {
    "rows": int(len(d)), "default_rate": float(d.TARGET.mean()),
    "class_counts": {"0": int((d.TARGET == 0).sum()), "1": int((d.TARGET == 1).sum())},
    "numeric": numeric, "categorical": categorical,
    "correlation": {"labels": list(corr.columns), "matrix": corr.values.round(4).tolist()},
    "missing_top20": {"features": list(miss.index), "pct": miss.round(2).tolist()},
    "days_employed_anomaly_rows": int((df["DAYS_EMPLOYED"] == 365243).sum()),
}
write_json(REPORTS / "statistical_summary.json", statistical)

print("\nWritten: reports/validation_summary.json, reports/statistical_summary.json")
print(f"Prevalence (validation): {validation['prevalence']:.4f}")
print("Threshold 0.5 ->", {k: round(float(v), 4) for k, v in validation["at_threshold_0_5"].items()})
print("Risk bands:", [(r["band"], r["applicants"],
                       round(r["actual_rate"], 3) if r["actual_rate"] is not None else None)
                      for r in band_rows])