"""Builds explainability artifacts from the SAVED model. Does not retrain.
Writes reports/explainability.json."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.risk_scoring import engineer_features  # noqa: E402

DATA = ROOT / "datasets" / "application_train.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def to_dense(m):
    return m.toarray() if sparse.issparse(m) else np.asarray(m)


print("Loading model and data...")
model = joblib.load(MODELS / "credence_risk_model.joblib")
cols = json.loads((MODELS / "feature_columns.json").read_text())
saved = json.loads((MODELS / "model_metrics.json").read_text())
pre = model.named_steps["preprocessor"]
clf = model.named_steps["classifier"]

num_features, cat_features = [], []
for name, _, c in pre.transformers_:
    if name == "num":
        num_features = list(c)
    elif name == "cat":
        cat_features = list(c)

names = [str(n) for n in pre.get_feature_names_out()]
coef = np.asarray(clf.coef_, dtype=float).ravel()
intercept = float(clf.intercept_[0])
if len(names) != len(coef):
    sys.exit(f"STOP: {len(names)} transformed features but {len(coef)} coefficients.")

cats_sorted = sorted(cat_features, key=len, reverse=True)
original, display = [], []
for n in names:
    if n.startswith("num__"):
        original.append(n[5:])
        display.append(n[5:])
    elif n.startswith("cat__"):
        rest = n[5:]
        feat = next((c for c in cats_sorted if rest.startswith(c + "_")), rest)
        original.append(feat)
        display.append(f"{feat} = {rest[len(feat) + 1:]}" if feat != rest else rest)
    else:
        original.append(n)
        display.append(n)

df = engineer_features(pd.read_csv(DATA))
X, y = df[cols], df["TARGET"]
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
if len(X_va) != saved["validation_rows"]:
    sys.exit("STOP: validation split size differs from the saved metrics.")

print("Computing training-split feature means...")
Xt_tr = pre.transform(X_tr)
mean = np.asarray(Xt_tr.mean(axis=0)).ravel()
base_value = intercept + float(coef @ mean)

print("Computing global SHAP values on 10,000 validation applicants...")
sample = X_va.sample(10000, random_state=0)
Xs = to_dense(pre.transform(sample))
contrib = (Xs - mean) * coef
agg = pd.DataFrame(contrib, columns=original).T.groupby(level=0).sum().T
mean_abs = agg.abs().mean().sort_values(ascending=False)

kind = {f: "numeric" for f in num_features}
kind.update({f: "categorical" for f in cat_features})
coef_by_feature = {o: float(coef[i]) for i, o in enumerate(original) if kind.get(o) == "numeric"}
global_rows = [{"feature": f, "mean_abs": float(v), "kind": kind.get(f, "numeric"),
                "coef": coef_by_feature.get(f)} for f, v in mean_abs.items()]

print("Cross-checking against the shap library...")
check = {"verified": False, "shap_version": None, "max_abs_diff": None, "note": ""}
try:
    import shap

    check["shap_version"] = shap.__version__
    test = Xs[:300]
    explainer = shap.LinearExplainer(clf, shap.maskers.Independent(mean.reshape(1, -1)))
    values = explainer.shap_values(test)
    values = np.asarray(values[1] if isinstance(values, list) else values)
    manual = (test - mean) * coef
    diff = float(np.abs(values - manual).max())
    exp_val = float(np.ravel(explainer.expected_value)[0])
    check["max_abs_diff"] = diff
    check["expected_value_diff"] = abs(exp_val - base_value)
    check["verified"] = bool(diff < 1e-6 and abs(exp_val - base_value) < 1e-6)
    if not check["verified"]:
        check["note"] = "Values differed from the closed-form calculation."
except Exception as exc:  # the app does not depend on shap at runtime
    check["note"] = f"{type(exc).__name__}: {exc}"

report = {
    "model": saved.get("model"), "intercept": intercept, "base_value": base_value,
    "base_probability": float(1 / (1 + np.exp(-base_value))),
    "transformed_names": names, "original": original, "display": display,
    "mean": mean.tolist(), "global": global_rows,
    "global_sample_rows": int(len(sample)), "shap_check": check,
}
(REPORTS / "explainability.json").write_text(json.dumps(report), encoding="utf-8")

print("\nWritten: reports/explainability.json")
print(f"Transformed features: {len(names)} | original features: {len(set(original))}")
print(f"Base value (log-odds): {base_value:.4f} | base probability: {report['base_probability']:.4f}")
print("shap check:", check)
print("Top 10 features by mean |SHAP|:")
for r in global_rows[:10]:
    print(f"  {r['feature']:<28} {r['mean_abs']:.4f}")