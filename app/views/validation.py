"""Page 5: Model Validation."""
import pandas as pd
import streamlit as st

from app.components import val_charts
from app.components.metrics import banner, card, hero, metric_row, section
from app.services.cache import load_report
from app.theme import AMBER, BLUE, CRIMSON, TEAL


def pct(v, digits: int = 1) -> str:
    return "n/a" if v is None else f"{v * 100:.{digits}f}%"


def render(art) -> None:
    val = load_report("validation_summary.json")
    hero("Model validation", "Model Validation",
         "How well the saved model separates applicants, what its thresholds trade off, and where its "
         "probabilities should not be taken at face value. All results are from one validation split.")
    if not val:
        banner("Validation report not found. Run scripts\\build_validation_artifacts.py first.", warn=True)
        return

    m = art.metrics
    at = val["at_threshold_0_5"]
    prevalence = val["prevalence"]
    n = val["validation_rows"]

    metric_row([
        dict(label="ROC-AUC", value=f"{m['roc_auc']:.4f}", sub="Ranking ability", accent=BLUE),
        dict(label="Average Precision", value=f"{m['average_precision']:.4f}",
             sub=f"No-skill baseline {prevalence:.3f}", accent=TEAL),
        dict(label="Precision (0.50)", value=f"{at['precision']:.4f}", sub="Flagged who are TARGET = 1", accent=AMBER),
        dict(label="Recall (0.50)", value=f"{at['recall']:.4f}", sub="TARGET = 1 cases caught", accent=CRIMSON),
        dict(label="F1 (0.50)", value=f"{at['f1']:.4f}", sub="Harmonic mean", accent="#A78BFA"),
    ])
    banner(f"ROC-AUC and Average Precision are the saved experiment values, reproduced from the saved model. "
           f"Precision, recall, curves, and calibration were computed from the saved model on the same "
           f"validation split ({n:,} applicants, {prevalence * 100:.2f}% TARGET = 1). These are preliminary "
           "validation results, not production approval.")

    section("Discrimination", "Can the model rank applicants who struggle to pay above those who do not?")
    c1, c2 = st.columns(2)
    c1.plotly_chart(val_charts.roc_chart(val["roc_curve"], m["roc_auc"]), width="stretch")
    c2.plotly_chart(val_charts.pr_chart(val["pr_curve"], m["average_precision"], prevalence), width="stretch")
    c3, c4 = st.columns(2)
    c3.markdown(card("ROC-AUC", [
        f"An AUC of {m['roc_auc']:.3f} means a randomly chosen TARGET = 1 applicant receives a higher "
        "score than a randomly chosen TARGET = 0 applicant about that share of the time (0.5 is chance).",
        "AUC measures ranking only; it says nothing about whether the probabilities are accurate.",
    ]), unsafe_allow_html=True)
    c4.markdown(card("Average Precision", [
        f"With only {prevalence * 100:.1f}% positives, a no-skill model scores about {prevalence:.3f}. "
        f"The saved model scores {m['average_precision']:.3f}, roughly {m['average_precision'] / prevalence:.1f} times better.",
        "Average Precision is more informative than AUC when the positive class is rare.",
    ]), unsafe_allow_html=True)

    rows = val["thresholds"]
    section("Threshold trade-offs", "Precision and recall move in opposite directions as the cut-off changes")
    left, right = st.columns([1.3, 1])
    left.plotly_chart(val_charts.threshold_chart(rows), width="stretch")
    right.dataframe(pd.DataFrame([{
        "Threshold": f"{r['threshold']:.2f}",
        "Precision": "n/a" if r["flagged_share"] == 0 else f"{r['precision']:.3f}",
        "Recall": f"{r['recall']:.3f}",
        "F1": f"{r['f1']:.3f}",
        "Flagged": pct(r["flagged_share"]),
    } for r in rows]), width="stretch", hide_index=True, height=360)

    best = max(rows, key=lambda r: r["f1"])
    half = next((r for r in rows if abs(r["threshold"] - 0.5) < 1e-9), None)
    bullets = [
        "Raising the threshold flags fewer applicants, so precision tends to rise and recall falls. "
        "Lowering it does the opposite.",
        f"Among the thresholds shown, F1 is highest at {best['threshold']:.2f} "
        f"(F1 {best['f1']:.3f}, recall {best['recall']:.3f}).",
    ]
    if half:
        bullets.append(f"At 0.50 the model flags {pct(half['flagged_share'])} of applicants and catches "
                       f"{pct(half['recall'])} of TARGET = 1 cases, but only {pct(half['precision'])} of "
                       "those flagged are TARGET = 1.")
    bullets.append("The right threshold depends on the relative cost of missing a case versus flagging a "
                   "good applicant. No threshold is recommended here.")
    st.markdown(card("Reading the trade-off", bullets), unsafe_allow_html=True)

    section("Calibration", "Does a predicted 30% correspond to roughly 30% observed?")
    mean_pred = val["mean_predicted_probability"]
    left, right = st.columns(2)
    left.plotly_chart(val_charts.calibration_chart(val["calibration"]), width="stretch")
    cal_bullets = [
        "Each point is one tenth of validation applicants, grouped by predicted probability. "
        "A well-calibrated model lies on the dashed diagonal.",
        f"The average predicted probability is {pct(mean_pred)} while the observed rate is {pct(prevalence)}.",
    ]
    if mean_pred > prevalence:
        cal_bullets.append("Points below the diagonal mean the model overstates risk. This is expected after "
                           "training with balanced class weights.")
    cal_bullets.append("Recalibration methods exist but have not been applied, so scores should not be read "
                       "as default rates.")
    right.markdown(card("Reading the calibration curve", cal_bullets), unsafe_allow_html=True)

    section("Discrimination versus calibration")
    d1, d2 = st.columns(2)
    d1.markdown(card("Discrimination (ranking)", [
        "Answers: does the model put riskier applicants above safer ones?",
        "Measured by ROC-AUC and Average Precision, which depend only on the order of scores.",
        "Rescaling every probability by the same monotonic rule leaves these metrics unchanged.",
    ]), unsafe_allow_html=True)
    d2.markdown(card("Calibration (probability accuracy)", [
        "Answers: does a predicted probability match the observed frequency?",
        "Matters when scores are used as probabilities, for example in expected-loss calculations.",
        "A model can rank well and still be poorly calibrated, which is the case here.",
    ]), unsafe_allow_html=True)

    section("Illustrative risk bands against observed outcomes")
    band_table = []
    for r in val["risk_bands"]:
        rate = r["actual_rate"]
        band_table.append({
            "Band": r["band"], "Applicants": f"{r['applicants']:,}",
            "Share of validation": pct(r["applicants"] / n),
            "Observed TARGET = 1 rate": pct(rate),
            "Lift versus overall": "n/a" if rate is None else f"{rate / prevalence:.2f}x",
        })
    st.dataframe(pd.DataFrame(band_table), width="stretch", hide_index=True)
    high = next((r for r in val["risk_bands"] if r["band"] == "High"), None)
    note = ("Observed rates rise from band to band, which supports using the model for ranking. ")
    if high and high["actual_rate"] is not None:
        note += (f"The High band is defined by a predicted probability of 0.50 or more, yet its observed rate "
                 f"is {pct(high['actual_rate'])}, which is the calibration gap again. ")
    note += "Band names are demonstration labels, not lending grades."
    st.caption(note)

    section("Limitations of this validation")
    st.markdown(card("What this evidence does not show", [
        "Results come from a single 80/20 stratified split with no confidence intervals or cross-validated spread.",
        "There is no out-of-time validation, so stability across periods is untested.",
        "Only application-level features are used; the historical Home Credit tables are not included.",
        "No fairness, stability, or regulatory assessment has been performed.",
        "Thresholds were not tuned to any cost or profit assumptions.",
    ]), unsafe_allow_html=True)