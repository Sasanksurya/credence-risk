"""Page 1: Executive Overview."""
import streamlit as st

from app.components import charts
from app.components.metrics import banner, card, hero, kv_table, metric_row, section
from app.config import EXPECTED_SKLEARN_VERSION, RISK_BAND_THRESHOLDS
from app.services.cache import load_report
from app.theme import AMBER, BLUE, TEAL


def render(art) -> None:
    val = load_report("validation_summary.json")
    m = art.metrics
    clf = art.model.named_steps["classifier"]
    params = clf.get_params()

    hero("Credit Risk Intelligence", "Credence Risk",
         "A validation and scoring workspace for the Home Credit Default Risk dataset. "
         "It estimates the probability of the payment-difficulty outcome (TARGET = 1) from "
         "application-level data using a saved, tuned Logistic Regression pipeline.")

    prevalence = f"{val['prevalence'] * 100:.2f}%" if val else "n/a"
    metric_row([
        dict(label="ROC-AUC", value=f"{m['roc_auc']:.4f}", sub="Ranking ability, validation split", accent=BLUE),
        dict(label="Average Precision", value=f"{m['average_precision']:.4f}",
             sub="Precision-recall summary", accent=TEAL),
        dict(label="Validation applicants", value=f"{m['validation_rows']:,}", sub="Stratified 20% hold-out", accent=AMBER),
        dict(label="Positive-class prevalence", value=prevalence, sub="Share of TARGET = 1 in validation", accent="#A78BFA"),
    ])

    banner("This application is a demonstration for educational and analytical purposes. "
           "Outputs are model estimates, not lending decisions, and do not guarantee future performance.")

    left, right = st.columns([1, 1.15])
    with left:
        section("Saved experiment details", "Read from the saved model and metrics files")
        st.markdown(kv_table([
            ("Model", m.get("model", "n/a")),
            ("Classifier", type(clf).__name__),
            ("Regularisation C", params.get("C")),
            ("Class weight", params.get("class_weight")),
            ("Penalty", params.get("penalty")),
            ("Input features", len(art.feature_columns)),
            ("Random state", m.get("random_state")),
            ("Saved with scikit-learn", EXPECTED_SKLEARN_VERSION),
        ]), unsafe_allow_html=True)
    with right:
        section("Do the risk bands rank applicants?", "Observed outcomes in the validation split")
        if val:
            st.plotly_chart(charts.default_rate_by_band(val["risk_bands"]), width="stretch")
        else:
            banner("Validation report not found. Run scripts\\build_validation_artifacts.py.", warn=True)

    section("Validated results versus illustrative bands")
    t1, t2, t3 = RISK_BAND_THRESHOLDS
    c1, c2 = st.columns(2)
    validated = ["ROC-AUC and Average Precision come from the saved experiment and were reproduced "
                 "from the saved model on the same split."]
    if val:
        a = val["at_threshold_0_5"]
        validated.append(f"At a 0.50 threshold: precision {a['precision']:.4f}, recall {a['recall']:.4f}, "
                         f"F1 {a['f1']:.4f}.")
    validated.append("These describe performance on one validation split only.")
    c1.markdown(card("Validated", validated), unsafe_allow_html=True)
    c2.markdown(card("Illustrative (not validated)", [
        f"Bands: Low below {t1:.2f}, Moderate {t1:.2f} to {t2:.2f}, Elevated {t2:.2f} to {t3:.2f}, "
        f"High {t3:.2f} or above.",
        "These cut-offs are demonstration values, not calibrated lending grades, loss thresholds, or policy.",
        "Class weighting inflates predicted probabilities, so a score is not a default rate.",
    ]), unsafe_allow_html=True)

    section("Intended use and limitations")
    c3, c4 = st.columns(2)
    c3.markdown(card("Intended use", [
        "Portfolio demonstration of a credit-risk modelling and validation workflow.",
        "Exploring model behaviour, ranking quality, and explanations.",
        "Educational analysis on public, anonymised data.",
    ]), unsafe_allow_html=True)
    c4.markdown(card("Limitations", [
        "Trained on application-level features only; historical tables are not used.",
        "Probabilities are not calibrated; use scores for ranking, not as default rates.",
        "No fairness, regulatory, or production-readiness assessment has been performed.",
        "The application never approves, rejects, or recommends lending decisions.",
    ]), unsafe_allow_html=True)