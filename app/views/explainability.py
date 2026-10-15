"""Page 6: Explainability."""
import pandas as pd
import streamlit as st

from app.components import explain_charts
from app.components.metrics import banner, card, hero, metric_row, section
from app.config import ROOT
from app.services.cache import load_report
from app.services.explain import ExplainError, build_context, explain_row
from app.services.risk_scoring import SchemaError, assign_risk_bands, required_input_columns
from app.theme import AMBER, BLUE, CRIMSON, TEAL
from app.views.batch import read_upload

SAMPLE_FILE = ROOT / "samples" / "sample_applicants.csv"
TOP_N = 10
EMPLOYED_PLACEHOLDER = 365243


def fmt_value(v) -> str:
    if pd.isna(v):
        return "missing (imputed)"
    if isinstance(v, (int, float)):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3g}"
    return str(v)


def sigmoid(z: float) -> float:
    return 1 / (1 + 2.718281828459045 ** (-z))


def global_tab(report: dict) -> None:
    rows = report["global"]
    check = report["shap_check"]
    n = report["global_sample_rows"]
    section("Global feature importance",
            f"Mean absolute SHAP value over {n:,} randomly sampled validation applicants")
    banner("DAYS_EMPLOYED = 365243 is a placeholder the Home Credit data uses for applicants who are not "
           "currently employed (about 18% of applicants). The saved model was trained with it as a real "
           "number, which is why DAYS_EMPLOYED ranks first. Its effect is largely offset by features that "
           "move with it (income type, organisation type, FLAG_EMP_PHONE), so read these bars as the "
           "behaviour of this model, not as separate real-world effects.", warn=True)
    left, right = st.columns([1.5, 1])
    left.plotly_chart(explain_charts.global_importance(rows), width="stretch")
    with right:
        st.markdown(card("How to read this chart", [
            "Each bar is the average size of a feature's push on the score, in log-odds, regardless of "
            "direction. Longer bars mean the feature moves scores more.",
            "One-hot encoded categories are summed back into their original feature, so a categorical "
            "feature appears once.",
            "Importance describes this model, not the real world. Correlated inputs share credit, so "
            "treat close rankings as ties.",
        ]), unsafe_allow_html=True)
        if check.get("verified"):
            st.markdown(card("Method check", [
                f"The closed-form contributions match shap {check['shap_version']} LinearExplainer "
                f"(largest difference {check['max_abs_diff']:.1e}).",
            ]), unsafe_allow_html=True)
        else:
            st.markdown(card("Method check", [
                "The shap library cross-check did not complete" +
                (f": {check['note']}" if check.get("note") else "."),
                "Contributions are the exact closed form for a linear model but were not cross-checked.",
            ]), unsafe_allow_html=True)

    table = []
    for r in rows[:TOP_N * 2]:
        if r["kind"] == "numeric" and r.get("coef") is not None:
            direction = "Higher value raises score" if r["coef"] > 0 else "Higher value lowers score"
        else:
            direction = "Depends on category"
        table.append({"Feature": r["feature"], "Mean |SHAP|": round(r["mean_abs"], 4),
                      "Type": r["kind"].capitalize(), "Direction": direction})
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
    st.caption("Direction refers to the model's coefficient for a standardised numeric feature. "
               "DAYS_ columns are stored as negative day counts, so a higher value means a shorter period.")


def individual_tab(art, ctx) -> None:
    section("Individual explanation", "Why one applicant received their score")
    uploaded = st.file_uploader("Upload an applicant CSV (same format as Batch Scoring)", type=["csv"],
                                key="explain_upload")
    df = None
    if uploaded is not None:
        try:
            df = read_upload(uploaded)
        except ValueError as exc:
            st.error(str(exc))
            return
    elif SAMPLE_FILE.is_file():
        df = pd.read_csv(SAMPLE_FILE)
        st.caption("No file uploaded, so the bundled sample file samples\\sample_applicants.csv is used.")
    if df is None:
        st.info("Upload an applicant CSV to see an explanation.")
        return

    missing = [c for c in required_input_columns(art.feature_columns) if c not in df.columns]
    if missing:
        st.error(f"{len(missing)} required column(s) are missing, for example: {', '.join(missing[:8])}")
        return

    row_no = int(st.number_input("Applicant row number", 1, len(df), 1, 1))
    if "SK_ID_CURR" in df.columns:
        st.caption(f"SK_ID_CURR: {df['SK_ID_CURR'].iloc[row_no - 1]}")

    try:
        result = explain_row(art, ctx, df.iloc[[row_no - 1]])
    except (SchemaError, ExplainError) as exc:
        st.error(str(exc))
        return

    prob = result["probability"]
    band = str(assign_risk_bands([prob])[0])
    metric_row([
        dict(label="Predicted probability", value=f"{prob * 100:.1f}%", sub="TARGET = 1, uncalibrated", accent=CRIMSON),
        dict(label="Illustrative band", value=band, sub="Demonstration cut-offs", accent=AMBER),
        dict(label="Base probability", value=f"{sigmoid(result['base_value']) * 100:.1f}%",
             sub="Output for an average applicant", accent=TEAL),
        dict(label="Model output", value=f"{result['logit']:+.2f}", sub="Log-odds (base "
             f"{result['base_value']:+.2f})", accent=BLUE),
    ])
    for w in result["warnings"]:
        st.warning(w)
    if result["values"].get("DAYS_EMPLOYED") == EMPLOYED_PLACEHOLDER:
        st.info("This applicant has the DAYS_EMPLOYED placeholder value (365243). The large DAYS_EMPLOYED "
                "bar is offset by income type, organisation type and FLAG_EMP_PHONE, which move together. "
                "Interpret those bars as a group.")

    contribs = result["contributions"]
    ranked = sorted(contribs.items(), key=lambda kv: -abs(kv[1]))
    top = ranked[:TOP_N]
    rest = ranked[TOP_N:]
    items = [(f"{f} = {fmt_value(result['values'].get(f))}", v) for f, v in top]
    st.plotly_chart(explain_charts.waterfall(result["base_value"], items, sum(v for _, v in rest),
                                             len(rest), result["logit"]), width="stretch")

    st.dataframe(pd.DataFrame([{
        "Feature": f, "Applicant value": fmt_value(result["values"].get(f)),
        "Contribution (log-odds)": round(v, 4),
        "Effect": "Raises score" if v > 0 else "Lowers score",
    } for f, v in top]), width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    c1.markdown(card("Reading the waterfall", [
        "Red bars push the score up (toward TARGET = 1) and teal bars push it down, relative to an "
        "applicant whose features all equal the training average.",
        "Bars are in log-odds. Start from the base value, add every bar, and you reach the model output, "
        "which converts to the predicted probability.",
        "Missing inputs were filled by the saved imputers, so their bars reflect the filled value.",
    ]), unsafe_allow_html=True)
    c2.markdown(card("Limitations", [
        "Contributions describe associations learned by the model, not causal effects. Changing a "
        "feature in real life would not necessarily change the outcome.",
        "Credit, annuity and goods price move together, so their individual bars are not independent.",
        "The probability is not a calibrated default rate, and the band is illustrative.",
    ]), unsafe_allow_html=True)


def render(art) -> None:
    hero("Explainability", "Explainability",
         "Which features drive the model's scores, globally and for one applicant, using SHAP values "
         "from the saved Logistic Regression pipeline.")
    report = load_report("explainability.json")
    if not report:
        banner("Explainability report not found. Run scripts\\build_explainability_artifacts.py first.",
               warn=True)
        return
    try:
        ctx = build_context(art, report)
    except ExplainError as exc:
        st.error(str(exc))
        return

    banner("SHAP values explain how the model reached a score. They show associations, not causes, and "
           "they do not make the model fair, calibrated or suitable for lending decisions.")
    tab_global, tab_individual = st.tabs(["Global feature importance", "Individual explanation"])
    with tab_global:
        global_tab(report)
    with tab_individual:
        individual_tab(art, ctx)