"""Page 2: Individual applicant risk scoring."""
import numpy as np
import pandas as pd
import streamlit as st

from app.components import charts
from app.components.metrics import banner, card, hero, metric_card, section
from app.config import RISK_BAND_THRESHOLDS
from app.services.cache import load_report
from app.services.risk_scoring import (BAND_COLUMN, SCORE_COLUMN, SchemaError, engineer_features,
                                       required_input_columns, score_dataframe)
from app.theme import BAND_COLORS, BORDER, MUTED, PANEL, TEXT

NOT_PROVIDED = "Not provided"
LABELS = {"Y": "Yes", "N": "No", "M": "Male", "F": "Female", "XNA": "Not specified"}


def cat_select(art, label, col, optional=False, index=0):
    options = list(art.known_categories.get(col, []))
    if optional:
        options = [NOT_PROVIDED] + options
    if not options:
        st.caption(f"{label}: not available in the saved model.")
        return None
    choice = st.selectbox(label, options, index=min(index, len(options) - 1), key=f"cat_{col}",
                          format_func=lambda v: LABELS.get(v, v))
    return None if choice == NOT_PROVIDED else choice


def optional_number(label, key, min_value, max_value, default, step, na_default=True, fmt=None):
    c1, c2 = st.columns([3, 1])
    na = c2.checkbox("N/A", value=na_default, key=f"{key}_na")
    val = c1.number_input(label, min_value=min_value, max_value=max_value, value=default,
                          step=step, key=key, disabled=na, format=fmt)
    return None if na else val


def validate(income, credit, annuity, goods, age, years_employed):
    errors, notes = [], []
    if income <= 0:
        errors.append("Annual income must be greater than zero.")
    if credit <= 0:
        errors.append("Credit amount must be greater than zero.")
    if annuity <= 0:
        errors.append("Annuity amount must be greater than zero.")
    if goods is not None and goods <= 0:
        errors.append("Goods price must be greater than zero, or marked N/A.")
    if years_employed is not None and years_employed > age - 14:
        errors.append("Years employed cannot exceed age minus 14.")
    if income > 0 and credit / income > 20:
        notes.append("Credit amount is more than 20 times annual income, which is far outside typical "
                     "training data. Treat the score with extra caution.")
    return errors, notes


def build_record(art, values: dict) -> pd.DataFrame:
    required = required_input_columns(art.feature_columns)
    row = {c: np.nan for c in required}
    for k, v in values.items():
        if k in row:
            row[k] = np.nan if v is None else v
    return pd.DataFrame([row], columns=required)


def placeholder():
    st.markdown(f'<div class="cr-card"><h4>Prediction result</h4><p style="color:{MUTED};font-size:.9rem">'
                'Enter applicant details and select <b>Calculate risk score</b>. The result, illustrative '
                'band, and interpretation appear here.</p></div>', unsafe_allow_html=True)


def render(art) -> None:
    val = load_report("validation_summary.json")
    hero("Individual scoring", "Applicant Risk Scoring",
         "Estimate the model's probability of the payment-difficulty outcome (TARGET = 1) for one applicant. "
         "Example values are pre-filled for the required fields; replace them with applicant data.")
    banner("This form collects the most informative inputs. All other model inputs are treated as missing "
           "and filled by the saved model's median or most-frequent imputers, so the score is an approximation. "
           "For complete records, use Batch Scoring.")

    left, right = st.columns([1.6, 1])
    with left:
        with st.expander("Demographics and household", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                gender = cat_select(art, "Gender", "CODE_GENDER")
                family = cat_select(art, "Family status", "NAME_FAMILY_STATUS")
            with c2:
                age = st.number_input("Age (years)", 18, 80, 38, 1)
                children = st.number_input("Number of children", 0, 20, 0, 1)
            with c3:
                education = cat_select(art, "Education level", "NAME_EDUCATION_TYPE")
                housing = cat_select(art, "Housing type", "NAME_HOUSING_TYPE")
        with st.expander("Employment and income", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                income_type = cat_select(art, "Income type", "NAME_INCOME_TYPE")
                occupation = cat_select(art, "Occupation", "OCCUPATION_TYPE", optional=True)
            with c2:
                income = st.number_input("Annual income", 0.0, 1e9, 150000.0, 10000.0, format="%.0f")
                years_emp = optional_number("Years employed", "years_emp", 0.0, 60.0, 5.0, 0.5,
                                            na_default=False, fmt="%.1f")
        with st.expander("Loan details", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                contract = cat_select(art, "Contract type", "NAME_CONTRACT_TYPE")
                credit = st.number_input("Credit amount", 0.0, 1e9, 450000.0, 10000.0, format="%.0f")
            with c2:
                annuity = st.number_input("Annuity amount (per period)", 0.0, 1e8, 25000.0, 1000.0, format="%.0f")
                goods = optional_number("Goods price", "goods", 0.0, 1e9, 400000.0, 10000.0, fmt="%.0f")
        with st.expander("Assets", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                own_car = cat_select(art, "Owns a car", "FLAG_OWN_CAR")
            with c2:
                own_realty = cat_select(art, "Owns real estate", "FLAG_OWN_REALTY")
        with st.expander("External risk scores (0 to 1, optional)", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                ext1 = optional_number("EXT_SOURCE_1", "ext1", 0.0, 1.0, 0.5, 0.01, fmt="%.3f")
            with c2:
                ext2 = optional_number("EXT_SOURCE_2", "ext2", 0.0, 1.0, 0.5, 0.01, fmt="%.3f")
            with c3:
                ext3 = optional_number("EXT_SOURCE_3", "ext3", 0.0, 1.0, 0.5, 0.01, fmt="%.3f")
        go_clicked = st.button("Calculate risk score", type="primary")

    with right:
        if not go_clicked:
            placeholder()
            return

        errors, notes = validate(income, credit, annuity, goods, age, years_emp)
        if errors:
            for e in errors:
                st.error(e)
            return

        values = {
            "CODE_GENDER": gender, "NAME_FAMILY_STATUS": family, "NAME_EDUCATION_TYPE": education,
            "NAME_HOUSING_TYPE": housing, "NAME_INCOME_TYPE": income_type, "OCCUPATION_TYPE": occupation,
            "NAME_CONTRACT_TYPE": contract, "FLAG_OWN_CAR": own_car, "FLAG_OWN_REALTY": own_realty,
            "CNT_CHILDREN": int(children), "AMT_INCOME_TOTAL": float(income), "AMT_CREDIT": float(credit),
            "AMT_ANNUITY": float(annuity), "AMT_GOODS_PRICE": goods,
            "DAYS_BIRTH": -int(round(age * 365.25)),
            "DAYS_EMPLOYED": None if years_emp is None else -int(round(years_emp * 365.25)),
            "EXT_SOURCE_1": ext1, "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3,
        }
        record = build_record(art, values)
        try:
            result = score_dataframe(record, art)
        except SchemaError as exc:
            st.error(str(exc))
            return

        prob = float(result.results[SCORE_COLUMN].iloc[0])
        band = str(result.results[BAND_COLUMN].iloc[0])
        provided = int(record.notna().sum(axis=1).iloc[0])
        total = record.shape[1]

        st.markdown('<div class="cr-card"><h4>Prediction result</h4>', unsafe_allow_html=True)
        st.plotly_chart(charts.probability_gauge(prob, RISK_BAND_THRESHOLDS, band), width="stretch")
        color = BAND_COLORS[band]
        st.markdown(f'<div style="text-align:center;margin:-8px 0 10px 0"><span style="background:{color}22;'
                    f'color:{color};border:1px solid {color};padding:6px 18px;border-radius:999px;'
                    f'font-weight:700;letter-spacing:.06em">{band.upper()} (illustrative band)</span></div>',
                    unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        ratios = engineer_features(record.assign(**{k: record[k] for k in []}))[
            ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_GOODS_RATIO"]].iloc[0]
        fmt = lambda v: "n/a" if pd.isna(v) else f"{v:.2f}"
        st.markdown(card("Derived financial ratios", [
            f"Credit to income: {fmt(ratios['CREDIT_INCOME_RATIO'])}",
            f"Annual annuity to income: {fmt(ratios['ANNUITY_INCOME_RATIO'])}",
            f"Credit to goods price: {fmt(ratios['CREDIT_GOODS_RATIO'])}",
        ]), unsafe_allow_html=True)

    section("How to read this score")
    observed = None
    if val:
        observed = next((r["actual_rate"] for r in val["risk_bands"] if r["band"] == band), None)
    obs_text = (f"In the validation split, {band} band applicants had an observed TARGET = 1 rate of "
                f"{observed * 100:.1f}%." if observed is not None else "")
    c1, c2 = st.columns(2)
    c1.markdown(card("Meaning", [
        f"The model estimates a {prob * 100:.1f}% probability of the payment-difficulty outcome (TARGET = 1).",
        obs_text or "Validation rates by band are available after running the validation script.",
        f"{provided} of {total} model inputs were supplied; the rest were filled by the saved imputers.",
    ]), unsafe_allow_html=True)
    c2.markdown(card("Limitations", [
        "The model was trained with class weighting, so probabilities are inflated and not calibrated "
        "default rates.",
        "Bands are illustrative cut-offs, not lending grades or policy.",
        "The score does not approve, reject, or recommend any lending decision.",
        *notes,
    ]), unsafe_allow_html=True)