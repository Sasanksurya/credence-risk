"""Page 4: Statistical Analysis."""
import pandas as pd
import streamlit as st

from app.components import stat_charts
from app.components.metrics import banner, card, hero, metric_row, section
from app.services.cache import load_report
from app.theme import AMBER, BLUE, CRIMSON, TEAL


def fmt_p(p) -> str:
    if p is None:
        return "n/a"
    if p < 1e-300:
        return "< 1e-300"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def num(v) -> str:
    if v is None:
        return "n/a"
    return f"{v:,.3f}" if abs(v) < 10 else f"{v:,.0f}"


def effect_label(v: float) -> str:
    v = abs(v)
    return "Negligible" if v < 0.1 else "Small" if v < 0.3 else "Moderate" if v < 0.5 else "Large"


def v_label(v: float) -> str:
    return "Negligible" if v < 0.05 else "Weak" if v < 0.15 else "Moderate" if v < 0.30 else "Strong"


def render(art) -> None:
    stat = load_report("statistical_summary.json")
    val = load_report("validation_summary.json")
    hero("Statistical analysis", "Statistical Analysis",
         "Distributional and association tests on the Home Credit application data. These describe "
         "how applicant characteristics relate to the payment-difficulty outcome (TARGET = 1).")
    if not stat:
        banner("Statistical summary not found. Run scripts\\build_validation_artifacts.py first.", warn=True)
        return

    counts = stat["class_counts"]
    c0, c1 = counts["0"], counts["1"]
    rate = stat["default_rate"]
    metric_row([
        dict(label="Applicants analysed", value=f"{stat['rows']:,}", sub="application_train.csv", accent=BLUE),
        dict(label="TARGET = 1 rate", value=f"{rate * 100:.2f}%", sub="Share with payment difficulty", accent=CRIMSON),
        dict(label="Class ratio", value=f"{c0 / c1:.1f} : 1", sub="TARGET = 0 to TARGET = 1", accent=TEAL),
        dict(label="Placeholder values", value=f"{stat['days_employed_anomaly_rows']:,}",
             sub="DAYS_EMPLOYED = 365243, treated as missing here", accent=AMBER),
    ])
    banner("Statistics on this page describe associations in historical data. They do not show that any "
           "feature causes payment difficulty.")

    # 1. Class imbalance
    section("1. Class imbalance", "Why accuracy is the wrong yardstick for this problem")
    left, right = st.columns([1, 1.2])
    with left:
        st.plotly_chart(stat_charts.class_donut(counts), width="stretch")
    with right:
        baseline = val["prevalence"] if val else rate
        st.markdown(card("What the imbalance means", [
            f"Only {rate * 100:.1f}% of applicants are TARGET = 1. A model that predicts TARGET = 0 for everyone "
            f"would be {(1 - rate) * 100:.1f}% accurate and useless, so accuracy is not reported.",
            f"Average Precision is judged against a no-skill baseline equal to prevalence "
            f"({baseline * 100:.1f}%). The saved model reaches {art.metrics['average_precision']:.3f}.",
            "The model was trained with class weighting to counter the imbalance. This lifts recall but "
            "inflates predicted probabilities, which is why scores are not calibrated default rates.",
            "The train and validation splits were stratified, so both keep the same class ratio.",
        ]), unsafe_allow_html=True)

    # 2. Numeric features
    numeric = [r for r in stat["numeric"] if r.get("rank_biserial") is not None]
    numeric.sort(key=lambda r: -abs(r["rank_biserial"]))
    section("2. Numeric features: TARGET = 0 versus TARGET = 1",
            "Mann-Whitney U test with rank-biserial effect size (robust to skewed financial data)")
    st.plotly_chart(stat_charts.effect_size_bar(numeric), width="stretch")
    table = pd.DataFrame([{
        "Feature": r["feature"],
        "Median (TARGET = 0)": num(r["median_target0"]),
        "Median (TARGET = 1)": num(r["median_target1"]),
        "Effect size": round(r["rank_biserial"], 3),
        "Strength": effect_label(r["rank_biserial"]),
        "In TARGET = 1": "Higher" if r["rank_biserial"] > 0 else "Lower",
        "p-value": fmt_p(r["mann_whitney_p"]),
        "Missing %": round(r["missing_pct"], 1),
    } for r in numeric])
    st.dataframe(table, width="stretch", hide_index=True)
    top3 = ", ".join(r["feature"] for r in numeric[:3])
    st.caption(f"Strongest separators: {top3}. DAYS_EMPLOYED is stored as negative days before the "
               "application, so a higher value means a shorter employment history. With this many "
               "applicants almost every p-value is tiny, so judge a feature by its effect size, "
               "not its p-value.")

    # 3. Correlation
    section("3. Correlation structure", "Spearman rank correlation, which captures monotonic relationships")
    st.plotly_chart(stat_charts.correlation_heatmap(stat["correlation"]), width="stretch")
    st.caption("Strongly correlated inputs (for example credit, annuity, and goods price) carry overlapping "
               "information. Logistic Regression coefficients on correlated inputs should not be read one at a time.")

    # 4. Categorical
    section("4. Categorical features", "Chi-square test of independence with Cramér's V as the effect size")
    cats = sorted(stat["categorical"], key=lambda c: -c["cramers_v"])
    by_name = {c["feature"]: c for c in cats}
    left, right = st.columns([1, 1.5])
    with left:
        st.dataframe(pd.DataFrame([{
            "Feature": c["feature"], "Cramér's V": round(c["cramers_v"], 3),
            "Strength": v_label(c["cramers_v"]), "p-value": fmt_p(c["chi2_p"]),
        } for c in cats]), width="stretch", hide_index=True)
        feature = st.selectbox("Explore a feature", list(by_name),
                               format_func=lambda f: f"{f} (V = {by_name[f]['cramers_v']:.3f})")
    with right:
        chosen = by_name[feature]
        shown = [r for r in chosen["rates"] if r["applicants"] >= 100 and r["default_rate"] is not None]
        hidden = len(chosen["rates"]) - len(shown)
        st.plotly_chart(stat_charts.category_rate_bar(shown, rate), width="stretch")
        if hidden:
            st.caption(f"{hidden} categor{'y' if hidden == 1 else 'ies'} with fewer than 100 applicants hidden "
                       "because their rates are unreliable.")
    st.caption("Cramér's V ranges from 0 to 1. Rough guide: below 0.05 negligible, up to 0.15 weak, "
               "up to 0.30 moderate, above that strong.")

    # 5. Missing values
    section("5. Missing-value profile", "Handled by the saved median and most-frequent imputers")
    miss = stat["missing_top20"]
    st.plotly_chart(stat_charts.missing_bar(miss["features"], miss["pct"]), width="stretch")

    # 6. Methods
    section("6. Methods and interpretation")
    a, b = st.columns(2)
    a.markdown(card("Methods", [
        "Mann-Whitney U compares the two classes without assuming normality; the effect size is the "
        "rank-biserial correlation (0 means no separation, 1 means complete separation).",
        "Chi-square tests whether a category and TARGET are independent; Cramér's V measures strength.",
        "Spearman correlation uses ranks, so extreme incomes and credit amounts do not dominate.",
        "DAYS_EMPLOYED = 365243 is a placeholder, treated as missing in this analysis copy only. "
        "The model's input data is unchanged.",
    ]), unsafe_allow_html=True)
    b.markdown(card("How to interpret", [
        "Significant does not mean important. At this sample size, effect size is the useful number.",
        "Differences between groups can reflect other variables, so none of this is causal.",
        "These results describe one historical dataset and do not cover fairness or policy suitability.",
        "Category rates for small groups are noisy and are hidden below 100 applicants.",
    ]), unsafe_allow_html=True)