"""Page 3: Batch applicant scoring."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.metrics import banner, card, hero, metric_row, section
from app.config import MAX_UPLOAD_MB, MAX_UPLOAD_ROWS, RISK_BAND_LABELS
from app.services.risk_scoring import (BAND_COLUMN, ID_COLUMN, IGNORED_COLUMNS, SCORE_COLUMN,
                                       SchemaError, required_input_columns, score_dataframe)
from app.theme import AMBER, BAND_COLORS, BLUE, TEAL, apply_plotly_theme

STATE_KEY = "batch_state"


def read_upload(uploaded) -> pd.DataFrame:
    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"The file is larger than the {MAX_UPLOAD_MB} MB limit.")
    try:
        df = pd.read_csv(uploaded, nrows=MAX_UPLOAD_ROWS + 1, low_memory=False)
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty.")
    except UnicodeDecodeError:
        raise ValueError("The file is not valid UTF-8 text. Save it as CSV (UTF-8) and try again.")
    except pd.errors.ParserError as exc:
        raise ValueError(f"The file could not be parsed as CSV: {exc}")
    if df.empty:
        raise ValueError("The file has a header but no data rows.")
    if len(df) > MAX_UPLOAD_ROWS:
        raise ValueError(f"The file has more than {MAX_UPLOAD_ROWS:,} rows. Split it into smaller files.")
    return df


def band_chart(counts: pd.Series):
    labels = list(RISK_BAND_LABELS)
    values = [int(counts.get(b, 0)) for b in labels]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=[BAND_COLORS[b] for b in labels],
                           text=[f"{v:,}" for v in values], textposition="outside"))
    fig.update_yaxes(title="Applicants", range=[0, max(max(values), 1) * 1.25])
    return apply_plotly_theme(fig, height=300, title="Scored applicants by illustrative risk band")


def run_scoring(df: pd.DataFrame, art, signature: str) -> dict:
    required = required_input_columns(art.feature_columns)
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"signature": signature, "error_missing": missing}

    scored = score_dataframe(df, art)
    base = df.drop(columns=[SCORE_COLUMN, BAND_COLUMN], errors="ignore").reset_index(drop=True)
    res = scored.results.drop(columns=[ID_COLUMN], errors="ignore").reset_index(drop=True)
    output = pd.concat([base, res], axis=1)

    preview_cols = ([ID_COLUMN] if ID_COLUMN in df.columns else []) + [SCORE_COLUMN, BAND_COLUMN]
    ignored = [c for c in IGNORED_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in art.feature_columns and c not in IGNORED_COLUMNS
             and c not in (SCORE_COLUMN, BAND_COLUMN)]
    return {
        "signature": signature, "rows": len(output), "warnings": scored.warnings,
        "ignored": ignored, "extra": extra,
        "preview": output[preview_cols].head(100),
        "band_counts": output[BAND_COLUMN].value_counts(),
        "mean_probability": float(output[SCORE_COLUMN].mean()),
        "csv": output.to_csv(index=False).encode("utf-8"),
    }


def render(art) -> None:
    required = required_input_columns(art.feature_columns)
    hero("Portfolio scoring", "Batch Applicant Scoring",
         "Upload a CSV of applicants to estimate the probability of the payment-difficulty outcome "
         "(TARGET = 1) for every row. Rows are scored in order and your original columns are preserved.")
    banner("Uploaded data is processed in memory for this session only and is not stored by the application. "
           f"Limits: {MAX_UPLOAD_MB} MB and {MAX_UPLOAD_ROWS:,} rows per file.")

    c1, c2 = st.columns([2.2, 1])
    with c2:
        template = pd.DataFrame(columns=[ID_COLUMN] + required).to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV template", template, "applicant_template.csv", "text/csv")
        st.caption(f"The template lists the {len(required)} required input columns plus an optional "
                   f"{ID_COLUMN}. The three financial ratios are calculated by the app.")
    with c1:
        uploaded = st.file_uploader("Upload applicant CSV", type=["csv"])

    if uploaded is None:
        st.session_state.pop(STATE_KEY, None)
        st.markdown(card("Preparing your file", [
            f"Include all {len(required)} required columns; extra columns are kept in the download but not used.",
            f"{ID_COLUMN} is optional and, with TARGET, is never used as a model input.",
            "Empty cells are allowed; the saved model fills them with training medians or most frequent values.",
            "Non-numeric values in numeric columns and unseen categories are reported as warnings.",
        ]), unsafe_allow_html=True)
        return

    signature = f"{uploaded.name}-{uploaded.size}"
    if st.button("Score all applicants", type="primary"):
        try:
            df = read_upload(uploaded)
            with st.spinner("Scoring applicants..."):
                st.session_state[STATE_KEY] = run_scoring(df, art, signature)
        except (ValueError, SchemaError) as exc:
            st.session_state.pop(STATE_KEY, None)
            st.error(str(exc))
            return

    state = st.session_state.get(STATE_KEY)
    if not state or state.get("signature") != signature:
        st.info("File ready. Select Score all applicants to run the model.")
        return

    if "error_missing" in state:
        miss = state["error_missing"]
        st.error(f"{len(miss)} required column(s) are missing, so the file cannot be scored.")
        with st.expander("Show missing columns", expanded=True):
            st.code(", ".join(miss))
        return

    section("Scoring results", f"{state['rows']:,} applicants scored")
    counts = state["band_counts"]
    n = state["rows"]
    high_share = (counts.get("High", 0) + counts.get("Elevated", 0)) / n * 100
    metric_row([
        dict(label="Applicants scored", value=f"{n:,}", sub="Order preserved", accent=BLUE),
        dict(label="Mean probability", value=f"{state['mean_probability'] * 100:.1f}%",
             sub="Average model estimate", accent=TEAL),
        dict(label="Elevated or High", value=f"{high_share:.1f}%", sub="Illustrative bands", accent=AMBER),
    ])

    for w in state["warnings"]:
        st.warning(w)
    if state["ignored"]:
        st.caption(f"Ignored as model inputs: {', '.join(state['ignored'])}.")
    if state["extra"]:
        st.caption(f"{len(state['extra'])} extra column(s) are kept in the download but were not used.")

    left, right = st.columns([1, 1.3])
    with left:
        st.plotly_chart(band_chart(counts), width="stretch")
    with right:
        st.markdown("**Preview (first 100 rows)**")
        st.dataframe(state["preview"], width="stretch", hide_index=True, height=300)

    st.download_button("Download scored CSV", state["csv"], "scored_applicants.csv", "text/csv")
    st.caption("Probabilities are model estimates from class-weighted training, not calibrated default rates. "
               "Bands are illustrative and do not represent lending decisions.")