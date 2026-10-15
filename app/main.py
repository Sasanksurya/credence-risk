"""Credence Risk - Streamlit entry point. Run: python -m streamlit run app/main.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import config  # noqa: E402
from app.services.cache import get_artifacts  # noqa: E402
from app.services.model_loader import ArtifactError  # noqa: E402
from app.theme import inject_css  # noqa: E402
from app.views import analysis, batch, explainability, overview, scoring, validation  # noqa: E402

st.set_page_config(page_title=config.APP_TITLE, page_icon=":material/shield:", layout="wide")
inject_css()

PAGES = {
    "Overview": overview.render,
    "Applicant Scoring": scoring.render,
    "Batch Scoring": batch.render,
    "Statistical Analysis": analysis.render,
    "Model Validation": validation.render,
    "Explainability": explainability.render,
}

with st.sidebar:
    st.markdown('<div class="cr-brand">Credence <span>Risk</span></div>'
                '<div class="cr-brand-sub">Credit Risk Intelligence &amp; Validation Platform</div>',
                unsafe_allow_html=True)
    choice = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.markdown('<div class="cr-foot">Predictions are estimates, not lending decisions.</div>',
                unsafe_allow_html=True)

try:
    artifacts = get_artifacts()
except ArtifactError as exc:
    st.error(f"Model artifacts could not be loaded. {exc}")
    st.stop()

if artifacts.version_warning:
    st.warning(artifacts.version_warning)

PAGES[choice](artifacts)