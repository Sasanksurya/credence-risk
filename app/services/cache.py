"""Streamlit-cached access to artifacts and saved reports."""
import json

import streamlit as st

from app.config import REPORTS_DIR
from app.services.model_loader import load_artifacts


@st.cache_resource(show_spinner="Loading saved model...")
def get_artifacts():
    return load_artifacts()


@st.cache_data(show_spinner=False)
def load_report(name: str):
    path = REPORTS_DIR / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))