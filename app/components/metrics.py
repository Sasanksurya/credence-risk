"""Reusable UI building blocks."""
import streamlit as st

from app.theme import TEAL


def metric_card(label: str, value: str, sub: str = "", accent: str = TEAL) -> str:
    return (f'<div class="cr-metric" style="--accent:{accent}"><div class="cr-label">{label}</div>'
            f'<div class="cr-value">{value}</div><div class="cr-sub">{sub}</div></div>')


def metric_row(items: list) -> None:
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.markdown(metric_card(**item), unsafe_allow_html=True)


def hero(eyebrow: str, title: str, lead: str) -> None:
    st.markdown(f'<div class="cr-eyebrow">{eyebrow}</div><div class="cr-title">{title}</div>'
                f'<div class="cr-lead">{lead}</div>', unsafe_allow_html=True)


def banner(text: str, warn: bool = False) -> None:
    st.markdown(f'<div class="cr-banner{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    sub = f"<small>{subtitle}</small>" if subtitle else ""
    st.markdown(f'<div class="cr-section">{title}{sub}</div>', unsafe_allow_html=True)


def card(title: str, bullets: list) -> str:
    items = "".join(f"<li>{b}</li>" for b in bullets)
    return f'<div class="cr-card"><h4>{title}</h4><ul>{items}</ul></div>'


def kv_table(rows: list) -> str:
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f'<div class="cr-card"><table class="cr-table">{body}</table></div>'