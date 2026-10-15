"""Plotly chart builders. Every chart reads saved, verified numbers only."""
import plotly.graph_objects as go

from app.theme import BAND_COLORS, MUTED, TEXT, apply_plotly_theme


def default_rate_by_band(rows: list):
    labels = [r["band"] for r in rows]
    rates = [(r["actual_rate"] or 0) * 100 for r in rows]
    counts = [r["applicants"] for r in rows]
    fig = go.Figure(go.Bar(
        x=labels, y=rates, marker_color=[BAND_COLORS[b] for b in labels],
        text=[f"{v:.1f}%" for v in rates], textposition="outside", customdata=counts,
        hovertemplate="%{x}<br>Observed rate: %{y:.2f}%<br>Applicants: %{customdata:,}<extra></extra>"))
    fig.update_yaxes(title="Observed TARGET = 1 rate (%)", range=[0, max(rates) * 1.3])
    return apply_plotly_theme(fig, height=330, title="Observed outcome rate by illustrative risk band")


def probability_gauge(prob: float, thresholds, band: str):
    t1, t2, t3 = thresholds
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        number={"suffix": "%", "valueformat": ".1f", "font": {"color": TEXT, "size": 44}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
            "bar": {"color": BAND_COLORS[band], "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0, t1 * 100], "color": "rgba(45,212,191,.18)"},
                {"range": [t1 * 100, t2 * 100], "color": "rgba(245,184,65,.18)"},
                {"range": [t2 * 100, t3 * 100], "color": "rgba(251,146,60,.18)"},
                {"range": [t3 * 100, 100], "color": "rgba(244,63,94,.18)"},
            ],
        }))
    return apply_plotly_theme(fig, height=250)