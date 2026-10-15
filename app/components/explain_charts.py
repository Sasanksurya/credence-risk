"""Plotly charts for the Explainability page."""
import plotly.graph_objects as go

from app.theme import BLUE, CRIMSON, MUTED, TEAL, apply_plotly_theme


def global_importance(rows: list, top: int = 20):
    rows = rows[:top][::-1]
    xs = [r["mean_abs"] for r in rows]
    fig = go.Figure(go.Bar(
        y=[r["feature"] for r in rows], x=xs, orientation="h", marker_color=TEAL,
        text=[f"{v:.3f}" for v in xs], textposition="outside",
        hovertemplate="%{y}<br>Mean |SHAP|: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title="Mean absolute SHAP value (log-odds)", range=[0, max(xs) * 1.2])
    fig.update_yaxes(automargin=True)
    return apply_plotly_theme(fig, height=640, title="Top 20 features by mean absolute SHAP value")


def waterfall(base: float, items: list, other: float, other_count: int, final_logit: float):
    labels = ["Base value"] + [label for label, _ in items]
    values = [base] + [v for _, v in items]
    measures = ["absolute"] + ["relative"] * len(items)
    text = [f"{base:+.2f}"] + [f"{v:+.2f}" for _, v in items]
    if other_count > 0:
        labels.append(f"{other_count} other features")
        values.append(other)
        measures.append("relative")
        text.append(f"{other:+.2f}")
    labels.append("Model output")
    values.append(0)
    measures.append("total")
    text.append(f"{final_logit:+.2f}")

    fig = go.Figure(go.Waterfall(
        orientation="h", y=labels, x=values, measure=measures, text=text, textposition="outside",
        connector=dict(line=dict(color=MUTED, width=1)),
        increasing=dict(marker=dict(color=CRIMSON)),
        decreasing=dict(marker=dict(color=TEAL)),
        totals=dict(marker=dict(color=BLUE))))
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(title="Log-odds of TARGET = 1")
    return apply_plotly_theme(fig, height=max(380, 44 * len(labels) + 120),
                              title="How each feature moves this applicant's score")