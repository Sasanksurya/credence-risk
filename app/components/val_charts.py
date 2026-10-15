"""Plotly charts for the Model Validation page. Inputs come from saved validation reports."""
import plotly.graph_objects as go

from app.theme import AMBER, BLUE, CRIMSON, MUTED, TEAL, apply_plotly_theme


def roc_chart(roc: dict, auc: float):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=f"Model (AUC = {auc:.4f})",
                             line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="No skill",
                             line=dict(color=MUTED, dash="dash")))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    fig.update_layout(legend=dict(x=0.4, y=0.08))
    return apply_plotly_theme(fig, height=360, title="ROC curve")


def pr_chart(pr: dict, ap: float, prevalence: float):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name=f"Model (AP = {ap:.4f})",
                             line=dict(color=TEAL, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[prevalence, prevalence], mode="lines",
                             name=f"No skill ({prevalence:.3f})", line=dict(color=MUTED, dash="dash")))
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1.02])
    fig.update_layout(legend=dict(x=0.35, y=0.95))
    return apply_plotly_theme(fig, height=360, title="Precision-recall curve")


def threshold_chart(rows: list):
    xs = [r["threshold"] for r in rows]
    precision = [r["precision"] if r["flagged_share"] > 0 else None for r in rows]
    fig = go.Figure()
    for name, ys, color in [("Precision", precision, AMBER),
                            ("Recall", [r["recall"] for r in rows], TEAL),
                            ("F1", [r["f1"] for r in rows], BLUE)]:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=name,
                                 line=dict(color=color, width=3), connectgaps=False))
    fig.update_xaxes(title="Classification threshold")
    fig.update_yaxes(title="Score", range=[0, 1.02])
    apply_plotly_theme(fig, height=360, title="Precision, recall and F1 by threshold")
    fig.update_layout(margin=dict(l=10, r=10, t=95, b=10),
                      legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom"))
    return fig


def calibration_chart(cal: dict):
    xs, ys = cal["mean_predicted"], cal["observed_rate"]
    top = max(max(xs), max(ys), 0.05) * 1.1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Perfect calibration",
                             line=dict(color=MUTED, dash="dash")))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="Model (10 groups)",
                             line=dict(color=CRIMSON, width=3), marker=dict(size=9)))
    fig.update_xaxes(title="Mean predicted probability", range=[0, top])
    fig.update_yaxes(title="Observed TARGET = 1 rate", range=[0, top])
    fig.update_layout(legend=dict(x=0.02, y=0.98))
    return apply_plotly_theme(fig, height=380, title="Calibration (reliability) curve")