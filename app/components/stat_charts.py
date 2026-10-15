"""Plotly charts for the Statistical Analysis page. All inputs come from saved reports."""
import plotly.graph_objects as go

from app.theme import AMBER, CRIMSON, MUTED, TEAL, TEXT, apply_plotly_theme


def class_donut(counts: dict):
    fig = go.Figure(go.Pie(
        labels=["TARGET = 0 (no payment difficulty)", "TARGET = 1 (payment difficulty)"],
        values=[counts["0"], counts["1"]], hole=0.65, sort=False, textinfo="percent",
        marker=dict(colors=[TEAL, CRIMSON], line=dict(color="#0B1220", width=2)),
        hovertemplate="%{label}<br>%{value:,} applicants<extra></extra>"))
    fig.update_layout(legend=dict(orientation="h", y=-0.05))
    return apply_plotly_theme(fig, height=320, title="Target class balance (application_train)")


def effect_size_bar(numeric: list):
    rows = sorted(numeric, key=lambda r: abs(r["rank_biserial"]))
    feats = [r["feature"] for r in rows]
    fig = go.Figure()
    for name, color, test in [("Higher in TARGET = 1", CRIMSON, lambda v: v > 0),
                              ("Lower in TARGET = 1", TEAL, lambda v: v <= 0)]:
        fig.add_trace(go.Bar(
            y=feats, orientation="h", name=name, marker_color=color,
            x=[abs(r["rank_biserial"]) if test(r["rank_biserial"]) else None for r in rows],
            hovertemplate="%{y}<br>Effect size: %{x:.3f}<extra></extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Effect size (absolute rank-biserial correlation)")
    apply_plotly_theme(fig, height=540, title="How strongly each numeric feature separates the two classes")
    fig.update_layout(margin=dict(l=10, r=10, t=100, b=10),
                      legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom"))
    return fig


def correlation_heatmap(corr: dict):
    labels = corr["labels"]
    fig = go.Figure(go.Heatmap(
        z=corr["matrix"], x=labels, y=labels, zmin=-1, zmax=1,
        colorscale=[[0, TEAL], [0.5, "#111A2E"], [1, CRIMSON]],
        texttemplate="%{z:.2f}", textfont=dict(size=10, color=TEXT),
        colorbar=dict(title=dict(text="Spearman"), tickfont=dict(color=MUTED)),
        hovertemplate="%{y} vs %{x}<br>%{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-40)
    return apply_plotly_theme(fig, height=640, title="Spearman rank correlation: numeric features and TARGET")


def category_rate_bar(rates: list, overall_rate: float):
    rows = sorted(rates, key=lambda r: r["default_rate"])
    fig = go.Figure(go.Bar(
        y=[r["category"] for r in rows], x=[r["default_rate"] * 100 for r in rows], orientation="h",
        marker_color=[CRIMSON if r["default_rate"] > overall_rate else TEAL for r in rows],
        text=[f"{r['default_rate'] * 100:.1f}%" for r in rows], textposition="outside",
        customdata=[r["applicants"] for r in rows],
        hovertemplate="%{y}<br>TARGET = 1 rate: %{x:.2f}%<br>Applicants: %{customdata:,}<extra></extra>"))
    fig.add_vline(x=overall_rate * 100, line_dash="dash", line_color=AMBER)
    fig.add_annotation(x=overall_rate * 100, y=1.0, xref="x", yref="paper", yanchor="bottom",
                       text=f"Overall {overall_rate * 100:.1f}%", showarrow=False,
                       font=dict(color=AMBER, size=12))
    top = max(r["default_rate"] for r in rows) * 100
    fig.update_xaxes(title="TARGET = 1 rate (%)", range=[0, top * 1.25])
    apply_plotly_theme(fig, height=max(320, 38 * len(rows) + 140),
                       title="Observed TARGET = 1 rate by category")
    fig.update_layout(margin=dict(l=10, r=10, t=90, b=10))
    return fig


def missing_bar(features: list, pct: list):
    fig = go.Figure(go.Bar(
        y=features[::-1], x=pct[::-1], orientation="h", marker_color=AMBER,
        text=[f"{v:.1f}%" for v in pct[::-1]], textposition="outside",
        hovertemplate="%{y}<br>Missing: %{x:.1f}%<extra></extra>"))
    fig.update_xaxes(title="Missing values (%)", range=[0, 105])
    return apply_plotly_theme(fig, height=620, title="Top 20 model inputs by share of missing values")