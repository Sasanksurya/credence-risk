"""Visual identity: 'Midnight Ledger' dark theme with teal and amber accents."""
import streamlit as st

TEAL, AMBER, ORANGE, CRIMSON, BLUE = "#2DD4BF", "#F5B841", "#FB923C", "#F43F5E", "#60A5FA"
INK, PANEL, BORDER, TEXT, MUTED = "#0B1220", "#111A2E", "#22304D", "#E6EDF7", "#8EA0BD"
BAND_COLORS = {"Low": TEAL, "Moderate": AMBER, "Elevated": ORANGE, "High": CRIMSON}

_CSS = """
<style>
.stApp { background: radial-gradient(1100px 520px at 8% -8%, #12304a 0%, %INK% 58%); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1280px; }
section[data-testid="stSidebar"] { background: #0D1630; border-right: 1px solid %BORDER%; }
.cr-brand { font-size: 1.35rem; font-weight: 700; letter-spacing: .02em; color: %TEXT%; }
.cr-brand span { color: %TEAL%; }
.cr-brand-sub { color: %MUTED%; font-size: .78rem; margin-bottom: 1.2rem; }
.cr-eyebrow { color: %TEAL%; font-size: .75rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
.cr-title { font-size: 2.3rem; font-weight: 750; line-height: 1.15; margin: .2rem 0 .4rem 0; color: %TEXT%; }
.cr-lead { color: %MUTED%; font-size: 1.02rem; max-width: 62rem; margin-bottom: 1.4rem; }
.cr-metric { background: %PANEL%; border: 1px solid %BORDER%; border-left: 3px solid var(--accent);
  border-radius: 12px; padding: 16px 18px; height: 100%; }
.cr-label { color: %MUTED%; font-size: .72rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; }
.cr-value { color: %TEXT%; font-size: 2rem; font-weight: 700; margin: 4px 0 2px 0; font-variant-numeric: tabular-nums; }
.cr-sub { color: %MUTED%; font-size: .78rem; }
.cr-card { background: %PANEL%; border: 1px solid %BORDER%; border-radius: 12px; padding: 18px 20px; height: 100%; }
.cr-card h4 { margin: 0 0 .6rem 0; color: %TEXT%; font-size: 1rem; }
.cr-card ul { margin: 0; padding-left: 1.1rem; color: #C3CFE3; font-size: .9rem; line-height: 1.65; }
.cr-banner { border-radius: 10px; padding: 12px 16px; font-size: .9rem; margin: 1.1rem 0; color: %TEXT%;
  background: rgba(45,212,191,.08); border: 1px solid rgba(45,212,191,.35); }
.cr-banner.warn { background: rgba(245,184,65,.08); border-color: rgba(245,184,65,.4); }
.cr-section { margin: 1.6rem 0 .6rem 0; font-size: 1.15rem; font-weight: 650; color: %TEXT%; }
.cr-section small { display: block; color: %MUTED%; font-weight: 400; font-size: .82rem; margin-top: 2px; }
table.cr-table { width: 100%; border-collapse: collapse; font-size: .9rem; }
table.cr-table td { padding: 7px 4px; border-bottom: 1px solid %BORDER%; color: #C3CFE3; }
table.cr-table td:first-child { color: %MUTED%; width: 46%; }
.cr-foot { color: %MUTED%; font-size: .75rem; border-top: 1px solid %BORDER%; padding-top: .8rem; margin-top: 1.5rem; }
</style>
"""


def inject_css() -> None:
    css = _CSS
    for token, val in {"%INK%": INK, "%PANEL%": PANEL, "%BORDER%": BORDER, "%TEXT%": TEXT,
                       "%MUTED%": MUTED, "%TEAL%": TEAL}.items():
        css = css.replace(token, val)
    st.markdown(css, unsafe_allow_html=True)


def apply_plotly_theme(fig, height: int = 340, title: str = ""):
    fig.update_layout(
        height=height, title=dict(text=title, font=dict(size=14, color=TEXT)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, family="Segoe UI, Arial, sans-serif"),
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=BORDER, zeroline=False)
    fig.update_yaxes(gridcolor=BORDER, zeroline=False)
    return fig