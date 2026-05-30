"""
app.py
Streamlit frontend — Pharma Analytics Copilot

Three screens:
  1. Overview Dashboard  — KPI cards, risk score, charts
  2. Consulting Analysis — streaming Claude output, section-by-section
  3. Methodology        — transparency layer, how it works

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub → connect to Streamlit Cloud → set ANTHROPIC_API_KEY secret
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Allow src/ imports
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Load .env in local dev
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from data_loader import load_all
from kpi_engine import compute_kpis
from analysis_engine import (
    stream_analysis, run_analysis,
    SECTION_META, get_rag_color,
)
from prompt_builder import build_methodology_footer


# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Pharma Analytics Copilot",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════
# STYLING
# Refined dark consulting aesthetic — think McKinsey internal tool
# Deep navy base, sharp amber accent, tight editorial typography
# ═══════════════════════════════════════════════════════════

st.markdown("""
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ══════════════════════════════
     DARK THEME (default)
  ══════════════════════════════ */
  :root {
    --bg:         #0d1117;
    --bg-mid:     #161b22;
    --bg-light:   #21262d;
    --border:     #30363d;
    --amber:      #e6a817;
    --red:        #e63946;
    --green:      #2ea04f;
    --watch:      #f4a261;
    --text:       #e6edf3;
    --text-dim:   #8b949e;
    --text-muted: #484f58;
    --btn-text:   #0d1117;
    --code-bg:    #21262d;
  }

  /* ══════════════════════════════
     LIGHT THEME
  ══════════════════════════════ */
  @media (prefers-color-scheme: light) {
    :root {
      --bg:         #f6f8fa;
      --bg-mid:     #ffffff;
      --bg-light:   #eaeef2;
      --border:     #d0d7de;
      --amber:      #b76e00;
      --red:        #cf222e;
      --green:      #1a7f37;
      --watch:      #bc4c00;
      --text:       #1f2328;
      --text-dim:   #57606a;
      --text-muted: #8c959f;
      --btn-text:   #ffffff;
      --code-bg:    #eaeef2;
    }
  }

  /* ── Base ── */
  html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background-color: var(--bg-mid) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] * { color: var(--text) !important; }

  /* ── Headers ── */
  h1 { font-family: 'DM Serif Display', serif !important; font-size: 2rem !important;
       color: var(--text) !important; letter-spacing: -0.02em; }
  h2 { font-family: 'DM Serif Display', serif !important; font-size: 1.35rem !important;
       color: var(--text) !important; }
  h3 { font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
       font-size: 0.95rem !important; color: var(--amber) !important;
       text-transform: uppercase; letter-spacing: 0.08em; }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background: var(--bg-mid) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem 1.25rem !important;
  }
  [data-testid="metric-container"] label {
    font-size: 0.7rem !important;
    color: var(--text-dim) !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.8rem !important;
    color: var(--text) !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: var(--amber) !important;
    color: var(--btn-text) !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.15s ease;
  }
  .stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px);
  }
  .stButton > button:disabled {
    background: var(--text-muted) !important;
    color: var(--bg) !important;
    transform: none !important;
  }

  /* ── Select / radio ── */
  .stSelectbox > div > div, .stRadio > div {
    background: var(--bg-mid) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
  }

  /* ── Expander ── */
  .streamlit-expanderHeader {
    background: var(--bg-mid) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-dim) !important;
    font-size: 0.8rem !important;
  }

  /* ── Divider ── */
  hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

  /* ── Code ── */
  code, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    background: var(--code-bg) !important;
    color: var(--amber) !important;
    padding: 0.15rem 0.4rem;
    border-radius: 3px;
  }

  /* ── Section cards ── */
  .section-card {
    background: var(--bg-mid);
    border: 1px solid var(--border);
    border-left: 3px solid var(--amber);
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
  }
  .section-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
  }

  /* ── KPI direction ── */
  .kpi-good  { color: var(--green); font-weight: 600; }
  .kpi-bad   { color: var(--red);   font-weight: 600; }
  .kpi-watch { color: var(--watch); font-weight: 600; }

  /* ── Flag row ── */
  .flag-row {
    background: var(--bg-light);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
    color: var(--text-dim);
  }

  /* ── Streaming output ── */
  .stream-output {
    font-size: 0.9rem;
    line-height: 1.75;
    color: var(--text);
    background: var(--bg-mid);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
  }

  /* ── Plotly ── */
  .js-plotly-plot { background: transparent !important; }

  /* ── Layout ── */
  .block-container { padding-top: 1.5rem !important; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-dim) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.25rem !important;
    border-bottom: 2px solid transparent !important;
  }
  .stTabs [aria-selected="true"] {
    color: var(--amber) !important;
    border-bottom-color: var(--amber) !important;
  }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════

if "runs_used"       not in st.session_state: st.session_state.runs_used       = 0
if "analysis_result" not in st.session_state: st.session_state.analysis_result = None
if "data_loaded"     not in st.session_state: st.session_state.data_loaded     = False
if "gold"            not in st.session_state: st.session_state.gold            = None
if "kpis"            not in st.session_state: st.session_state.kpis            = None

MAX_RUNS = 3


# ═══════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ═══════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data():
    _, _, gold = load_all()
    kpis = compute_kpis(gold)
    return gold, kpis


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════

def render_sidebar(kpis):
    with st.sidebar:
        st.markdown("## 💊 Pharma Copilot")
        st.markdown(
            "<p style='color:var(--text-dim);font-size:0.75rem;margin-top:-0.5rem;'>"
            "Consultant-grade analytics for oncology commercial teams"
            "</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        # Case summary
        st.markdown("### Case")
        st.markdown(
            "<div style='font-size:0.8rem;color:var(--text-dim);line-height:1.7'>"
            "🏢 <b style='color:var(--text)'>OurCo Pharmaceuticals</b><br>"
            "💊 <b style='color:var(--text)'>AURONIX</b> — NSCLC 2nd line<br>"
            "⚔️ vs <b style='color:var(--text)'>ZYLATEC</b> (CompetitorCo)<br>"
            "📅 Launch: Sept 2024 · Week 20"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # Risk score
        risk = kpis.get("launch_risk", {})
        rag  = risk.get("rag", "RED")
        score= risk.get("score", 0)
        rag_color = get_rag_color(rag)
        rag_emoji = {"RED": "🔴", "AMBER": "🟡", "GREEN": "🟢"}.get(rag, "⚪")

        st.markdown("### Launch Risk")
        st.markdown(
            f"<div style='text-align:center;padding:0.75rem;background:var(--bg-mid);"
            f"border:1px solid var(--border);border-radius:8px;margin-bottom:0.5rem'>"
            f"<div style='font-family:DM Serif Display,serif;font-size:2.5rem;"
            f"color:{rag_color};font-weight:700;line-height:1'>{score}</div>"
            f"<div style='font-size:0.6rem;color:var(--text-dim);letter-spacing:0.1em;"
            f"text-transform:uppercase;margin-top:0.2rem'>out of 100</div>"
            f"<div style='margin-top:0.4rem;font-size:0.75rem;font-weight:700;"
            f"color:{rag_color}'>{rag_emoji} {rag}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # Analysis type selector
        st.markdown("### Analysis Mode")
        analysis_type = st.radio(
            "Select focus",
            options=["full_diagnostic", "current_launch_only", "cost_elimination_only"],
            format_func=lambda x: {
                "full_diagnostic":        "Full Diagnostic",
                "current_launch_only":    "Current Launch Only",
                "cost_elimination_only":  "Cost Elimination",
            }[x],
            label_visibility="collapsed",
        )

        st.divider()

        # Run counter
        runs_left = MAX_RUNS - st.session_state.runs_used
        st.markdown(
            f"<div style='font-size:0.72rem;color:var(--text-dim);text-align:center'>"
            f"Demo runs remaining: <b style='color:var(--text)'>{runs_left} / {MAX_RUNS}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown(
            "<div style='font-size:0.65rem;color:var(--text-muted);line-height:1.6'>"
            "Built by a pharma commercial analyst who thinks in frameworks.<br><br>"
            "Synthetic data · Real methodology · Claude in the backend."
            "</div>",
            unsafe_allow_html=True,
        )

    return analysis_type


# ═══════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════════════════

def render_dashboard(gold, kpis):
    st.markdown("## Overview Dashboard")
    st.markdown(
        "<p style='color:var(--text-dim);font-size:0.85rem;margin-top:-0.5rem;margin-bottom:1.5rem'>"
        "Pre-computed KPIs across 5 diagnostic layers · VEXORIN last launch vs AURONIX current"
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Top KPI cards ──
    risk       = kpis["launch_risk"]
    comparison = gold["launch_comparison"]

    last_freq = None
    cur_freq  = None
    for k in kpis.get("sales_force_effectiveness", []):
        if "Last Launch" in k["name"] and "Calls/Week" in k["name"]:
            last_freq = k["value"]
        if "Current Launch" in k["name"] and "Calls/Week" in k["name"]:
            cur_freq = k["value"]

    last_share_gap = float(comparison.loc["market_share_gap_wk1_8", "LAST_LAUNCH"])
    cur_share_gap  = float(comparison.loc["market_share_gap_wk1_8", "CURRENT_LAUNCH"])
    sellthrough    = next(
        (k["value"] for k in kpis.get("revenue_diagnostic", []) if "Sell-Through" in k["name"]),
        65.4
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Call Freq D8+ HCPs · Last Launch",
            f"{last_freq:.1f}/wk" if last_freq else "—",
            delta="vs 3.0 benchmark",
            delta_color="inverse",
        )
    with c2:
        st.metric(
            "Call Freq D8+ HCPs · Current",
            f"{cur_freq:.1f}/wk" if cur_freq else "—",
            delta=f"{cur_freq - last_freq:+.1f} vs last launch" if cur_freq and last_freq else None,
        )
    with c3:
        st.metric(
            "Market Share Gap vs ZYLATEC wk1-8",
            f"{last_share_gap:+.1f}pp",
            delta=f"{cur_share_gap - last_share_gap:+.1f}pp current launch",
        )
    with c4:
        st.metric(
            "VEXORIN Sell-Through Rate",
            f"{sellthrough:.0f}%",
            delta="vs 85% benchmark",
            delta_color="inverse",
        )

    st.divider()

    # ── Risk flags ──
    flags = gold["risk_flags"]
    if len(flags) > 0:
        red_count   = len(flags[flags["severity"] == "RED"])
        amber_count = len(flags[flags["severity"] == "AMBER"])
        st.markdown(
            f"<div style='background:var(--bg);border:1px solid #e63946;border-radius:8px;"
            f"padding:0.75rem 1.25rem;margin-bottom:1rem'>"
            f"<span style='color:#e63946;font-weight:700;font-size:0.8rem'>"
            f"⚠️ ACTIVE RISK FLAGS — {red_count} RED · {amber_count} AMBER</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        for _, row in flags.iterrows():
            icon  = "🔴" if row["severity"] == "RED" else "🟡"
            color = "#e63946" if row["severity"] == "RED" else "#f4a261"
            st.markdown(
                f"<div class='flag-row'>"
                f"{icon} <b style='color:{color}'>[{row['flag_type']}]</b> "
                f"<span style='color:var(--text-dim)'>{row['territory_name']} —</span> "
                f"{row['message']}"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.divider()

    # ── Charts row ──
    col_l, col_r = st.columns(2)

    with col_l:
        _chart_launch_comparison(comparison)

    with col_r:
        _chart_territory_performance(gold["territory_performance"])

    st.divider()

    # ── Full KPI table ──
    with st.expander("📋 All KPIs — Expanded View", expanded=False):
        _render_kpi_table(kpis)


def _chart_launch_comparison(comparison: pd.DataFrame):
    st.markdown("### Launch Comparison — Key Metrics")

    metrics_to_plot = [
        ("avg_calls_per_week_d8plus_early", "Calls/wk D8+ HCPs (wk1-8)"),
        ("our_avg_market_share_wk1_8",      "Our Market Share % (wk1-8)"),
        ("zylatec_avg_market_share_wk1_8",  "ZYLATEC Market Share % (wk1-8)"),
        ("overall_conversion_rate_pct",      "Conversion Rate %"),
    ]

    labels, last_vals, cur_vals = [], [], []
    for metric, label in metrics_to_plot:
        if metric in comparison.index:
            labels.append(label)
            last_vals.append(float(comparison.loc[metric, "LAST_LAUNCH"]))
            cur_vals.append(float(comparison.loc[metric, "CURRENT_LAUNCH"]))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="VEXORIN (Last Launch)",
        x=labels, y=last_vals,
        marker_color="#e63946",
        marker_line_width=0,
    ))
    fig.add_trace(go.Bar(
        name="AURONIX (Current)",
        x=labels, y=cur_vals,
        marker_color="#e6a817",
        marker_line_width=0,
    ))
    fig.update_layout(
        barmode="group",
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font=dict(family="DM Sans", color="#8b949e", size=11),
        legend=dict(
            bgcolor="#161b22", bordercolor="#30363d", borderwidth=1,
            font=dict(size=10, color="#e6edf3"),
        ),
        xaxis=dict(tickfont=dict(size=9), gridcolor="#21262d", linecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_territory_performance(ter_perf: pd.DataFrame):
    st.markdown("### Territory — Market Share vs Potential")

    last = ter_perf[ter_perf["launch_phase"] == "LAST_LAUNCH"].copy()
    if last.empty:
        st.info("No territory data available.")
        return

    last["our_share_label"]  = last["our_avg_market_share"].round(1).astype(str) + "%"
    last["gap_color"]        = last.apply(
        lambda r: "#e63946" if (r["zylatec_avg_market_share"] - r["our_avg_market_share"]) > 5
                  else "#2ea04f",
        axis=1,
    )

    fig = px.scatter(
        last,
        x="market_potential_score",
        y="our_avg_market_share",
        size="high_decile_hcps",
        color="gap_color",
        color_discrete_map="identity",
        hover_name="territory_name",
        hover_data={
            "market_potential_score": True,
            "our_avg_market_share":   True,
            "zylatec_avg_market_share": True,
            "high_decile_hcps":       True,
            "gap_color":              False,
        },
        labels={
            "market_potential_score": "Territory Market Potential",
            "our_avg_market_share":   "Our Market Share %",
        },
    )
    fig.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font=dict(family="DM Sans", color="#8b949e", size=11),
        showlegend=False,
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "<p style='font-size:0.7rem;color:var(--text-muted);margin-top:-0.5rem'>"
        "🔴 ZYLATEC leads by >5pp · 🟢 We lead or at parity · "
        "Bubble size = high-decile HCP count"
        "</p>",
        unsafe_allow_html=True,
    )


def _render_kpi_table(kpis):
    sections = {
        "revenue_diagnostic":       "Layer 1 — Revenue Diagnostic",
        "sales_force_effectiveness": "Layer 2 — Sales Force Effectiveness",
        "territory_alignment":      "Layer 3 — Territory Alignment",
        "rep_alignment":            "Layer 4 — Rep Experience Alignment",
        "competitive_position":     "Layer 5 — Competitive Position",
    }
    direction_icon = {"GOOD": "✅", "BAD": "❌", "WATCH": "⚠️"}

    for key, label in sections.items():
        layer_kpis = kpis.get(key, [])
        if not layer_kpis:
            continue
        st.markdown(f"**{label}**")
        rows = []
        for k in layer_kpis:
            rows.append({
                "":          direction_icon.get(k["direction"], "?"),
                "KPI":       k["name"],
                "Value":     f"{k['value']} {k['unit']}",
                "Benchmark": k["benchmark"],
                "So What":   k["so_what"][:90] + "..." if len(k["so_what"]) > 90 else k["so_what"],
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "":          st.column_config.TextColumn(width="small"),
                "KPI":       st.column_config.TextColumn(width="medium"),
                "Value":     st.column_config.TextColumn(width="small"),
                "Benchmark": st.column_config.TextColumn(width="medium"),
                "So What":   st.column_config.TextColumn(width="large"),
            },
        )
        st.markdown("")


# ═══════════════════════════════════════════════════════════
# TAB 2 — CONSULTING ANALYSIS
# ═══════════════════════════════════════════════════════════

def render_analysis(kpis, analysis_type):
    st.markdown("## Consulting Analysis")
    st.markdown(
        "<p style='color:var(--text-dim);font-size:0.85rem;margin-top:-0.5rem;margin-bottom:1.5rem'>"
        "Minto Pyramid · Issue Tree · Hypothesis-Driven Recommendations"
        "</p>",
        unsafe_allow_html=True,
    )

    runs_left = MAX_RUNS - st.session_state.runs_used
    at_limit  = runs_left <= 0

    # ── Mode label ──
    mode_labels = {
        "full_diagnostic":       "Full Diagnostic — all 6 sections, complete issue tree",
        "current_launch_only":   "Current Launch Only — AURONIX status and 30-day actions",
        "cost_elimination_only": "Cost Elimination — proves the problem is not cost",
    }
    st.markdown(
        f"<div style='background:var(--bg-mid);border:1px solid var(--border);border-radius:6px;"
        f"padding:0.6rem 1rem;margin-bottom:1rem;font-size:0.8rem;color:var(--text-dim)'>"
        f"📋 Mode: <b style='color:var(--text)'>{mode_labels.get(analysis_type, analysis_type)}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Run button ──
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button(
            "▶ Run Analysis",
            disabled=at_limit,
            use_container_width=True,
        )
    with col_info:
        if at_limit:
            st.markdown(
                "<div style='padding:0.6rem;background:var(--bg);border:1px solid #e63946;"
                "border-radius:6px;font-size:0.78rem;color:#e63946;margin-top:0.2rem'>"
                "Demo limit reached (3 runs per session). "
                "Refresh the page to reset, or contact me for full access."
                "</div>",
                unsafe_allow_html=True,
            )
        elif st.session_state.analysis_result:
            st.markdown(
                f"<div style='padding:0.6rem;font-size:0.78rem;color:var(--text-dim);margin-top:0.4rem'>"
                f"Last run: {st.session_state.analysis_result.get('elapsed_sec','?')}s · "
                f"{st.session_state.analysis_result.get('usage',{}).get('cost_display','?')} · "
                f"{st.session_state.analysis_result.get('usage',{}).get('total_tokens','?')} tokens"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Streaming output ──
    if run_clicked and not at_limit:
        st.session_state.runs_used += 1
        st.session_state.analysis_result = None

        st.divider()
        st.markdown("### Analysis in progress...")

        output_placeholder = st.empty()
        full_text = ""

        try:
            for chunk in stream_analysis(kpis, analysis_type):
                full_text += chunk
                output_placeholder.markdown(
                    f"<div class='stream-output'>{full_text}</div>",
                    unsafe_allow_html=True,
                )

            # Parse sections from completed text
            from analysis_engine import _parse_sections
            sections = _parse_sections(full_text)

            st.session_state.analysis_result = {
                "full_text":    full_text,
                "sections":     sections,
                "elapsed_sec":  "—",
                "usage":        {"total_tokens": "—", "cost_display": "—"},
                "methodology":  build_methodology_footer(kpis),
                "error":        None,
            }

            # Re-render as structured sections
            output_placeholder.empty()
            _render_sections(sections)

        except Exception as e:
            output_placeholder.error(f"Analysis failed: {str(e)}")

    # ── Show previous result if exists ──
    elif st.session_state.analysis_result and not run_clicked:
        result = st.session_state.analysis_result
        if result.get("error"):
            st.error(result["error"])
        elif result.get("sections"):
            st.divider()
            _render_sections(result["sections"])

    elif not st.session_state.analysis_result:
        st.markdown(
            "<div style='background:var(--bg-mid);border:1px dashed var(--border);border-radius:8px;"
            "padding:2.5rem;text-align:center;margin-top:1rem'>"
            "<div style='font-size:1.5rem;margin-bottom:0.5rem'>🎯</div>"
            "<div style='color:var(--text-dim);font-size:0.85rem'>"
            "Click <b style='color:var(--amber)'>▶ Run Analysis</b> to generate "
            "a consultant-grade diagnostic using Claude."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def _render_sections(sections: dict):
    """Renders parsed Claude output as styled section cards."""
    section_order = [
        "SITUATION", "COMPLICATION", "ISSUE TREE",
        "KEY FINDINGS", "RECOMMENDATIONS", "LAUNCH RISK VERDICT",
        "FULL OUTPUT",
    ]

    border_colors = {
        "SITUATION":          "#457b9d",
        "COMPLICATION":       "#e63946",
        "ISSUE TREE":         "#6e40c9",
        "KEY FINDINGS":       "#2ea04f",
        "RECOMMENDATIONS":    "#e6a817",
        "LAUNCH RISK VERDICT":"#e63946",
        "FULL OUTPUT":        "#30363d",
    }

    rendered = set()
    for key in section_order:
        if key in sections:
            _render_one_section(key, sections[key], border_colors.get(key, "#30363d"))
            rendered.add(key)

    # Any remaining sections Claude added
    for key, content in sections.items():
        if key not in rendered:
            _render_one_section(key, content, "#30363d")


def _render_one_section(key: str, content: str, border_color: str):
    meta  = SECTION_META.get(key, {"icon": "•", "label": key})
    icon  = meta.get("icon", "•")
    label = meta.get("label", key)

    st.markdown(
        f"<div style='background:var(--bg-mid);border:1px solid var(--border);"
        f"border-left:3px solid {border_color};border-radius:8px;"
        f"padding:1.25rem 1.5rem;margin-bottom:0.75rem'>"
        f"<div style='font-size:0.65rem;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.1em;color:var(--text-muted);margin-bottom:0.5rem'>"
        f"{icon} {label}</div>"
        f"<div style='font-size:0.88rem;line-height:1.75;color:var(--text)'>"
        f"{_md_to_html(content)}"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def _md_to_html(text: str) -> str:
    """Minimal markdown-to-HTML for use inside HTML blocks."""
    import re
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Newlines
    text = text.replace("\n\n", "<br><br>").replace("\n", "<br>")
    # Numbered lists
    text = re.sub(r'(\d+)\. ', r'<br><b style="color:var(--amber)">\1.</b> ', text)
    return text


# ═══════════════════════════════════════════════════════════
# TAB 3 — METHODOLOGY
# ═══════════════════════════════════════════════════════════

def render_methodology(kpis):
    st.markdown("## Methodology")
    st.markdown(
        "<p style='color:var(--text-dim);font-size:0.85rem;margin-top:-0.5rem;margin-bottom:1.5rem'>"
        "How the analysis was produced — transparently"
        "</p>",
        unsafe_allow_html=True,
    )

    # Architecture diagram as styled HTML
    st.markdown(
        """
        <div style='background:var(--bg-mid);border:1px solid var(--border);border-radius:8px;
                    padding:1.5rem;margin-bottom:1.5rem;font-family:JetBrains Mono,monospace;
                    font-size:0.75rem;color:var(--text-dim);line-height:2'>
          <span style='color:var(--amber);font-weight:600'>BRONZE</span>
          &nbsp;→&nbsp; Raw CSVs (10 tables: HCP, rep, territory, call activity, competitor intel, COGS, HR...)
          <br>
          <span style='color:var(--amber);font-weight:600'>SILVER</span>
          &nbsp;→&nbsp; Cleaned, typed, joined — dates parsed, decile bands added, experience flags set
          <br>
          <span style='color:var(--amber);font-weight:600'>GOLD</span>
          &nbsp;&nbsp;&nbsp;→&nbsp; Analysis-ready views: rep×HCP coverage, frequency by decile, territory performance,
                    launch comparison, risk flags
          <br>
          <span style='color:#6e40c9;font-weight:600'>KPI ENGINE</span>
          &nbsp;→&nbsp; 23 metrics across 5 layers, each with benchmark + direction + so-what
          <br>
          <span style='color:#e63946;font-weight:600'>PROMPT BUILDER</span>
          &nbsp;→&nbsp; Minto Pyramid framework + issue tree logic + 6 non-negotiable rules injected to Claude
          <br>
          <span style='color:#2ea04f;font-weight:600'>CLAUDE</span>
          &nbsp;&nbsp;&nbsp;→&nbsp; Reasons over evidence, applies so-what discipline, produces consulting output
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Framework explanation
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Analytical Framework")
        framework_items = [
            ("Minto Pyramid", "Situation → Complication → Question → Answer — standard McKinsey narrative structure"),
            ("Issue Tree (MECE)", "Profit decomposed at each level, hypotheses eliminated with data before the next level"),
            ("So-What Discipline", "Every number must carry an implication. Claude is explicitly prohibited from stating observations without business consequences"),
            ("Hypothesis-Driven Recs", "3 recommendations, sequenced by root cause first. Each names a specific action, urgency, and risk mitigated"),
        ]
        for name, desc in framework_items:
            st.markdown(
                f"<div style='background:var(--bg-mid);border:1px solid var(--border);border-radius:6px;"
                f"padding:0.75rem 1rem;margin-bottom:0.5rem'>"
                f"<div style='font-size:0.75rem;font-weight:600;color:var(--amber);margin-bottom:0.25rem'>"
                f"{name}</div>"
                f"<div style='font-size:0.78rem;color:var(--text-dim);line-height:1.5'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("### What Claude Was Told Not To Do")
        rules = [
            "Say 'the data shows' — it must say what the data means",
            "State an observation without a business implication",
            "Make vague recommendations (no 'improve coverage')",
            "Skip a level of the issue tree",
            "Hedge on findings that are unambiguous",
            "Write for a non-pharma audience (no explaining what a decile is)",
            "Invent numbers beyond the KPIs provided",
        ]
        for rule in rules:
            st.markdown(
                f"<div style='font-size:0.8rem;color:var(--text-dim);padding:0.3rem 0;"
                f"border-bottom:1px solid #21262d'>❌ {rule}</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # Data dictionary
    st.markdown("### Source Data Dictionary")
    try:
        dict_path = Path(__file__).resolve().parent / "data" / "synthetic" / "data_dictionary.csv"
        data_dict = pd.read_csv(dict_path)
        tables    = sorted(data_dict["table_name"].unique())
        selected  = st.selectbox("Select table", tables)
        filtered  = data_dict[data_dict["table_name"] == selected][
            ["column_name", "description", "data_type", "example_value", "required"]
        ]
        st.dataframe(filtered, use_container_width=True, hide_index=True)
    except Exception:
        st.info("Data dictionary not found. Run generate_data.py first.")

    st.divider()

    # Honest footer
    footer = build_methodology_footer(kpis)
    st.markdown(footer)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    # Load data
    with st.spinner("Loading data and computing KPIs..."):
        try:
            gold, kpis = load_data()
        except FileNotFoundError as e:
            st.error(
                f"**Data not found.** Run `generate_data.py` first to create the synthetic dataset.\n\n"
                f"```\npython generate_data.py\n```\n\n{e}"
            )
            st.stop()

    # Sidebar — returns selected analysis type
    analysis_type = render_sidebar(kpis)

    # Header
    st.markdown(
        "<div style='margin-bottom:0.5rem'>"
        "<h1 style='margin-bottom:0'>Pharma Analytics Copilot</h1>"
        "<p style='color:var(--text-dim);font-size:0.82rem;margin-top:0.25rem'>"
        "AURONIX Launch · NSCLC 2nd line · OurCo vs ZYLATEC · "
        "<span style='color:var(--amber)'>20 weeks in</span>"
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "📊  Overview Dashboard",
        "🎯  Consulting Analysis",
        "🔬  Methodology",
    ])

    with tab1:
        render_dashboard(gold, kpis)

    with tab2:
        render_analysis(kpis, analysis_type)

    with tab3:
        render_methodology(kpis)


if __name__ == "__main__":
    main()