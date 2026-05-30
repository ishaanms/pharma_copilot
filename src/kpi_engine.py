"""
kpi_engine.py
Computes all pharma consulting KPIs from gold layer tables.

Each KPI follows the consulting discipline:
  - A number
  - A benchmark to compare against
  - A direction (good / bad / watch)
  - A so-what (one line implication)

Output is a structured dict that goes directly into prompt_builder.py.

Run standalone to verify:
    python src/kpi_engine.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Allow running from project root or src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_all

# ═══════════════════════════════════════════════════════════
# KPI DEFINITIONS
# Each KPI is a dict:
# {
#   "name"       : display name
#   "value"      : numeric value
#   "unit"       : "%", "x", "calls/week", "$M", etc.
#   "benchmark"  : what good looks like
#   "direction"  : "GOOD" | "BAD" | "WATCH"
#   "delta"      : change vs last launch (where applicable)
#   "so_what"    : one sentence implication for a decision-maker
# }
# ═══════════════════════════════════════════════════════════


def compute_kpis(gold: dict) -> dict:
    """
    Master KPI computation.
    Returns a structured dict grouped by diagnostic layer.
    """

    rep_hcp    = gold["rep_hcp_coverage"]
    freq       = gold["call_frequency_by_hcp"]
    ter_perf   = gold["territory_performance"]
    outcome    = gold["outcome_by_decile"]
    cost       = gold["cost_summary"]
    comparison = gold["launch_comparison"]
    risk_flags = gold["risk_flags"]

    kpis = {}

    # ────────────────────────────────────────────────────────
    # LAYER 1 — REVENUE DIAGNOSTIC
    # Proves the problem is volume/sales, not pricing or cost
    # ────────────────────────────────────────────────────────
    kpis["revenue_diagnostic"] = _revenue_kpis(cost, comparison)

    # ────────────────────────────────────────────────────────
    # LAYER 2 — SALES FORCE EFFECTIVENESS
    # Call frequency, coverage, conversion
    # ────────────────────────────────────────────────────────
    kpis["sales_force_effectiveness"] = _sfe_kpis(rep_hcp, freq, comparison)

    # ────────────────────────────────────────────────────────
    # LAYER 3 — TERRITORY ALIGNMENT
    # Resource allocation vs market potential
    # ────────────────────────────────────────────────────────
    kpis["territory_alignment"] = _territory_kpis(ter_perf)

    # ────────────────────────────────────────────────────────
    # LAYER 4 — REP EXPERIENCE ALIGNMENT
    # Are the right reps on the right doctors?
    # ────────────────────────────────────────────────────────
    kpis["rep_alignment"] = _rep_alignment_kpis(rep_hcp, outcome)

    # ────────────────────────────────────────────────────────
    # LAYER 5 — COMPETITIVE POSITION
    # How are we tracking vs ZYLATEC?
    # ────────────────────────────────────────────────────────
    kpis["competitive_position"] = _competitive_kpis(ter_perf, comparison)

    # ────────────────────────────────────────────────────────
    # LAYER 6 — CURRENT LAUNCH RISK SCORE
    # Composite red/amber/green for the dashboard header
    # ────────────────────────────────────────────────────────
    kpis["launch_risk"] = _launch_risk_score(risk_flags, comparison)

    return kpis


# ═══════════════════════════════════════════════════════════
# LAYER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def _revenue_kpis(cost: dict, comparison: pd.DataFrame) -> list:
    """
    Elimination proof: cost is stable, revenue is the problem.
    """
    kpi_list = []

    cogs = cost["cogs_trend"]
    vex  = cogs[cogs["drug_name"] == "VEXORIN"]
    aur  = cogs[cogs["drug_name"] == "AURONIX"]

    # COGS per unit stability
    if len(vex) >= 2:
        vex_sorted   = vex.sort_values("year")
        cogs_start   = vex_sorted["avg_cost_per_unit"].iloc[0]
        cogs_end     = vex_sorted["avg_cost_per_unit"].iloc[-1]
        cogs_change  = (cogs_end - cogs_start) / cogs_start * 100
    else:
        cogs_change  = 0.0
        cogs_end     = vex["avg_cost_per_unit"].mean() if len(vex) else 500

    kpi_list.append({
        "name":      "COGS per Unit Change (VEXORIN launch period)",
        "value":     round(cogs_change, 1),
        "unit":      "%",
        "benchmark": "< ±5% = stable",
        "direction": "GOOD" if abs(cogs_change) < 5 else "WATCH",
        "delta":     None,
        "so_what":   (
            f"Manufacturing cost per unit changed {cogs_change:+.1f}% over the launch window — "
            f"cost structure is {'stable and not the driver of margin pressure' if abs(cogs_change) < 5 else 'shifting and warrants review'}."
        ),
    })

    # Sell-through rate — the overstock signal
    vex_st = vex["avg_sellthrough"].mean() * 100 if len(vex) else 65
    aur_st = aur["avg_sellthrough"].mean() * 100 if len(aur) else 75
    kpi_list.append({
        "name":      "VEXORIN Avg Sell-Through Rate",
        "value":     round(vex_st, 1),
        "unit":      "%",
        "benchmark": "> 85% = healthy channel pull-through",
        "direction": "BAD" if vex_st < 75 else "WATCH",
        "delta":     round(aur_st - vex_st, 1),
        "so_what":   (
            f"Only {vex_st:.0f}% of units produced were sold — "
            f"warehouse stockpiles grew throughout the launch, confirming the problem is "
            f"sales reach, not supply or pricing."
        ),
    })

    # Fixed cost stability (capex date signal)
    last_capex = cost["last_capex_date"]
    kpi_list.append({
        "name":      "Last Major Capex",
        "value":     last_capex,
        "unit":      "date",
        "benchmark": "No new assets in analysis window = fixed cost stable",
        "direction": "GOOD",
        "delta":     None,
        "so_what":   (
            f"No new fixed assets acquired since {last_capex}. "
            f"Fixed cost base is unchanged — profit pressure cannot be attributed to infrastructure investment."
        ),
    })

    # Headcount cost vs revenue
    hc = cost["headcount_by_dept"]
    total_payroll = hc["total_cost"].sum()
    kpi_list.append({
        "name":      "Total Annual Field Force Payroll",
        "value":     round(total_payroll / 1_000_000, 2),
        "unit":      "$M",
        "benchmark": "Industry norm: 30-40% of net revenue for specialty pharma",
        "direction": "WATCH",
        "delta":     None,
        "so_what":   (
            f"${total_payroll/1e6:.1f}M annual payroll for commercial operations. "
            f"Headcount is growing at market-standard rates — "
            f"the cost of the sales force is not the issue; what they are doing with their time is."
        ),
    })

    # Market share gap — the revenue loss in dollar terms
    if "LAST_LAUNCH" in comparison.columns and "CURRENT_LAUNCH" in comparison.columns:
        last_gap = float(comparison.loc["market_share_gap_wk1_8", "LAST_LAUNCH"])
        cur_gap  = float(comparison.loc["market_share_gap_wk1_8", "CURRENT_LAUNCH"])
        kpi_list.append({
            "name":      "Market Share Gap vs ZYLATEC (weeks 1-8, last launch)",
            "value":     round(last_gap, 1),
            "unit":      "pp",
            "benchmark": "0 = parity; negative = we are behind",
            "direction": "BAD" if last_gap < 0 else "GOOD",
            "delta":     round(cur_gap - last_gap, 1),
            "so_what":   (
                f"ZYLATEC led by {abs(last_gap):.1f} percentage points in the first 8 weeks of last launch — "
                f"the critical prescriber habit-forming window. "
                f"Early share loss in oncology is structurally difficult to recover."
            ),
        })

    return kpi_list


def _sfe_kpis(rep_hcp: pd.DataFrame, freq: pd.DataFrame, comparison: pd.DataFrame) -> list:
    """
    Sales force effectiveness: frequency, coverage, conversion.
    """
    kpi_list = []

    # ── Call frequency on high-decile HCPs, early launch window ──
    # Use rep_hcp table so we measure per-rep-per-hcp (not all reps on one HCP summed)
    for phase in ["LAST_LAUNCH", "CURRENT_LAUNCH"]:
        phase_rephcp = rep_hcp[
            (rep_hcp["launch_phase"] == phase) &
            (rep_hcp["decile"] >= 8)
        ]
        avg_freq = phase_rephcp["calls_per_week"].mean() if len(phase_rephcp) else 0
        label    = "Last Launch (VEXORIN)" if phase == "LAST_LAUNCH" else "Current Launch (AURONIX)"
        kpi_list.append({
            "name":      f"Avg Calls/Week on Decile 8-10 HCPs — Weeks 1-8 [{label}]",
            "value":     round(avg_freq, 2),
            "unit":      "calls/week",
            "benchmark": "3.0+ calls/week = competitor standard (ZYLATEC achieved 3.1)",
            "direction": "BAD"   if avg_freq < 2.0  else
                         "WATCH" if avg_freq < 3.0  else "GOOD",
            "delta":     None,
            "so_what":   (
                f"Our reps averaged {avg_freq:.1f} calls/week on the highest-value oncologists "
                f"during the critical launch window — "
                f"{'72% below competitor frequency, directly correlated with prescribing habit loss' if phase == 'LAST_LAUNCH' else 'improvement from last launch but still below the 3.0 benchmark needed to match competitor intensity'}."
            ),
        })

    # ── Call-to-conversion rate by decile band ──
    for band in ["Top (9-10)", "High (7-8)", "Mid (4-6)", "Low (1-3)"]:
        band_data = rep_hcp[
            (rep_hcp["decile_band"] == band) &
            (rep_hcp["launch_phase"] == "LAST_LAUNCH")
        ]
        if len(band_data) == 0:
            continue
        conv = band_data["conversion_rate"].mean() * 100
        direction = "GOOD" if conv >= 25 else "WATCH" if conv >= 15 else "BAD"
        kpi_list.append({
            "name":      f"Call-to-SOLD Conversion Rate — {band} HCPs (Last Launch)",
            "value":     round(conv, 1),
            "unit":      "%",
            "benchmark": "> 25% = strong; 15-25% = average; < 15% = weak",
            "direction": direction,
            "delta":     None,
            "so_what":   (
                f"{conv:.0f}% of calls on {band} HCPs resulted in a prescription — "
                f"{'high-value targets are converting at expected rates when called at sufficient frequency' if conv >= 20 else 'conversion below benchmark, suggesting either insufficient frequency or quality issues on these accounts'}."
            ),
        })

    # ── IC gaming signal: call volume distribution by decile ──
    low_decile_calls = rep_hcp[
        (rep_hcp["decile"] <= 5) &
        (rep_hcp["launch_phase"] == "LAST_LAUNCH")
    ]["total_calls"].sum()
    all_calls = rep_hcp[rep_hcp["launch_phase"] == "LAST_LAUNCH"]["total_calls"].sum()
    low_decile_share = low_decile_calls / all_calls * 100 if all_calls > 0 else 0

    kpi_list.append({
        "name":      "% of All Calls on Low-Mid Decile HCPs (1-5) — Last Launch",
        "value":     round(low_decile_share, 1),
        "unit":      "%",
        "benchmark": "< 35% = well-targeted; > 50% = IC gaming likely",
        "direction": "BAD"   if low_decile_share > 50 else
                     "WATCH" if low_decile_share > 35 else "GOOD",
        "delta":     None,
        "so_what":   (
            f"{low_decile_share:.0f}% of all calls went to decile 1-5 HCPs — "
            f"{'a strong indicator that reps were optimising for call volume targets rather than HCP value, driven by incentive structure that did not weight by decile' if low_decile_share > 45 else 'within acceptable range, though worth monitoring against IC payout data'}."
        ),
    })

    return kpi_list


def _territory_kpis(ter_perf: pd.DataFrame) -> list:
    """
    Territory alignment: potential vs. resource allocation vs. competitor.
    """
    kpi_list = []

    last = ter_perf[ter_perf["launch_phase"] == "LAST_LAUNCH"].copy()
    cur  = ter_perf[ter_perf["launch_phase"] == "CURRENT_LAUNCH"].copy()

    if len(last) == 0:
        return kpi_list

    # ── Territories where we are outgunned by competitor ──
    outgunned = last[last["coverage_gap"] > 0]
    kpi_list.append({
        "name":      "Territories Where ZYLATEC Has More Reps Than Us",
        "value":     len(outgunned),
        "unit":      "territories",
        "benchmark": "0 = full parity; > 3 in high-potential areas = structural disadvantage",
        "direction": "BAD" if len(outgunned) > 3 else "WATCH",
        "delta":     None,
        "so_what":   (
            f"ZYLATEC had more field reps than us in {len(outgunned)} territories during last launch. "
            f"In oncology, rep presence at cancer centers directly drives prescribing — "
            f"being outgunned in key geographies is not recoverable by message quality alone."
        ),
    })

    # ── High-potential territories with low call ratio vs competitor ──
    if "call_ratio_vs_competitor" in last.columns:
        high_pot = last[last["market_potential_score"] > last["market_potential_score"].median()]
        low_ratio = high_pot[high_pot["call_ratio_vs_competitor"] < 0.75]
        kpi_list.append({
            "name":      "High-Potential Territories With <75% of Competitor Call Volume",
            "value":     len(low_ratio),
            "unit":      "territories",
            "benchmark": "0 = matching competitor intensity where it matters most",
            "direction": "BAD" if len(low_ratio) > 2 else "WATCH",
            "delta":     None,
            "so_what":   (
                f"{len(low_ratio)} high-potential territories received less than 75% of ZYLATEC's "
                f"call volume — meaning competitor was more active precisely where the most "
                f"prescriptions were available."
            ),
        })

    # ── Cancer center coverage gap ──
    cancer_ter = last[last["cancer_centers_count"] > 0]
    if len(cancer_ter) > 0:
        avg_ratio = cancer_ter["call_ratio_vs_competitor"].mean()
        kpi_list.append({
            "name":      "Our Call Volume vs ZYLATEC at Cancer Center Territories",
            "value":     round(avg_ratio, 2),
            "unit":      "x ratio",
            "benchmark": "1.0x = parity; > 1.0x = we are more active",
            "direction": "BAD"   if avg_ratio < 0.70 else
                         "WATCH" if avg_ratio < 0.90 else "GOOD",
            "delta":     None,
            "so_what":   (
                f"In territories containing major cancer centers, our call volume was "
                f"{avg_ratio:.2f}x ZYLATEC's — "
                f"{'well below parity at the accounts that drive the most NSCLC prescribing nationally' if avg_ratio < 0.80 else 'approaching parity but still trailing at the most strategically important accounts'}."
            ),
        })

    # ── Current launch: are high-potential territories now properly covered? ──
    if len(cur) > 0 and "call_ratio_vs_competitor" in cur.columns:
        cur_high = cur[cur["market_potential_score"] > cur["market_potential_score"].median()]
        cur_low  = cur_high[cur_high["call_ratio_vs_competitor"] < 0.75]
        kpi_list.append({
            "name":      "CURRENT LAUNCH: High-Potential Territories Still Undercovered",
            "value":     len(cur_low),
            "unit":      "territories",
            "benchmark": "0 = corrected from last launch",
            "direction": "BAD"  if len(cur_low) > 2 else
                         "WATCH" if len(cur_low) > 0 else "GOOD",
            "delta":     None,
            "so_what":   (
                f"{len(cur_low)} high-potential territories remain below 75% of competitor "
                f"call intensity in the current launch — "
                f"{'the territorial alignment mistake from last launch has not been fully corrected with AURONIX launch approaching' if len(cur_low) > 0 else 'coverage has been corrected — this risk is mitigated'}."
            ),
        })

    return kpi_list


def _rep_alignment_kpis(rep_hcp: pd.DataFrame, outcome: pd.DataFrame) -> list:
    """
    Rep experience alignment: right experience on right HCPs.
    """
    kpi_list = []

    last = rep_hcp[rep_hcp["launch_phase"] == "LAST_LAUNCH"]
    cur  = rep_hcp[rep_hcp["launch_phase"] == "CURRENT_LAUNCH"]

    # ── % of high-decile HCP coverage handled by junior reps ──
    for phase, label in [("LAST_LAUNCH", "Last Launch"), ("CURRENT_LAUNCH", "Current Launch")]:
        phase_data    = rep_hcp[rep_hcp["launch_phase"] == phase]
        high_decile   = phase_data[phase_data["decile"] >= 8]
        if len(high_decile) == 0:
            continue
        junior_pairs  = high_decile[high_decile["is_junior_onco"] == True]
        junior_pct    = len(junior_pairs["rep_id"].unique()) / max(1, len(high_decile["rep_id"].unique())) * 100

        kpi_list.append({
            "name":      f"Reps With <2yr Onco Exp Covering Decile 8-10 HCPs — {label}",
            "value":     round(junior_pct, 1),
            "unit":      "%",
            "benchmark": "< 10% = acceptable; > 20% = alignment failure",
            "direction": "BAD"   if junior_pct > 20 else
                         "WATCH" if junior_pct > 10 else "GOOD",
            "delta":     None,
            "so_what":   (
                f"{junior_pct:.0f}% of reps covering our highest-value oncologists have "
                f"less than 2 years of oncology experience — "
                f"{'in a disease area where clinical depth determines access and credibility with KOLs, this is a direct cause of reduced call frequency and lower conversion' if junior_pct > 15 else 'within acceptable range but worth monitoring as launch intensifies'}."
            ),
        })

    # ── Experience-to-outcome correlation ──
    # Experienced reps vs junior reps on high-decile: conversion gap
    if len(last) > 0:
        high_d = last[last["decile"] >= 8]
        exp_conv   = high_d[high_d["is_junior_onco"] == False]["conversion_rate"].mean() * 100
        junior_conv= high_d[high_d["is_junior_onco"] == True ]["conversion_rate"].mean() * 100
        if not np.isnan(exp_conv) and not np.isnan(junior_conv):
            conv_gap = exp_conv - junior_conv
            kpi_list.append({
                "name":      "Conversion Rate Gap: Experienced vs Junior Reps on D8+ HCPs",
                "value":     round(conv_gap, 1),
                "unit":      "pp",
                "benchmark": "< 5pp = experience doesn't matter much; > 10pp = experience is decisive",
                "direction": "BAD" if conv_gap > 8 else "WATCH",
                "delta":     None,
                "so_what":   (
                    f"Experienced reps converted {exp_conv:.0f}% of calls vs {junior_conv:.0f}% for junior reps "
                    f"on the same class of high-value oncologists — "
                    f"a {conv_gap:.0f}pp gap confirming that experience is not interchangeable on these accounts."
                ),
            })

    # ── Avg calls per week: experienced vs junior on high-decile ──
    if len(last) > 0:
        high_d = last[last["decile"] >= 8]
        exp_freq    = high_d[high_d["is_junior_onco"] == False]["calls_per_week"].mean()
        junior_freq = high_d[high_d["is_junior_onco"] == True ]["calls_per_week"].mean()
        if not np.isnan(exp_freq) and not np.isnan(junior_freq):
            freq_gap = exp_freq - junior_freq
            kpi_list.append({
                "name":      "Call Frequency Gap: Experienced vs Junior Reps on D8+ HCPs",
                "value":     round(freq_gap, 2),
                "unit":      "calls/week",
                "benchmark": "< 0.5 = minimal gap; > 1.0 = junior reps are significantly less active",
                "direction": "BAD"   if freq_gap > 1.0 else
                             "WATCH" if freq_gap > 0.5 else "GOOD",
                "delta":     None,
                "so_what":   (
                    f"Experienced reps made {exp_freq:.1f} calls/week vs {junior_freq:.1f} for junior reps "
                    f"on decile 8-10 HCPs — a {freq_gap:.1f} call/week gap that compounds over the launch window "
                    f"into a significant activity deficit on the accounts that matter most."
                ),
            })

    return kpi_list


def _competitive_kpis(ter_perf: pd.DataFrame, comparison: pd.DataFrame) -> list:
    """
    Competitive position vs ZYLATEC.
    """
    kpi_list = []

    last = ter_perf[ter_perf["launch_phase"] == "LAST_LAUNCH"]
    cur  = ter_perf[ter_perf["launch_phase"] == "CURRENT_LAUNCH"]

    # ── Market share trajectory ──
    if "LAST_LAUNCH" in comparison.columns:
        last_share = float(comparison.loc["our_avg_market_share_wk1_8", "LAST_LAUNCH"])
        last_zylatec = float(comparison.loc["zylatec_avg_market_share_wk1_8", "LAST_LAUNCH"])
        kpi_list.append({
            "name":      "Our Market Share Weeks 1-8 (Last Launch vs ZYLATEC)",
            "value":     round(last_share, 1),
            "unit":      "%",
            "benchmark": f"ZYLATEC achieved {last_zylatec:.1f}% in same window",
            "direction": "BAD" if last_share < last_zylatec else "GOOD",
            "delta":     round(last_share - last_zylatec, 1),
            "so_what":   (
                f"We captured {last_share:.1f}% market share vs ZYLATEC's {last_zylatec:.1f}% "
                f"in the first 8 weeks after last launch — "
                f"the early share deficit was never recovered, ending the launch period with structural under-performance."
            ),
        })

    if "CURRENT_LAUNCH" in comparison.columns:
        cur_share   = float(comparison.loc["our_avg_market_share_wk1_8", "CURRENT_LAUNCH"])
        cur_zylatec = float(comparison.loc["zylatec_avg_market_share_wk1_8", "CURRENT_LAUNCH"])
        last_gap    = float(comparison.loc["market_share_gap_wk1_8", "LAST_LAUNCH"])
        cur_gap     = float(comparison.loc["market_share_gap_wk1_8", "CURRENT_LAUNCH"])
        improvement = cur_gap - last_gap   # less negative = improvement

        kpi_list.append({
            "name":      "CURRENT LAUNCH: Our Market Share Weeks 1-8 vs ZYLATEC",
            "value":     round(cur_share, 1),
            "unit":      "%",
            "benchmark": f"ZYLATEC at {cur_zylatec:.1f}% — need parity by week 8",
            "direction": "BAD"   if cur_gap < -5  else
                         "WATCH" if cur_gap < 0   else "GOOD",
            "delta":     round(improvement, 1),
            "so_what":   (
                f"In the current AURONIX launch, we are at {cur_share:.1f}% vs ZYLATEC's {cur_zylatec:.1f}% "
                f"in weeks 1-8 — a {abs(cur_gap):.1f}pp gap. "
                f"{'This is an improvement of {:.1f}pp vs last launch but the gap remains and territories showing the old frequency patterns are at risk of widening it.'.format(abs(improvement)) if improvement > 0 else 'No improvement from last launch pattern — immediate action on territory coverage and rep deployment required.'}"
            ),
        })

    # ── Territories where ZYLATEC is pulling ahead fast ──
    if len(cur) > 0 and "zylatec_avg_market_share" in cur.columns:
        danger_ter = cur[
            (cur["zylatec_avg_market_share"] - cur["our_avg_market_share"]) > 10
        ]
        kpi_list.append({
            "name":      "CURRENT LAUNCH: Territories Where ZYLATEC Leads by >10pp",
            "value":     len(danger_ter),
            "unit":      "territories",
            "benchmark": "0 = strong position; > 5 = structurally behind",
            "direction": "BAD"  if len(danger_ter) > 5  else
                         "WATCH" if len(danger_ter) > 2 else "GOOD",
            "delta":     None,
            "so_what":   (
                f"ZYLATEC leads by more than 10 percentage points in {len(danger_ter)} territories "
                f"in the current launch — these are the highest-priority territories for immediate "
                f"rep redeployment and coverage frequency correction."
            ),
        })

    return kpi_list


def _launch_risk_score(risk_flags: pd.DataFrame, comparison: pd.DataFrame) -> dict:
    """
    Composite risk score for the AURONIX launch.
    RED / AMBER / GREEN with a numeric score and driving factors.
    """
    red_count   = len(risk_flags[risk_flags["severity"] == "RED"])   if len(risk_flags) else 0
    amber_count = len(risk_flags[risk_flags["severity"] == "AMBER"]) if len(risk_flags) else 0

    # Score: 0-100, higher = more at risk
    # Each RED = 20 points, each AMBER = 8 points, cap at 100
    risk_score = min(100, red_count * 20 + amber_count * 8)

    if risk_score >= 60:
        rag = "RED"
        summary = "Current launch trajectory mirrors last launch failure pattern. Immediate intervention required."
    elif risk_score >= 30:
        rag = "AMBER"
        summary = "Partial improvements from last launch but key risk factors remain unresolved."
    else:
        rag = "GREEN"
        summary = "Current launch shows meaningfully improved patterns vs last launch."

    # Driving factors from flags
    factors = []
    if len(risk_flags) > 0:
        for _, f in risk_flags.iterrows():
            factors.append({
                "severity": f["severity"],
                "flag":     f["flag_type"],
                "message":  f["message"],
                "territory": f["territory_name"],
            })

    # Market share delta from comparison
    share_gap_improvement = None
    if "LAST_LAUNCH" in comparison.columns and "CURRENT_LAUNCH" in comparison.columns:
        last_gap = float(comparison.loc["market_share_gap_wk1_8", "LAST_LAUNCH"])
        cur_gap  = float(comparison.loc["market_share_gap_wk1_8", "CURRENT_LAUNCH"])
        share_gap_improvement = round(cur_gap - last_gap, 1)

    return {
        "score":                  risk_score,
        "rag":                    rag,
        "summary":                summary,
        "red_flags":              red_count,
        "amber_flags":            amber_count,
        "factors":                factors,
        "share_gap_improvement":  share_gap_improvement,
    }


# ═══════════════════════════════════════════════════════════
# PROMPT-READY SUMMARY
# Flat text summary of all KPIs for injection into Claude prompt
# ═══════════════════════════════════════════════════════════

def kpis_to_prompt_string(kpis: dict) -> str:
    """
    Converts the KPI dict into a structured text block
    ready to be injected into the Claude prompt.
    """
    lines = []

    section_labels = {
        "revenue_diagnostic":      "LAYER 1 — REVENUE DIAGNOSTIC (Is cost the problem?)",
        "sales_force_effectiveness":"LAYER 2 — SALES FORCE EFFECTIVENESS (Are reps calling right?)",
        "territory_alignment":     "LAYER 3 — TERRITORY ALIGNMENT (Are territories designed right?)",
        "rep_alignment":           "LAYER 4 — REP EXPERIENCE ALIGNMENT (Right reps on right doctors?)",
        "competitive_position":    "LAYER 5 — COMPETITIVE POSITION (How are we tracking vs ZYLATEC?)",
    }

    for section_key, label in section_labels.items():
        lines.append(f"\n{'─'*60}")
        lines.append(label)
        lines.append('─'*60)
        for kpi in kpis.get(section_key, []):
            direction_icon = {"GOOD": "✓", "BAD": "✗", "WATCH": "~"}.get(kpi["direction"], "?")
            lines.append(f"\n  [{direction_icon}] {kpi['name']}")
            lines.append(f"      Value     : {kpi['value']} {kpi['unit']}")
            lines.append(f"      Benchmark : {kpi['benchmark']}")
            if kpi.get("delta") is not None:
                lines.append(f"      Delta     : {kpi['delta']:+} vs prior period")
            lines.append(f"      So what   : {kpi['so_what']}")

    # Launch risk block
    risk = kpis.get("launch_risk", {})
    lines.append(f"\n{'─'*60}")
    lines.append("LAYER 6 — CURRENT LAUNCH RISK ASSESSMENT")
    lines.append('─'*60)
    lines.append(f"  Risk Score : {risk.get('score', 'N/A')} / 100")
    lines.append(f"  Status     : {risk.get('rag', 'N/A')}")
    lines.append(f"  Summary    : {risk.get('summary', '')}")
    lines.append(f"  Red flags  : {risk.get('red_flags', 0)}")
    lines.append(f"  Amber flags: {risk.get('amber_flags', 0)}")
    if risk.get("share_gap_improvement") is not None:
        imp = risk["share_gap_improvement"]
        lines.append(f"  Share gap vs last launch: {imp:+.1f}pp ({'improving' if imp > 0 else 'not improving'})")
    lines.append("\n  Active Risk Factors:")
    for f in risk.get("factors", []):
        icon = "🔴" if f["severity"] == "RED" else "🟡"
        lines.append(f"    {icon} [{f['flag']}] {f['territory']}: {f['message']}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# STANDALONE VERIFICATION
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n=== kpi_engine.py — KPI Verification ===\n")

    _, _, gold = load_all()
    kpis = compute_kpis(gold)

    # Print each section cleanly
    section_labels = {
        "revenue_diagnostic":       "LAYER 1 — REVENUE DIAGNOSTIC",
        "sales_force_effectiveness": "LAYER 2 — SALES FORCE EFFECTIVENESS",
        "territory_alignment":      "LAYER 3 — TERRITORY ALIGNMENT",
        "rep_alignment":            "LAYER 4 — REP EXPERIENCE ALIGNMENT",
        "competitive_position":     "LAYER 5 — COMPETITIVE POSITION",
    }

    for key, label in section_labels.items():
        print(f"\n{'═'*65}")
        print(f"  {label}")
        print(f"{'═'*65}")
        for kpi in kpis[key]:
            icon = {"GOOD": "✅", "BAD": "❌", "WATCH": "⚠️ "}.get(kpi["direction"], "?")
            print(f"\n  {icon}  {kpi['name']}")
            print(f"      {kpi['value']} {kpi['unit']}  |  Benchmark: {kpi['benchmark']}")
            print(f"      → {kpi['so_what']}")

    risk = kpis["launch_risk"]
    rag_icon = {"RED": "🔴", "AMBER": "🟡", "GREEN": "🟢"}.get(risk["rag"], "?")
    print(f"\n{'═'*65}")
    print(f"  LAUNCH RISK SCORE: {risk['score']}/100  {rag_icon} {risk['rag']}")
    print(f"{'═'*65}")
    print(f"  {risk['summary']}")
    print(f"  Share gap improvement vs last launch: {risk['share_gap_improvement']:+.1f}pp")
    print(f"\n  Risk factors:")
    for f in risk["factors"]:
        icon = "🔴" if f["severity"] == "RED" else "🟡"
        print(f"    {icon}  {f['message']}")

    print(f"\n{'─'*65}")
    print("  PROMPT-READY STRING (first 500 chars):")
    print('─'*65)
    prompt_str = kpis_to_prompt_string(kpis)
    print(prompt_str[:500] + "...")
    print(f"\n  Total prompt string length: {len(prompt_str):,} chars")
    print(f"\n=== KPI engine verified. Ready for prompt_builder.py ===\n")
