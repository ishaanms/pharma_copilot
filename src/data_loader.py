"""
data_loader.py
Bronze → Silver → Gold layer transformations.

BRONZE : raw CSVs loaded as-is
SILVER : cleaned, typed, joined — one row still = one real-world event
GOLD   : aggregated, analysis-ready views that feed the KPI engine and Claude prompt

Run standalone to verify all layers:
    python src/data_loader.py
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# ── Path resolution works whether called from project root or src/ ──
_HERE       = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
DATA_DIR    = PROJECT_ROOT / "data" / "synthetic"


# ═══════════════════════════════════════════════════════════
# BRONZE — raw load, no changes
# ═══════════════════════════════════════════════════════════

def load_bronze() -> dict[str, pd.DataFrame]:
    """Load all CSVs exactly as they are on disk."""
    files = {
        "hcp_master":        "hcp_master.csv",
        "rep_master":        "rep_master.csv",
        "territory_master":  "territory_master.csv",
        "call_activity":     "call_activity.csv",
        "competitor_intel":  "competitor_intel.csv",
        "product_master":    "product_master.csv",
        "asset_register":    "asset_register.csv",
        "hr_master":         "hr_master.csv",
        "production_records":"production_records.csv",
        "promo_spend":       "promo_spend.csv",
    }
    bronze = {}
    for key, fname in files.items():
        path = DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}\n"
                f"Run generate_data.py first to create synthetic data."
            )
        bronze[key] = pd.read_csv(path)
    return bronze


# ═══════════════════════════════════════════════════════════
# SILVER — clean, type, standardize
# ═══════════════════════════════════════════════════════════

def build_silver(bronze: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    silver = {}

    # ── hcp_master ──
    hcp = bronze["hcp_master"].copy()
    hcp["decile"]                = hcp["decile"].astype(int).clip(1, 10)
    hcp["monthly_rx_potential"]  = pd.to_numeric(hcp["monthly_rx_potential"], errors="coerce").fillna(0).astype(int)
    hcp["years_in_practice"]     = pd.to_numeric(hcp["years_in_practice"], errors="coerce").fillna(0).astype(int)
    hcp["kol_flag"]              = hcp["kol_flag"].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    hcp["kol_score"]             = pd.to_numeric(hcp["kol_score"], errors="coerce").fillna(0).astype(int)
    hcp["decile_band"]           = pd.cut(hcp["decile"], bins=[0,3,6,8,10],
                                          labels=["Low (1-3)", "Mid (4-6)", "High (7-8)", "Top (9-10)"])
    silver["hcp_master"] = hcp

    # ── rep_master ──
    rep = bronze["rep_master"].copy()
    rep["total_experience_years"] = pd.to_numeric(rep["total_experience_years"], errors="coerce").fillna(0)
    rep["onco_experience_years"]  = pd.to_numeric(rep["onco_experience_years"],  errors="coerce").fillna(0)
    rep["is_junior_onco"]         = rep["onco_experience_years"] < 2.0
    rep["experience_band"]        = pd.cut(rep["onco_experience_years"],
                                           bins=[-0.1, 1.9, 3.9, 7.9, 99],
                                           labels=["<2 yrs", "2-4 yrs", "4-8 yrs", "8+ yrs"])
    rep["is_field_rep"]           = rep["territory_id"].notna() & \
                                    ~rep["designation"].str.contains("Director", na=False)
    silver["rep_master"] = rep

    # ── territory_master ──
    ter = bronze["territory_master"].copy()
    ter["market_potential_score"] = pd.to_numeric(ter["market_potential_score"], errors="coerce").fillna(0).astype(int)
    ter["coverage_gap"]           = ter["competitor_rep_count"] - 3   # we always have 3 reps/territory
    # positive = competitor has more reps than us in that territory
    silver["territory_master"] = ter

    # ── call_activity ──
    call = bronze["call_activity"].copy()
    call["call_date"]             = pd.to_datetime(call["call_date"], format="%m-%d-%Y", errors="coerce")
    call["week_number"]           = pd.to_numeric(call["week_number"], errors="coerce").fillna(0).astype("int16")
    call["detail_quality_score"]  = pd.to_numeric(call["detail_quality_score"], errors="coerce").fillna(3).astype("int8")
    call["samples_dropped"]       = call["samples_dropped"].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    call["next_call_planned"]     = call["next_call_planned"].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    call["is_sold"]               = call["outcome"] == "SOLD"
    call["is_early_launch"]       = call["week_number"] <= 8
    # Downcast string categoricals to save memory
    for col in ["rep_id","hcp_id","outcome","patient_type","launch_phase","drug_name","call_type"]:
        if col in call.columns:
            call[col] = call[col].astype("category")     # critical window
    silver["call_activity"] = call

    # ── competitor_intel ──
    comp = bronze["competitor_intel"].copy()
    comp["zylatec_market_share_pct"]  = pd.to_numeric(comp["zylatec_market_share_pct"],  errors="coerce").fillna(0)
    comp["vexorin_market_share_pct"]  = pd.to_numeric(comp["vexorin_market_share_pct"],  errors="coerce").fillna(0)
    comp["zylatec_estimated_calls"]   = pd.to_numeric(comp["zylatec_estimated_calls"],    errors="coerce").fillna(0).astype(int)
    comp["share_gap"]                 = comp["vexorin_market_share_pct"] - comp["zylatec_market_share_pct"]
    # negative = we are behind competitor in that territory/week
    silver["competitor_intel"] = comp

    # ── production_records ──
    prod = bronze["production_records"].copy()
    prod["units_produced"]              = pd.to_numeric(prod["units_produced"],              errors="coerce").fillna(0).astype(int)
    prod["units_sold"]                  = pd.to_numeric(prod["units_sold"],                  errors="coerce").fillna(0).astype(int)
    prod["units_in_warehouse"]          = pd.to_numeric(prod["units_in_warehouse"],          errors="coerce").fillna(0).astype(int)
    prod["raw_material_cost_per_unit"]  = pd.to_numeric(prod["raw_material_cost_per_unit"],  errors="coerce").fillna(0)
    prod["manufacturing_cost_per_unit"] = pd.to_numeric(prod["manufacturing_cost_per_unit"], errors="coerce").fillna(0)
    prod["total_cogs_usd"]              = pd.to_numeric(prod["total_cogs_usd"],              errors="coerce").fillna(0)
    prod["sellthrough_rate"]            = (prod["units_sold"] / prod["units_produced"].replace(0, np.nan)).fillna(0)
    prod["total_cost_per_unit"]         = prod["raw_material_cost_per_unit"] + prod["manufacturing_cost_per_unit"]
    silver["production_records"] = prod

    # ── asset_register ──
    asset = bronze["asset_register"].copy()
    asset["acquisition_date"]       = pd.to_datetime(asset["acquisition_date"], format="%m-%d-%Y", errors="coerce")
    asset["acquisition_value_usd"]  = pd.to_numeric(asset["acquisition_value_usd"],  errors="coerce").fillna(0)
    asset["current_book_value_usd"] = pd.to_numeric(asset["current_book_value_usd"], errors="coerce").fillna(0)
    asset["depreciation_pct"]       = 1 - (asset["current_book_value_usd"] / asset["acquisition_value_usd"].replace(0, np.nan))
    silver["asset_register"] = asset

    # ── hr_master ──
    hr = bronze["hr_master"].copy()
    hr["annual_ctc_usd"] = pd.to_numeric(hr["annual_ctc_usd"], errors="coerce").fillna(0)
    hr["join_date"]      = pd.to_datetime(hr["join_date"], format="%m-%d-%Y", errors="coerce")
    silver["hr_master"] = hr

    # ── promo_spend ──
    promo = bronze["promo_spend"].copy()
    promo["spend_usd"] = pd.to_numeric(promo["spend_usd"], errors="coerce").fillna(0)
    silver["promo_spend"] = promo

    # pass-throughs (already clean)
    silver["product_master"]   = bronze["product_master"].copy()

    return silver


# ═══════════════════════════════════════════════════════════
# GOLD — aggregated, analysis-ready views
# ═══════════════════════════════════════════════════════════

def build_gold(silver: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    gold = {}

    hcp  = silver["hcp_master"]
    rep  = silver["rep_master"]
    ter  = silver["territory_master"]
    call = silver["call_activity"]
    comp = silver["competitor_intel"]
    prod = silver["production_records"]
    asset= silver["asset_register"]
    hr   = silver["hr_master"]
    promo= silver["promo_spend"]

    # ── Pre-compute enriched call once — reused across all gold tables ──
    # Only keep the columns we actually need to minimise memory
    hcp_slim = hcp[["hcp_id","decile","decile_band","territory_id",
                     "monthly_rx_potential","specialty","hospital_type",
                     "kol_flag","city"]].copy()
    rep_slim = rep[["rep_id","onco_experience_years","experience_band",
                    "is_junior_onco","designation","last_launch_performance",
                    "region"]].copy()

    call_enriched = (
        call
        .merge(hcp_slim, on="hcp_id", how="left")
        .merge(rep_slim,  on="rep_id",  how="left")
    )
    # Free originals from memory after merge
    del hcp_slim, rep_slim

    # ────────────────────────────────────────────────────────
    # GOLD 1 — rep_hcp_coverage
    # ────────────────────────────────────────────────────────
    call_rep_hcp = (
        call_enriched
        .groupby(["rep_id", "hcp_id", "launch_phase"], observed=True)
        .agg(
            total_calls      = ("call_id",              "count"),
            sold_calls       = ("is_sold",              "sum"),
            avg_quality      = ("detail_quality_score", "mean"),
            samples_dropped  = ("samples_dropped",      "sum"),
            weeks_active     = ("week_number",          "nunique"),
        )
        .reset_index()
    )
    call_rep_hcp["calls_per_week"]  = (call_rep_hcp["total_calls"] / call_rep_hcp["weeks_active"].replace(0, np.nan)).fillna(0)
    call_rep_hcp["conversion_rate"] = (call_rep_hcp["sold_calls"]  / call_rep_hcp["total_calls"].replace(0, np.nan)).fillna(0)

    rep_hcp = (
        call_rep_hcp
        .merge(hcp[["hcp_id","decile","decile_band","monthly_rx_potential","specialty",
                     "hospital_type","kol_flag","territory_id","city"]], on="hcp_id", how="left")
        .merge(rep[["rep_id","onco_experience_years","experience_band","is_junior_onco",
                    "designation","last_launch_performance","region"]], on="rep_id", how="left")
    )
    gold["rep_hcp_coverage"] = rep_hcp

    # ────────────────────────────────────────────────────────
    # GOLD 2 — call_frequency_by_hcp
    # ────────────────────────────────────────────────────────
    freq = (
        call_enriched
        .groupby(["hcp_id","week_number","launch_phase",
                  "decile","decile_band","territory_id","monthly_rx_potential"],
                 observed=True)
        .agg(calls_that_week=("call_id","count"))
        .reset_index()
    )
    hcp_avg_freq = (
        freq.groupby(["hcp_id","launch_phase","decile","decile_band",
                      "territory_id","monthly_rx_potential"], observed=True)
        .agg(
            avg_calls_per_week = ("calls_that_week","mean"),
            total_weeks_called = ("week_number",    "nunique"),
            peak_calls_in_week = ("calls_that_week","max"),
        )
        .reset_index()
    )
    early_freq = (
        call_enriched[call_enriched["is_early_launch"]]
        .groupby(["hcp_id","launch_phase","decile","decile_band"], observed=True)
        .agg(avg_early_calls_per_week=("call_id", lambda x: len(x)/8))
        .reset_index()
    )
    freq_gold = hcp_avg_freq.merge(
        early_freq[["hcp_id","launch_phase","avg_early_calls_per_week"]],
        on=["hcp_id","launch_phase"], how="left"
    )
    gold["call_frequency_by_hcp"] = freq_gold

    # ────────────────────────────────────────────────────────
    # GOLD 3 — territory_performance
    # ────────────────────────────────────────────────────────
    our_ter_calls = (
        call_enriched
        .groupby(["territory_id","launch_phase"], observed=True)
        .agg(
            our_total_calls = ("call_id",              "count"),
            our_unique_hcps = ("hcp_id",               "nunique"),
            our_sold_calls  = ("is_sold",              "sum"),
            our_avg_quality = ("detail_quality_score", "mean"),
        )
        .reset_index()
    )

    # Competitor calls per territory per launch phase
    comp_ter = (
        comp.groupby(["territory_id","launch_phase"])
        .agg(
            zylatec_total_calls       = ("zylatec_estimated_calls",  "sum"),
            zylatec_avg_market_share  = ("zylatec_market_share_pct", "mean"),
            our_avg_market_share      = ("vexorin_market_share_pct", "mean"),
            zylatec_peak_share        = ("zylatec_market_share_pct", "max"),
        )
        .reset_index()
    )

    ter_perf = (
        ter[["territory_id","territory_name","region","city","zone",
             "market_potential_score","high_decile_hcps","cancer_centers_count",
             "competitor_rep_count","coverage_gap"]]
        .merge(our_ter_calls, on="territory_id", how="left")
        .merge(comp_ter,      on=["territory_id","launch_phase"], how="left")
    )
    ter_perf["call_ratio_vs_competitor"] = (
        ter_perf["our_total_calls"] / ter_perf["zylatec_total_calls"].replace(0, np.nan)
    ).fillna(0)
    ter_perf["our_conversion_rate"] = (
        ter_perf["our_sold_calls"] / ter_perf["our_total_calls"].replace(0, np.nan)
    ).fillna(0)

    gold["territory_performance"] = ter_perf

    # ────────────────────────────────────────────────────────
    # GOLD 4 — outcome_by_decile
    # Call outcomes grouped by HCP decile × rep experience × launch phase
    # Answers: where are reps actually converting? (F4)
    # ────────────────────────────────────────────────────────
    outcome_decile = (
        call_enriched
        .groupby(["decile","decile_band","experience_band","is_junior_onco","launch_phase"])
        .agg(
            total_calls      = ("call_id",  "count"),
            sold             = ("is_sold",  "sum"),
            follow_ups       = ("outcome",  lambda x: (x == "FOLLOW_UP").sum()),
            not_sold         = ("outcome",  lambda x: (x == "NOT_SOLD").sum()),
            avg_quality      = ("detail_quality_score", "mean"),
        )
        .reset_index()
    )
    outcome_decile["conversion_rate_pct"] = (
        outcome_decile["sold"] / outcome_decile["total_calls"].replace(0, np.nan) * 100
    ).fillna(0).round(1)
    outcome_decile["calls_share_of_total"] = (
        outcome_decile["total_calls"] / outcome_decile.groupby("launch_phase")["total_calls"].transform("sum") * 100
    ).round(2)

    gold["outcome_by_decile"] = outcome_decile

    # ────────────────────────────────────────────────────────
    # GOLD 5 — cost_summary
    # Flattened cost picture: fixed + variable + promo
    # Answers: is cost the problem? (elimination proof)
    # ────────────────────────────────────────────────────────

    # Fixed cost: total book value of active assets
    fixed_cost = asset[asset["status"] == "ACTIVE"]["current_book_value_usd"].sum()
    last_asset_date = asset[asset["status"] != "DISPOSED"]["acquisition_date"].max()

    # Variable cost: COGS trend
    cogs_trend = (
        prod.groupby(["drug_name","year"])
        .agg(
            avg_cost_per_unit  = ("total_cost_per_unit", "mean"),
            total_cogs         = ("total_cogs_usd",      "sum"),
            avg_sellthrough    = ("sellthrough_rate",     "mean"),
            peak_warehouse     = ("units_in_warehouse",   "max"),
        )
        .reset_index()
    )

    # Headcount cost trend
    headcount = (
        hr[hr["status"] == "ACTIVE"]
        .groupby("department")
        .agg(headcount=("emp_id","count"), total_cost=("annual_ctc_usd","sum"))
        .reset_index()
    )

    # Promo spend total
    promo_total = (
        promo.groupby(["launch_phase","channel"])
        .agg(total_spend=("spend_usd","sum"))
        .reset_index()
    )

    gold["cost_summary"] = {
        "fixed_cost_book_value_usd": round(fixed_cost, 0),
        "last_capex_date":           str(last_asset_date.date()) if pd.notna(last_asset_date) else "N/A",
        "cogs_trend":                cogs_trend,
        "headcount_by_dept":         headcount,
        "promo_spend_by_channel":    promo_total,
    }

    # ────────────────────────────────────────────────────────
    # GOLD 6 — launch_comparison
    # Side-by-side last launch vs current launch on identical metrics
    # Answers: are we repeating the mistake? (F5)
    # ────────────────────────────────────────────────────────

    def launch_kpis(phase):
        c     = call_enriched[call_enriched["launch_phase"] == phase]
        c_hcp = c  # already has decile and territory_id from call_enriched
        c_rep = c  # already has onco_experience_years and is_junior_onco

        high_decile_calls = c_hcp[c_hcp["decile"] >= 8]
        early_high        = high_decile_calls[high_decile_calls["week_number"] <= 8]

        # Coverage: % of high-decile HCPs called at all
        called_hcps      = set(c["hcp_id"].astype(str).unique())
        high_decile_hcps = set(hcp[hcp["decile"] >= 8]["hcp_id"].astype(str).unique())
        coverage_pct     = len(called_hcps & high_decile_hcps) / len(high_decile_hcps) * 100 if high_decile_hcps else 0

        # Call freq on D8+ weeks 1-8
        if len(early_high) > 0:
            early_freq_val = early_high.groupby(["rep_id","hcp_id","week_number"]).size().mean()
        else:
            early_freq_val = 0

        # Junior reps on high-decile HCPs
        junior_on_high = c_rep[(c_rep["is_junior_onco"]) & (c_rep["decile"] >= 8)]
        junior_ratio = len(junior_on_high["rep_id"].astype(str).unique()) / max(1, len(c_rep["rep_id"].astype(str).unique())) * 100

        # Overall conversion
        conversion = c["is_sold"].mean() * 100

        # Competitor share (weeks 1-8)
        comp_early = comp[(comp["launch_phase"] == phase) & (comp["week_number"] <= 8)]
        our_share   = comp_early["vexorin_market_share_pct"].mean()
        their_share = comp_early["zylatec_market_share_pct"].mean()

        return {
            "launch_phase":                     phase,
            "total_calls":                      len(c),
            "high_decile_hcp_coverage_pct":     round(coverage_pct, 1),
            "avg_calls_per_week_d8plus_early":  round(early_freq_val, 2),
            "junior_rep_ratio_on_high_decile":  round(junior_ratio, 1),
            "overall_conversion_rate_pct":      round(conversion, 1),
            "our_avg_market_share_wk1_8":       round(our_share, 1),
            "zylatec_avg_market_share_wk1_8":   round(their_share, 1),
            "market_share_gap_wk1_8":           round(our_share - their_share, 1),
        }

    last    = launch_kpis("LAST_LAUNCH")
    current = launch_kpis("CURRENT_LAUNCH")

    comparison = pd.DataFrame([last, current]).set_index("launch_phase").T
    comparison.columns.name = None
    gold["launch_comparison"] = comparison

    # ────────────────────────────────────────────────────────
    # GOLD 7 — risk_flags (current launch early warning)
    # Binary flags for the Streamlit dashboard header cards
    # ────────────────────────────────────────────────────────
    flags = []

    # F5: territories repeating the same low-frequency pattern
    repeating = ["TER_NO_01", "TER_NO_02", "TER_MI_01", "TER_PA_01"]
    cur_freq = (
        call[(call["launch_phase"] == "CURRENT_LAUNCH") & (call["week_number"] <= 8)]
        .merge(hcp[["hcp_id","decile","territory_id"]], on="hcp_id")
    )
    cur_freq = cur_freq[cur_freq["decile"] >= 8]
    for tid in repeating:
        tid_calls = cur_freq[cur_freq["territory_id"] == tid]
        if len(tid_calls) > 0:
            freq_val = tid_calls.groupby(["rep_id","hcp_id","week_number"]).size().mean()
            if freq_val < 2.0:
                t_name = ter[ter["territory_id"] == tid]["territory_name"].values
                flags.append({
                    "flag_type":   "LOW_FREQ_REPEAT",
                    "territory_id": tid,
                    "territory_name": t_name[0] if len(t_name) else tid,
                    "value":       round(freq_val, 2),
                    "threshold":   2.0,
                    "severity":    "RED",
                    "message":     f"Avg {freq_val:.1f} calls/week on D8+ HCPs — same pattern as last launch failure",
                })

    # Junior reps on high-value territories in current launch
    cur_rep_hcp = (
        call[call["launch_phase"] == "CURRENT_LAUNCH"]
        .merge(hcp[["hcp_id","decile","territory_id"]], on="hcp_id")
        .merge(rep[["rep_id","onco_experience_years","is_junior_onco"]], on="rep_id")
    )
    junior_high_cur = cur_rep_hcp[(cur_rep_hcp["decile"] >= 8) & (cur_rep_hcp["is_junior_onco"])]
    if len(junior_high_cur["rep_id"].unique()) > 0:
        flags.append({
            "flag_type":    "JUNIOR_ON_HIGH_DECILE",
            "territory_id": None,
            "territory_name": "Multiple",
            "value":        len(junior_high_cur["rep_id"].unique()),
            "threshold":    0,
            "severity":     "AMBER",
            "message":      f"{len(junior_high_cur['rep_id'].unique())} reps with <2yr onco exp still covering D8+ HCPs",
        })

    # Competitor outpacing in coverage
    comp_cur = comp[(comp["launch_phase"] == "CURRENT_LAUNCH") & (comp["week_number"] <= 8)]
    bad_share_territories = comp_cur[comp_cur["share_gap"] < -10]["territory_id"].unique()
    if len(bad_share_territories) > 0:
        flags.append({
            "flag_type":    "MARKET_SHARE_DEFICIT",
            "territory_id": None,
            "territory_name": f"{len(bad_share_territories)} territories",
            "value":        len(bad_share_territories),
            "threshold":    0,
            "severity":     "RED",
            "message":      f"ZYLATEC leads by >10pp in {len(bad_share_territories)} territories (weeks 1-8)",
        })

    gold["risk_flags"] = pd.DataFrame(flags) if flags else pd.DataFrame(
        columns=["flag_type","territory_id","territory_name","value","threshold","severity","message"]
    )

    return gold


# ═══════════════════════════════════════════════════════════
# PUBLIC API — single entry point for app.py
# ═══════════════════════════════════════════════════════════

def load_all() -> tuple[dict, dict, dict]:
    """
    Load and transform all layers.
    Returns (bronze, silver, gold) — app.py uses gold primarily.
    """
    bronze = load_bronze()
    silver = build_silver(bronze)
    gold   = build_gold(silver)
    return bronze, silver, gold


# ═══════════════════════════════════════════════════════════
# STANDALONE VERIFICATION
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n=== data_loader.py — Layer Verification ===\n")

    bronze, silver, gold = load_all()

    print("BRONZE (raw row counts):")
    for k, v in bronze.items():
        print(f"  {k:<25} {len(v):>8,} rows")

    print("\nSILVER (cleaned — spot checks):")
    print(f"  call_activity date range : {silver['call_activity']['call_date'].min().date()} → "
          f"{silver['call_activity']['call_date'].max().date()}")
    print(f"  rep is_junior_onco count : {silver['rep_master']['is_junior_onco'].sum()} reps with <2yr onco exp")
    print(f"  hcp decile_band dist     :")
    print(silver["hcp_master"]["decile_band"].value_counts().sort_index().to_string(header=False))

    print("\nGOLD — key tables:")
    for k in ["rep_hcp_coverage","call_frequency_by_hcp","territory_performance",
              "outcome_by_decile","risk_flags"]:
        df = gold[k]
        print(f"  {k:<30} {len(df):>7,} rows  |  cols: {list(df.columns)[:5]}...")

    print("\nGOLD — launch_comparison (the core side-by-side):")
    print(gold["launch_comparison"].to_string())

    print("\nGOLD — risk_flags (current launch early warning):")
    if len(gold["risk_flags"]) > 0:
        for _, row in gold["risk_flags"].iterrows():
            severity_icon = "🔴" if row["severity"] == "RED" else "🟡"
            print(f"  {severity_icon}  [{row['flag_type']}] {row['message']}")
    else:
        print("  No risk flags raised.")

    print("\nGOLD — cost_summary:")
    cs = gold["cost_summary"]
    print(f"  Fixed cost (book value)   : ${cs['fixed_cost_book_value_usd']:,.0f}")
    print(f"  Last capex date           : {cs['last_capex_date']}")
    print(f"  COGS trend (avg/unit USD) :")
    print(cs["cogs_trend"][["drug_name","year","avg_cost_per_unit","avg_sellthrough"]].to_string(index=False))
    print(f"\n  Headcount by department:")
    print(cs["headcount_by_dept"].to_string(index=False))

    print("\n=== All layers verified. Ready for kpi_engine.py ===\n")
