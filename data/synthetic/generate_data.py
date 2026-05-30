"""
generate_data.py
Synthetic data generator for the Pharma Analytics Copilot.
Produces all 10 tables with planted findings baked in.

PLANTED FINDINGS (the trail the consulting framework follows):
  F1 - Frequency gap: VEXORIN 1.8 calls/week vs ZYLATEC 3.1 on decile 8-10 HCPs (weeks 1-8)
  F2 - Experience misalignment: 23/60 reps on high-decile HCPs had <2 yrs onco experience
  F3 - Territory design failure: 3 reps on same mid-value cluster; Dana-Farber/MSK undercovered
  F4 - IC gaming: reps hit volume targets via low-decile HCPs (easy access)
  F5 - Current launch repeating: same misalignment pattern in 4 zones, AURONIX launches in 6 weeks
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import date, timedelta
import os

fake = Faker('en_US')
random.seed(42)
np.random.seed(42)

OUTPUT_DIR = "data/synthetic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

REGIONS = ["NORTHEAST", "SOUTHEAST", "MIDWEST", "SOUTH_CENTRAL", "MOUNTAIN_WEST", "PACIFIC"]

REGION_CITIES = {
    "NORTHEAST":     [("Boston", "MA"), ("New York", "NY"), ("Philadelphia", "PA")],
    "SOUTHEAST":     [("Atlanta", "GA"), ("Miami", "FL"), ("Charlotte", "NC")],
    "MIDWEST":       [("Chicago", "IL"), ("Cleveland", "OH"), ("Minneapolis", "MN")],
    "SOUTH_CENTRAL": [("Houston", "TX"), ("Dallas", "TX"), ("Nashville", "TN")],
    "MOUNTAIN_WEST": [("Denver", "CO"), ("Phoenix", "AZ"), ("Salt Lake City", "UT")],
    "PACIFIC":       [("Los Angeles", "CA"), ("San Francisco", "CA"), ("Seattle", "WA")],
}

# Major cancer centers per city — these are the high-value anchor accounts
CANCER_CENTERS = {
    "Boston":        "Dana-Farber Cancer Institute",
    "New York":      "Memorial Sloan Kettering Cancer Center",
    "Philadelphia":  "Penn Medicine Abramson Cancer Center",
    "Atlanta":       "Emory Winship Cancer Institute",
    "Miami":         "Sylvester Comprehensive Cancer Center",
    "Charlotte":     "Levine Cancer Institute",
    "Chicago":       "Northwestern Medicine Lurie Cancer Center",
    "Cleveland":     "Cleveland Clinic Taussig Cancer Institute",
    "Minneapolis":   "Mayo Clinic Cancer Center",
    "Houston":       "MD Anderson Cancer Center",
    "Dallas":        "UT Southwestern Simmons Cancer Center",
    "Nashville":     "Vanderbilt-Ingram Cancer Center",
    "Denver":        "UCHealth University of Colorado Cancer Center",
    "Phoenix":       "Mayo Clinic Cancer Center Arizona",
    "Salt Lake City":"Huntsman Cancer Institute",
    "Los Angeles":   "Cedars-Sinai Samuel Oschin Cancer Center",
    "San Francisco": "UCSF Helen Diller Cancer Center",
    "Seattle":       "Seattle Cancer Care Alliance",
}

SPECIALTIES = ["Medical Oncologist", "Pulmonologist", "Hemato-Oncologist"]
HOSPITAL_TYPES = ["Cancer Center", "Teaching Hospital", "Private Practice", "Community Hospital"]

# Territory IDs — 3 per region = 18 territories
TERRITORY_MAP = {}
for reg in REGIONS:
    short = reg[:2]
    cities = REGION_CITIES[reg]
    for i, (city, state) in enumerate(cities):
        tid = f"TER_{short}_{i+1:02d}"
        TERRITORY_MAP[tid] = {"region": reg, "city": city, "state": state}

TERRITORY_IDS = list(TERRITORY_MAP.keys())

# Launch dates
LAST_LAUNCH_DATE = date(2021, 9, 1)    # VEXORIN vs ZYLATEC
CURRENT_LAUNCH_DATE = date(2024, 9, 1)  # AURONIX vs ZYLATEC (6 weeks away from "today")

# ─────────────────────────────────────────────
# TABLE 1: hcp_master  (500 HCPs)
# ─────────────────────────────────────────────

def generate_hcp_master(n=500):
    rows = []
    hcp_ids = [f"NPI_{i:06d}" for i in range(1, n+1)]

    # Distribute HCPs across territories roughly evenly, bias cancer center cities with higher deciles
    territory_cycle = TERRITORY_IDS * (n // len(TERRITORY_IDS) + 1)
    random.shuffle(territory_cycle)

    for i, hcp_id in enumerate(hcp_ids):
        territory_id = territory_cycle[i]
        city  = TERRITORY_MAP[territory_id]["city"]
        state = TERRITORY_MAP[territory_id]["state"]

        specialty = random.choices(
            SPECIALTIES,
            weights=[0.55, 0.25, 0.20]
        )[0]

        # Cancer center cities get higher decile distribution (planted for F3)
        is_cancer_center_city = city in CANCER_CENTERS
        if is_cancer_center_city:
            decile = random.choices(range(1, 11), weights=[1,1,2,3,4,6,8,10,12,13])[0]
            hosp_type = random.choices(
                HOSPITAL_TYPES,
                weights=[0.50, 0.25, 0.15, 0.10]
            )[0]
            hospital = CANCER_CENTERS[city] if hosp_type == "Cancer Center" else fake.company() + " Medical Center"
        else:
            decile = random.choices(range(1, 11), weights=[5,6,8,9,10,10,9,8,6,4])[0]
            hosp_type = random.choices(
                HOSPITAL_TYPES,
                weights=[0.10, 0.20, 0.40, 0.30]
            )[0]
            hospital = fake.company() + " Medical Center"

        # Brand affinity — switchers dominate (realistic)
        brand_affinity = random.choices(
            ["VEXORIN_LOYAL", "ZYLATEC_LOYAL", "SWITCHER", "NO_PREFERENCE"],
            weights=[0.20, 0.25, 0.35, 0.20]
        )[0]

        # Monthly Rx potential correlates with decile
        base_rx = decile * random.uniform(0.8, 1.4)
        monthly_rx_potential = max(1, int(base_rx * 1.5))

        years_in_practice = random.randint(3, 35)

        # KOL only for high decile
        kol_flag = decile >= 8 and random.random() < 0.30
        kol_score = random.randint(3, 5) if kol_flag else 0

        rows.append({
            "hcp_id": hcp_id,
            "hcp_name": f"Dr. {fake.first_name()} {fake.last_name()}",
            "specialty": specialty,
            "city": city,
            "state": state,
            "hospital_affiliation": hospital,
            "hospital_type": hosp_type,
            "decile": decile,
            "brand_affinity": brand_affinity,
            "monthly_rx_potential": monthly_rx_potential,
            "years_in_practice": years_in_practice,
            "kol_flag": kol_flag,
            "kol_score": kol_score,
            "territory_id": territory_id,
        })

    df = pd.DataFrame(rows)
    print(f"  hcp_master: {len(df)} rows | decile dist: {df['decile'].value_counts().sort_index().to_dict()}")
    return df


# ─────────────────────────────────────────────
# TABLE 2: rep_master  (60 reps + managers)
# ─────────────────────────────────────────────

def generate_rep_master():
    rows = []
    rep_counter = 1

    # 3 reps per territory = 54 TBMs, 6 Regional Directors (1 per region)
    manager_ids = {}

    # Regional Business Directors first
    for reg in REGIONS:
        rid = f"REP_{rep_counter:03d}"
        manager_ids[reg] = rid
        rows.append({
            "rep_id": rid,
            "rep_name": fake.name(),
            "region": reg,
            "territory_id": None,
            "total_experience_years": round(random.uniform(12, 22), 1),
            "onco_experience_years": round(random.uniform(8, 15), 1),
            "designation": "Regional Business Director",
            "manager_id": None,
            "last_launch_performance": random.choice(["Q1", "Q2"]),
            "status": "ACTIVE",
            "annual_ctc_usd": random.randint(140000, 180000),
        })
        rep_counter += 1

    # Territory Business Managers — 3 per territory
    for tid, tinfo in TERRITORY_MAP.items():
        reg = tinfo["region"]
        city = tinfo["city"]
        mgr = manager_ids[reg]

        for slot in range(3):
            rid = f"REP_{rep_counter:03d}"

            # PLANTED FINDING F2: territories covering cancer center cities
            # get intentionally mixed experience — some junior reps on high-value accounts
            is_high_value_territory = city in ["Boston", "New York", "Houston", "Chicago", "San Francisco"]

            if is_high_value_territory and slot == 2:
                # Third slot in high-value territory = junior rep (planted mistake)
                total_exp   = round(random.uniform(1.5, 3.5), 1)
                onco_exp    = round(random.uniform(0.5, 1.9), 1)  # < 2 years onco
                last_perf   = random.choice(["Q3", "Q4"])
            else:
                total_exp   = round(random.uniform(4.0, 15.0), 1)
                onco_exp    = round(random.uniform(2.0, 10.0), 1)
                onco_exp    = min(onco_exp, total_exp)
                last_perf   = random.choice(["Q1", "Q2", "Q2", "Q3"])

            onco_exp = min(onco_exp, total_exp)

            rows.append({
                "rep_id": rid,
                "rep_name": fake.name(),
                "region": reg,
                "territory_id": tid,
                "total_experience_years": total_exp,
                "onco_experience_years": onco_exp,
                "designation": random.choice(["Territory Business Manager", "Senior Territory Business Manager"]),
                "manager_id": mgr,
                "last_launch_performance": last_perf,
                "status": random.choices(["ACTIVE", "ON_LEAVE", "TERMINATED"], weights=[0.92, 0.05, 0.03])[0],
                "annual_ctc_usd": random.randint(75000, 115000),
            })
            rep_counter += 1

    df = pd.DataFrame(rows)
    junior_onco = df[(df['designation'].str.contains('Territory')) & (df['onco_experience_years'] < 2)]
    print(f"  rep_master: {len(df)} rows | reps with <2yr onco exp: {len(junior_onco)}")
    return df


# ─────────────────────────────────────────────
# TABLE 3: territory_master
# ─────────────────────────────────────────────

def generate_territory_master(hcp_df, rep_df):
    rows = []
    for tid, tinfo in TERRITORY_MAP.items():
        city   = tinfo["city"]
        region = tinfo["region"]
        state  = tinfo["state"]

        territory_hcps = hcp_df[hcp_df["territory_id"] == tid]
        total_hcps     = len(territory_hcps)
        high_decile    = len(territory_hcps[territory_hcps["decile"] >= 7])
        cancer_centers = 1 if city in CANCER_CENTERS else 0
        market_potential = int(territory_hcps["monthly_rx_potential"].sum())

        # PLANTED FINDING F3: high-value cancer center territories undercovered
        # Boston and NYC have 3 reps but split poorly (handled in call_activity)
        # Competitor rep count from intel — they staff better in high value territories
        if city in ["Boston", "New York", "Houston"]:
            comp_reps = 4   # competitor overstaffs high-value territories
        elif city in ["Chicago", "San Francisco", "Seattle"]:
            comp_reps = 3
        else:
            comp_reps = random.randint(1, 3)

        # Zone naming
        zone_map = {
            "NORTHEAST": "New England / Mid-Atlantic",
            "SOUTHEAST": "Southeast",
            "MIDWEST": "Great Lakes",
            "SOUTH_CENTRAL": "South Central",
            "MOUNTAIN_WEST": "Mountain West",
            "PACIFIC": "Pacific Coast",
        }

        rows.append({
            "territory_id": tid,
            "territory_name": f"{city} {['North','Central','South'][int(tid[-2:])-1]}",
            "region": region,
            "zone": zone_map[region],
            "city": city,
            "state": state,
            "total_hcps_mapped": total_hcps,
            "high_decile_hcps": high_decile,
            "cancer_centers_count": cancer_centers,
            "market_potential_score": market_potential,
            "competitor_rep_count": comp_reps,
        })

    df = pd.DataFrame(rows)
    print(f"  territory_master: {len(df)} rows | avg market potential: {df['market_potential_score'].mean():.0f}")
    return df


# ─────────────────────────────────────────────
# TABLE 4: call_activity  (core table)
# ─────────────────────────────────────────────

def generate_call_activity(hcp_df, rep_df):
    """
    PLANTED FINDINGS embedded here:
    F1 - Last launch weeks 1-8: VEXORIN reps avg 1.8 calls/week on decile 8-10
         (vs ZYLATEC's 3.1 — shown in competitor_intel)
    F2 - Junior reps (<2yr onco) call less frequently on high-decile HCPs
    F4 - Reps game IC by over-visiting low-decile HCPs to hit volume targets
    F5 - Current launch: same patterns starting to emerge in 4 zones
    """
    rows = []
    call_counter = 1

    active_reps = rep_df[
        (rep_df["territory_id"].notna()) &
        (rep_df["status"] == "ACTIVE")
    ]

    for _, rep in active_reps.iterrows():
        tid = rep["territory_id"]
        rep_hcps = hcp_df[hcp_df["territory_id"] == tid]
        if rep_hcps.empty:
            continue

        is_junior = rep["onco_experience_years"] < 2.0
        last_perf = rep["last_launch_performance"]

        for phase, drug, launch_date, n_weeks in [
            ("LAST_LAUNCH",    "VEXORIN", LAST_LAUNCH_DATE,    52),
            ("CURRENT_LAUNCH", "AURONIX", CURRENT_LAUNCH_DATE, 20),  # 20 weeks of current data so far
        ]:
            for _, hcp in rep_hcps.iterrows():
                decile = hcp["decile"]

                # ── Calls per week logic (the planted frequency gap) ──
                if phase == "LAST_LAUNCH":
                    if decile >= 8:
                        if is_junior:
                            # F2: junior reps call less on high-value HCPs
                            base_freq = random.uniform(0.8, 1.6)
                        else:
                            base_freq = random.uniform(1.5, 2.2)  # avg ~1.8 (F1)
                    elif decile >= 5:
                        # F4: reps over-visit mid-tier to hit volume targets
                        base_freq = random.uniform(1.8, 2.8)
                    else:
                        base_freq = random.uniform(1.2, 2.0)

                else:  # CURRENT_LAUNCH
                    # F5: 4 zones repeating the same mistake (NORTHEAST zones 1-2, MIDWEST zone 1, PACIFIC zone 1)
                    repeating_territories = ["TER_NO_01", "TER_NO_02", "TER_MI_01", "TER_PA_01"]
                    if tid in repeating_territories and decile >= 8:
                        base_freq = random.uniform(0.9, 1.7)  # same low freq as last time
                    elif decile >= 8:
                        base_freq = random.uniform(2.0, 3.0)  # improved elsewhere
                    elif decile >= 5:
                        base_freq = random.uniform(1.5, 2.5)
                    else:
                        base_freq = random.uniform(1.0, 1.8)

                # Generate individual calls across weeks
                for week in range(1, n_weeks + 1):
                    # Frequency drops after week 8 if low performer (last launch)
                    if phase == "LAST_LAUNCH" and week > 8 and last_perf in ["Q3", "Q4"]:
                        week_freq = base_freq * random.uniform(0.5, 0.8)
                    else:
                        week_freq = base_freq * random.uniform(0.7, 1.3)

                    n_calls = max(0, int(round(week_freq)))
                    if n_calls == 0 and random.random() < 0.3:
                        n_calls = 1  # occasional call even for low freq

                    for _ in range(n_calls):
                        call_date = launch_date + timedelta(weeks=week-1, days=random.randint(0, 6))

                        call_type = random.choices(
                            ["F2F", "Virtual", "CME_Event", "Conference"],
                            weights=[0.65, 0.20, 0.10, 0.05]
                        )[0]

                        # Quality score — junior reps score lower on high-decile HCPs
                        if is_junior and decile >= 8:
                            quality = random.choices([1,2,3,4,5], weights=[0.10,0.25,0.35,0.20,0.10])[0]
                        else:
                            quality = random.choices([1,2,3,4,5], weights=[0.05,0.10,0.30,0.35,0.20])[0]

                        # Outcome — high decile + experienced rep + high freq = more SOLD
                        if decile >= 8 and not is_junior and week <= 8:
                            outcome = random.choices(
                                ["SOLD", "NOT_SOLD", "FOLLOW_UP", "ACCOUNT_CLOSED"],
                                weights=[0.30, 0.25, 0.42, 0.03]
                            )[0]
                        elif decile >= 8 and is_junior:
                            outcome = random.choices(
                                ["SOLD", "NOT_SOLD", "FOLLOW_UP", "ACCOUNT_CLOSED"],
                                weights=[0.12, 0.38, 0.46, 0.04]
                            )[0]
                        else:
                            outcome = random.choices(
                                ["SOLD", "NOT_SOLD", "FOLLOW_UP", "ACCOUNT_CLOSED"],
                                weights=[0.20, 0.30, 0.46, 0.04]
                            )[0]

                        patient_type = random.choices(
                            ["NEW", "CONTINUING"],
                            weights=[0.40, 0.60] if week > 12 else [0.65, 0.35]
                        )[0]

                        rows.append({
                            "call_id": f"CALL_{call_counter:07d}",
                            "rep_id": rep["rep_id"],
                            "hcp_id": hcp["hcp_id"],
                            "call_date": call_date.strftime("%m-%d-%Y"),
                            "week_number": week,
                            "call_type": call_type,
                            "detail_quality_score": quality,
                            "samples_dropped": random.random() < 0.45,
                            "outcome": outcome,
                            "patient_type": patient_type,
                            "next_call_planned": outcome in ["FOLLOW_UP", "NOT_SOLD"],
                            "launch_phase": phase,
                            "drug_name": drug,
                        })
                        call_counter += 1

    df = pd.DataFrame(rows)

    # Verify planted F1
    last_launch_calls = df[df["launch_phase"] == "LAST_LAUNCH"]
    hcp_decile_map = hcp_df.set_index("hcp_id")["decile"]
    last_launch_calls = last_launch_calls.copy()
    last_launch_calls["decile"] = last_launch_calls["hcp_id"].map(hcp_decile_map)
    high_decile_early = last_launch_calls[
        (last_launch_calls["decile"] >= 8) &
        (last_launch_calls["week_number"] <= 8)
    ]
    avg_calls = high_decile_early.groupby(["rep_id","hcp_id","week_number"]).size().groupby(["rep_id","hcp_id"]).mean().mean()
    print(f"  call_activity: {len(df):,} rows | avg calls/week on D8+ HCPs (last launch, wk1-8): {avg_calls:.2f} (target ~1.8)")
    return df


# ─────────────────────────────────────────────
# TABLE 5: competitor_intel
# ─────────────────────────────────────────────

def generate_competitor_intel():
    rows = []
    intel_counter = 1

    for tid in TERRITORY_IDS:
        city = TERRITORY_MAP[tid]["city"]

        for phase, n_weeks in [("LAST_LAUNCH", 52), ("CURRENT_LAUNCH", 20)]:

            # ZYLATEC call intensity — higher in cancer center cities (planted F1)
            if city in ["Boston", "New York", "Houston", "Chicago"]:
                zylatec_base_calls = random.randint(38, 55)   # high intensity
                zylatec_share_start = random.uniform(28, 38)
            else:
                zylatec_base_calls = random.randint(18, 32)
                zylatec_share_start = random.uniform(20, 30)

            for week in range(1, n_weeks + 1):
                # ZYLATEC ramps fast in early weeks — the key insight
                if week <= 8:
                    zylatec_calls = int(zylatec_base_calls * random.uniform(1.1, 1.4))
                    # Market share grows week over week
                    zylatec_share = zylatec_share_start + (week * random.uniform(0.8, 1.5))
                else:
                    zylatec_calls = int(zylatec_base_calls * random.uniform(0.85, 1.15))
                    zylatec_share = min(65, zylatec_share_start + 12 + random.uniform(-3, 3))

                # VEXORIN / AURONIX share — the residual (they lose ground in last launch)
                if phase == "LAST_LAUNCH":
                    our_share = max(5, 55 - zylatec_share + random.uniform(-5, 5))
                else:
                    # Current launch — still early, shares more even
                    our_share = max(10, 48 - zylatec_share * 0.6 + random.uniform(-4, 4))

                rows.append({
                    "intel_id": f"INTEL_{intel_counter:06d}",
                    "territory_id": tid,
                    "week_number": week,
                    "zylatec_estimated_calls": zylatec_calls,
                    "zylatec_market_share_pct": round(zylatec_share, 2),
                    "vexorin_market_share_pct": round(our_share, 2),
                    "launch_phase": phase,
                    "data_source": random.choice(["IQVIA", "Symphony Health", "IQVIA"]),
                })
                intel_counter += 1

    df = pd.DataFrame(rows)
    print(f"  competitor_intel: {len(df):,} rows")
    return df


# ─────────────────────────────────────────────
# TABLE 6: product_master
# ─────────────────────────────────────────────

def generate_product_master():
    rows = [
        {
            "drug_name": "VEXORIN",
            "company": "OurCo Pharmaceuticals",
            "indication": "NSCLC 2nd line treatment",
            "fda_approval_date": "06-12-2021",
            "launch_date": "09-01-2021",
            "asp_usd": 12500.00,
            "ndc_code": "10001-201-30",
            "launch_phase": "LAST_LAUNCH",
        },
        {
            "drug_name": "AURONIX",
            "company": "OurCo Pharmaceuticals",
            "indication": "NSCLC 2nd line treatment",
            "fda_approval_date": "03-15-2024",
            "launch_date": "09-01-2024",
            "asp_usd": 14200.00,
            "ndc_code": "10001-301-30",
            "launch_phase": "CURRENT_LAUNCH",
        },
        {
            "drug_name": "ZYLATEC",
            "company": "CompetitorCo Pharma",
            "indication": "NSCLC 2nd line treatment",
            "fda_approval_date": "08-20-2021",
            "launch_date": "10-28-2021",    # launched 8 weeks after VEXORIN
            "asp_usd": 13500.00,
            "ndc_code": "20002-401-30",
            "launch_phase": "LAST_LAUNCH",
        },
    ]
    df = pd.DataFrame(rows)
    print(f"  product_master: {len(df)} rows")
    return df


# ─────────────────────────────────────────────
# TABLE 7: asset_register
# ─────────────────────────────────────────────

def generate_asset_register():
    rows = [
        # All assets acquired well before the analysis window — proves fixed cost stable
        {"asset_id": "ASSET_001", "asset_type": "Manufacturing Plant",
         "asset_name": "NJ Primary Manufacturing Facility",
         "acquisition_date": "03-15-2015", "acquisition_value_usd": 85000000,
         "current_book_value_usd": 52000000, "status": "ACTIVE"},
        {"asset_id": "ASSET_002", "asset_type": "Manufacturing Plant",
         "asset_name": "North Carolina Secondary Plant",
         "acquisition_date": "07-01-2017", "acquisition_value_usd": 42000000,
         "current_book_value_usd": 29400000, "status": "ACTIVE"},
        {"asset_id": "ASSET_003", "asset_type": "Lab Equipment",
         "asset_name": "QA/QC Laboratory Suite",
         "acquisition_date": "11-20-2018", "acquisition_value_usd": 8500000,
         "current_book_value_usd": 4250000, "status": "ACTIVE"},
        {"asset_id": "ASSET_004", "asset_type": "Tech License",
         "asset_name": "Veeva CRM Enterprise License",
         "acquisition_date": "01-01-2019", "acquisition_value_usd": 1200000,
         "current_book_value_usd": 480000, "status": "ACTIVE"},
        {"asset_id": "ASSET_005", "asset_type": "Office",
         "asset_name": "HQ Office — Parsippany NJ",
         "acquisition_date": "06-30-2016", "acquisition_value_usd": 22000000,
         "current_book_value_usd": 14300000, "status": "ACTIVE"},
        {"asset_id": "ASSET_006", "asset_type": "Tech License",
         "asset_name": "IQVIA Data Feed License",
         "acquisition_date": "01-01-2020", "acquisition_value_usd": 950000,
         "current_book_value_usd": 380000, "status": "ACTIVE"},
        {"asset_id": "ASSET_007", "asset_type": "Lab Equipment",
         "asset_name": "R&D Analytical Instruments",
         "acquisition_date": "05-14-2018", "acquisition_value_usd": 3200000,
         "current_book_value_usd": 1280000, "status": "ACTIVE"},
        # One disposed — shows normal asset lifecycle, not cost spike
        {"asset_id": "ASSET_008", "asset_type": "Vehicle Fleet",
         "asset_name": "Field Force Vehicle Fleet (Gen 1)",
         "acquisition_date": "01-01-2016", "acquisition_value_usd": 4800000,
         "current_book_value_usd": 0, "status": "DISPOSED"},
        {"asset_id": "ASSET_009", "asset_type": "Vehicle Fleet",
         "asset_name": "Field Force Vehicle Fleet (Gen 2)",
         "acquisition_date": "01-01-2021", "acquisition_value_usd": 5200000,
         "current_book_value_usd": 3120000, "status": "ACTIVE"},
    ]
    df = pd.DataFrame(rows)
    print(f"  asset_register: {len(df)} rows | last new asset: 2021 (before analysis window)")
    return df


# ─────────────────────────────────────────────
# TABLE 8: hr_master
# ─────────────────────────────────────────────

def generate_hr_master(rep_df):
    rows = []
    # Field reps from rep_master
    for _, rep in rep_df.iterrows():
        rows.append({
            "emp_id": rep["rep_id"].replace("REP", "EMP"),
            "role": rep["designation"],
            "department": "Commercial",
            "join_date": (date(2020, 1, 1) + timedelta(days=random.randint(0, 1000))).strftime("%m-%d-%Y"),
            "annual_ctc_usd": rep["annual_ctc_usd"],
            "region": rep["region"],
            "status": rep["status"],
        })

    # HQ staff — medical affairs, commercial ops, marketing
    hq_roles = [
        ("VP Commercial", "Commercial", 210000),
        ("Director Market Access", "Commercial", 165000),
        ("Brand Manager AURONIX", "Commercial", 130000),
        ("Brand Manager VEXORIN", "Commercial", 125000),
        ("Commercial Analytics Lead", "Commercial", 120000),
        ("Medical Science Liaison", "Medical Affairs", 135000),
        ("Medical Science Liaison", "Medical Affairs", 128000),
        ("Head of Manufacturing", "Manufacturing", 175000),
        ("Plant Operations Manager", "Manufacturing", 118000),
        ("Supply Chain Manager", "Manufacturing", 105000),
        ("CFO", "Corporate", 280000),
        ("HR Business Partner", "Corporate", 95000),
        ("IT Director", "Corporate", 140000),
    ]
    for i, (role, dept, ctc) in enumerate(hq_roles):
        rows.append({
            "emp_id": f"EMP_HQ_{i+1:03d}",
            "role": role,
            "department": dept,
            "join_date": (date(2017, 1, 1) + timedelta(days=random.randint(0, 1500))).strftime("%m-%d-%Y"),
            "annual_ctc_usd": ctc + random.randint(-5000, 5000),
            "region": None,
            "status": "ACTIVE",
        })

    df = pd.DataFrame(rows)
    total_cost = df[df["status"] == "ACTIVE"]["annual_ctc_usd"].sum()
    print(f"  hr_master: {len(df)} rows | total annual payroll: ${total_cost:,.0f}")
    return df


# ─────────────────────────────────────────────
# TABLE 9: production_records
# ─────────────────────────────────────────────

def generate_production_records():
    rows = []
    rec_counter = 1

    for drug, start_year, n_weeks in [
        ("VEXORIN", 2021, 104),   # 2 years post launch
        ("AURONIX", 2024,  20),   # current launch 20 weeks in
    ]:
        base_units = 3800 if drug == "VEXORIN" else 4200
        warehouse  = 5000  # starting inventory

        for week in range(1, n_weeks + 1):
            # Production grows steadily — cost per unit stays flat (proves variable cost ok)
            units_produced = int(base_units * (1 + week * 0.002) * random.uniform(0.92, 1.08))

            # Warehouse builds up during last launch (the overstock finding)
            if drug == "VEXORIN":
                units_sold = int(units_produced * random.uniform(0.55, 0.75))  # can't sell enough
                warehouse  = warehouse + units_produced - units_sold
            else:
                units_sold = int(units_produced * random.uniform(0.65, 0.85))  # slightly better
                warehouse  = max(1000, warehouse + units_produced - units_sold)

            raw_material_cost    = round(random.uniform(175, 190), 2)
            manufacturing_cost   = round(random.uniform(310, 335), 2)
            total_cogs           = round((raw_material_cost + manufacturing_cost) * units_produced, 2)

            year = start_year + (week - 1) // 52

            rows.append({
                "record_id": f"PROD_{rec_counter:05d}",
                "drug_name": drug,
                "week_number": week,
                "year": year,
                "units_produced": units_produced,
                "units_sold": units_sold,
                "units_in_warehouse": warehouse,
                "raw_material_cost_per_unit": raw_material_cost,
                "manufacturing_cost_per_unit": manufacturing_cost,
                "total_cogs_usd": total_cogs,
            })
            rec_counter += 1

    df = pd.DataFrame(rows)
    vexorin_wh = df[df["drug_name"] == "VEXORIN"]["units_in_warehouse"].max()
    print(f"  production_records: {len(df)} rows | VEXORIN peak warehouse stock: {vexorin_wh:,} units (overstock signal)")
    return df


# ─────────────────────────────────────────────
# TABLE 10: promo_spend
# ─────────────────────────────────────────────

def generate_promo_spend():
    rows = []
    spend_counter = 1

    channels = ["CME_Events", "Digital", "Print", "KOL_Engagement", "Samples", "Congress"]

    for phase, drug, start_year, n_months in [
        ("LAST_LAUNCH",    "VEXORIN", 2021, 18),
        ("CURRENT_LAUNCH", "AURONIX", 2024,  5),
    ]:
        # National spend buckets
        national_monthly = {
            "Digital":        random.randint(180000, 220000),
            "Print":          random.randint(40000,  60000),
            "Congress":       random.randint(80000,  120000),
            "KOL_Engagement": random.randint(90000,  130000),
        }

        for month_offset in range(n_months):
            month = ((start_year - 2021) * 12 + month_offset) % 12 + 1
            year  = start_year + month_offset // 12

            # National channels
            for channel, base_spend in national_monthly.items():
                rows.append({
                    "spend_id": f"SPEND_{spend_counter:05d}",
                    "month": month,
                    "year": year,
                    "channel": channel,
                    "drug_name": drug,
                    "spend_usd": round(base_spend * random.uniform(0.85, 1.15), 2),
                    "territory_id": None,
                    "launch_phase": phase,
                })
                spend_counter += 1

            # Territory-level spend (CME Events + Samples)
            for tid in TERRITORY_IDS:
                for channel in ["CME_Events", "Samples"]:
                    base = 8000 if channel == "CME_Events" else 12000
                    rows.append({
                        "spend_id": f"SPEND_{spend_counter:05d}",
                        "month": month,
                        "year": year,
                        "channel": channel,
                        "drug_name": drug,
                        "spend_usd": round(base * random.uniform(0.7, 1.3), 2),
                        "territory_id": tid,
                        "launch_phase": phase,
                    })
                    spend_counter += 1

    df = pd.DataFrame(rows)
    print(f"  promo_spend: {len(df):,} rows")
    return df


# ─────────────────────────────────────────────
# DATA DICTIONARY (reference CSV)
# ─────────────────────────────────────────────

def generate_data_dictionary():
    rows = [
        # hcp_master
        ("hcp_master","hcp_id","Unique HCP identifier in NPI format","string","NPI_001234","NPI_XXXXXX","Yes"),
        ("hcp_master","hcp_name","Doctor full name","string","Dr. Sarah Chen","—","Yes"),
        ("hcp_master","specialty","Medical specialty","string","Medical Oncologist","Medical Oncologist / Pulmonologist / Hemato-Oncologist","Yes"),
        ("hcp_master","city","Practice city","string","Boston","US cities","Yes"),
        ("hcp_master","state","Practice state (2-letter code)","string","MA","US state codes","Yes"),
        ("hcp_master","hospital_affiliation","Primary hospital name","string","Dana-Farber Cancer Institute","—","Yes"),
        ("hcp_master","hospital_type","Type of institution","string","Cancer Center","Cancer Center / Teaching Hospital / Private Practice / Community Hospital","Yes"),
        ("hcp_master","decile","Prescriber potential score — 10 is highest","integer","9","1–10","Yes"),
        ("hcp_master","brand_affinity","Current prescribing preference","string","SWITCHER","VEXORIN_LOYAL / ZYLATEC_LOYAL / SWITCHER / NO_PREFERENCE","Yes"),
        ("hcp_master","monthly_rx_potential","Estimated scripts per month if fully converted","integer","12","Positive integer","Yes"),
        ("hcp_master","years_in_practice","Years since residency completion","integer","14","Positive integer","Yes"),
        ("hcp_master","kol_flag","Key Opinion Leader status","boolean","True","True / False","Yes"),
        ("hcp_master","kol_score","KOL influence score — 0 if not a KOL","integer","4","0–5","Yes"),
        ("hcp_master","territory_id","Territory this HCP belongs to","string","TER_NE_01","Links to territory_master","Yes"),
        # rep_master
        ("rep_master","rep_id","Unique rep identifier","string","REP_001","REP_XXX","Yes"),
        ("rep_master","rep_name","Full name","string","James Whitfield","—","Yes"),
        ("rep_master","region","Sales region","string","NORTHEAST","NORTHEAST / SOUTHEAST / MIDWEST / SOUTH_CENTRAL / MOUNTAIN_WEST / PACIFIC","Yes"),
        ("rep_master","territory_id","Assigned territory — NULL for managers","string","TER_NE_01","Links to territory_master","Yes"),
        ("rep_master","total_experience_years","Total pharma sales experience in years","float","6.5","Positive float","Yes"),
        ("rep_master","onco_experience_years","Oncology-specific experience in years","float","2.0","Positive float — must be <= total_experience_years","Yes"),
        ("rep_master","designation","Job title","string","Territory Business Manager","Territory Business Manager / Senior Territory Business Manager / Regional Business Director / Area Business Manager","Yes"),
        ("rep_master","manager_id","Direct manager rep_id","string","REP_022","Links to rep_master","Yes"),
        ("rep_master","last_launch_performance","Performance quartile during VEXORIN first launch","string","Q3","Q1 (top) / Q2 / Q3 / Q4","Yes"),
        ("rep_master","status","Current employment status","string","ACTIVE","ACTIVE / ON_LEAVE / TERMINATED","Yes"),
        ("rep_master","annual_ctc_usd","Annual compensation in USD","integer","95000","Positive integer","Yes"),
        # territory_master
        ("territory_master","territory_id","Unique territory identifier","string","TER_NE_01","TER_XX_XX","Yes"),
        ("territory_master","territory_name","Descriptive territory name","string","Boston North","—","Yes"),
        ("territory_master","region","Parent region","string","NORTHEAST","Same as rep_master","Yes"),
        ("territory_master","zone","Sub-region grouping","string","New England / Mid-Atlantic","—","Yes"),
        ("territory_master","city","Primary city in territory","string","Boston","US cities","Yes"),
        ("territory_master","state","State","string","MA","US state codes","Yes"),
        ("territory_master","total_hcps_mapped","Total HCPs in territory","integer","28","Positive integer","Yes"),
        ("territory_master","high_decile_hcps","Count of decile 7–10 HCPs in territory","integer","11","Positive integer","Yes"),
        ("territory_master","cancer_centers_count","Major cancer centers in territory","integer","2","Positive integer","Yes"),
        ("territory_master","market_potential_score","Weighted TRx potential — sum of monthly_rx_potential for all HCPs in territory","integer","340","Derived from hcp_master","Yes"),
        ("territory_master","competitor_rep_count","Estimated ZYLATEC rep count in territory — sourced from competitor_intel","integer","3","Positive integer","Yes"),
        # call_activity
        ("call_activity","call_id","Unique call identifier","string","CALL_0000001","CALL_XXXXXXX","Yes"),
        ("call_activity","rep_id","Rep who made the call","string","REP_001","Links to rep_master","Yes"),
        ("call_activity","hcp_id","HCP visited","string","NPI_001234","Links to hcp_master","Yes"),
        ("call_activity","call_date","Date of visit","date","03-15-2024","MM-DD-YYYY","Yes"),
        ("call_activity","week_number","Week relative to launch date — launch = week 1","integer","4","1–52","Yes"),
        ("call_activity","call_type","Mode of engagement","string","F2F","F2F / Virtual / CME_Event / Conference","Yes"),
        ("call_activity","detail_quality_score","Manager-rated call quality score","integer","3","1–5 (5=best)","Yes"),
        ("call_activity","samples_dropped","Starter pack left with HCP","boolean","True","True / False","Yes"),
        ("call_activity","outcome","Result of call","string","FOLLOW_UP","SOLD / NOT_SOLD / FOLLOW_UP / ACCOUNT_CLOSED","Yes"),
        ("call_activity","patient_type","Patient context discussed in call","string","NEW","NEW / CONTINUING","Yes"),
        ("call_activity","next_call_planned","Follow-up visit scheduled","boolean","True","True / False","Yes"),
        ("call_activity","launch_phase","Which launch this call belongs to","string","LAST_LAUNCH","LAST_LAUNCH / CURRENT_LAUNCH","Yes"),
        ("call_activity","drug_name","Drug being detailed","string","VEXORIN","VEXORIN (last launch) / AURONIX (current launch)","Yes"),
        # competitor_intel
        ("competitor_intel","intel_id","Unique record identifier","string","INTEL_000001","—","Yes"),
        ("competitor_intel","territory_id","Territory this data covers","string","TER_NE_01","Links to territory_master","Yes"),
        ("competitor_intel","week_number","Week relative to VEXORIN/AURONIX launch","integer","4","1–52","Yes"),
        ("competitor_intel","zylatec_estimated_calls","Estimated ZYLATEC rep calls in territory that week","integer","47","Positive integer — from IQVIA proxy","Yes"),
        ("competitor_intel","zylatec_market_share_pct","ZYLATEC share of NSCLC 2nd line Rx in territory","float","38.5","0–100","Yes"),
        ("competitor_intel","vexorin_market_share_pct","VEXORIN or AURONIX share of same market","float","29.2","0–100","Yes"),
        ("competitor_intel","launch_phase","Which launch period","string","LAST_LAUNCH","LAST_LAUNCH / CURRENT_LAUNCH","Yes"),
        ("competitor_intel","data_source","Vendor source for this data","string","IQVIA","IQVIA / Symphony Health / Internal Estimate","Yes"),
        # product_master
        ("product_master","drug_name","Drug identifier","string","VEXORIN","VEXORIN / AURONIX / ZYLATEC","Yes"),
        ("product_master","company","Manufacturer","string","OurCo Pharmaceuticals","OurCo Pharmaceuticals / CompetitorCo Pharma","Yes"),
        ("product_master","indication","Approved therapeutic indication","string","NSCLC 2nd line treatment","—","Yes"),
        ("product_master","fda_approval_date","FDA approval date","date","06-12-2021","MM-DD-YYYY","Yes"),
        ("product_master","launch_date","Commercial launch date","date","09-01-2021","MM-DD-YYYY","Yes"),
        ("product_master","asp_usd","Average Sales Price per unit in USD","float","12500.00","Positive float","Yes"),
        ("product_master","ndc_code","National Drug Code","string","10001-201-30","Standard NDC format","Yes"),
        ("product_master","launch_phase","Which launch this product belongs to","string","LAST_LAUNCH","LAST_LAUNCH / CURRENT_LAUNCH","Yes"),
        # asset_register
        ("asset_register","asset_id","Unique asset identifier","string","ASSET_001","—","Yes"),
        ("asset_register","asset_type","Asset category","string","Manufacturing Plant","Manufacturing Plant / Lab Equipment / Office / Tech License / Vehicle Fleet","Yes"),
        ("asset_register","asset_name","Descriptive asset name","string","NJ Manufacturing Facility","—","Yes"),
        ("asset_register","acquisition_date","Date purchased","date","04-01-2018","MM-DD-YYYY","Yes"),
        ("asset_register","acquisition_value_usd","Original purchase cost in USD","integer","45000000","Positive integer","Yes"),
        ("asset_register","current_book_value_usd","Depreciated book value today in USD","integer","31500000","Positive integer","Yes"),
        ("asset_register","status","Current asset state","string","ACTIVE","ACTIVE / DISPOSED / UNDER_MAINTENANCE","Yes"),
        # hr_master
        ("hr_master","emp_id","Employee identifier","string","EMP_001","—","Yes"),
        ("hr_master","role","Job function / title","string","Territory Business Manager","Field or HQ roles","Yes"),
        ("hr_master","department","Business unit","string","Commercial","Commercial / Medical Affairs / Manufacturing / Corporate","Yes"),
        ("hr_master","join_date","Employment start date","date","03-01-2020","MM-DD-YYYY","Yes"),
        ("hr_master","annual_ctc_usd","Annual compensation in USD","integer","95000","Positive integer","Yes"),
        ("hr_master","region","Assigned region — NULL for HQ roles","string","NORTHEAST","Same as rep_master — nullable","No"),
        ("hr_master","status","Employment status","string","ACTIVE","ACTIVE / ATTRITED / ON_LEAVE","Yes"),
        # production_records
        ("production_records","record_id","Unique production record","string","PROD_00001","—","Yes"),
        ("production_records","drug_name","Drug manufactured","string","VEXORIN","VEXORIN / AURONIX","Yes"),
        ("production_records","week_number","Production week number","integer","14","1–52","Yes"),
        ("production_records","year","Production calendar year","integer","2024","—","Yes"),
        ("production_records","units_produced","Units manufactured that week","integer","4200","Positive integer","Yes"),
        ("production_records","units_sold","Units sold that week","integer","2800","Positive integer","Yes"),
        ("production_records","units_in_warehouse","Closing inventory at week end","integer","18600","Positive integer","Yes"),
        ("production_records","raw_material_cost_per_unit","Input material cost per unit in USD","float","180.50","Positive float","Yes"),
        ("production_records","manufacturing_cost_per_unit","Labour and overhead per unit in USD","float","320.00","Positive float","Yes"),
        ("production_records","total_cogs_usd","Total cost of goods sold that week in USD","float","2100000","Positive float","Yes"),
        # promo_spend
        ("promo_spend","spend_id","Unique spend record","string","SPEND_00001","—","Yes"),
        ("promo_spend","month","Calendar month","integer","3","1–12","Yes"),
        ("promo_spend","year","Calendar year","integer","2024","—","Yes"),
        ("promo_spend","channel","Promotional channel","string","CME_Events","CME_Events / Digital / Print / KOL_Engagement / Samples / Congress","Yes"),
        ("promo_spend","drug_name","Drug this spend supports","string","AURONIX","VEXORIN / AURONIX","Yes"),
        ("promo_spend","spend_usd","Amount spent in USD","float","125000.00","Positive float","Yes"),
        ("promo_spend","territory_id","Territory if local spend — NULL for national","string","TER_NE_01","Links to territory_master — nullable","No"),
        ("promo_spend","launch_phase","Which launch period","string","CURRENT_LAUNCH","LAST_LAUNCH / CURRENT_LAUNCH","Yes"),
    ]

    df = pd.DataFrame(rows, columns=[
        "table_name","column_name","description","data_type",
        "example_value","allowed_values","required"
    ])
    print(f"  data_dictionary: {len(df)} column definitions across {df['table_name'].nunique()} tables")
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Pharma Analytics Copilot — Synthetic Data Generator ===\n")
    print("Generating tables...\n")

    hcp_df  = generate_hcp_master(500)
    rep_df  = generate_rep_master()
    ter_df  = generate_territory_master(hcp_df, rep_df)
    call_df = generate_call_activity(hcp_df, rep_df)
    comp_df = generate_competitor_intel()
    prod_df = generate_product_master()
    asset_df= generate_asset_register()
    hr_df   = generate_hr_master(rep_df)
    prec_df = generate_production_records()
    promo_df= generate_promo_spend()
    dict_df = generate_data_dictionary()

    print("\nSaving CSVs...\n")
    saves = [
        (hcp_df,   "hcp_master.csv"),
        (rep_df,   "rep_master.csv"),
        (ter_df,   "territory_master.csv"),
        (call_df,  "call_activity.csv"),
        (comp_df,  "competitor_intel.csv"),
        (prod_df,  "product_master.csv"),
        (asset_df, "asset_register.csv"),
        (hr_df,    "hr_master.csv"),
        (prec_df,  "production_records.csv"),
        (promo_df, "promo_spend.csv"),
        (dict_df,  "data_dictionary.csv"),
    ]

    for df, fname in saves:
        path = os.path.join(OUTPUT_DIR, fname)
        df.to_csv(path, index=False)
        print(f"  ✓ {fname:<35} ({len(df):>7,} rows)")

    print("\n=== Done. All files in data/synthetic/ ===\n")
    print("Planted findings verification:")
    print("  F1 - Check call_activity: avg calls/week on D8+ HCPs weeks 1-8 (last launch) should be ~1.8")
    print("  F2 - Check rep_master: reps with onco_experience_years < 2 on high-value territories")
    print("  F3 - Check territory_master: cancer center cities undercovered vs competitor_rep_count")
    print("  F4 - Check call_activity: outcome distribution skewed toward low-decile HCPs for volume")
    print("  F5 - Check call_activity current launch: TER_NO_01, TER_NO_02, TER_MI_01, TER_PA_01 repeating pattern")
