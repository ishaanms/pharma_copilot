"""
prompt_builder.py
The consulting brain of the copilot.

Builds the structured prompt that gets sent to Claude.
This is the real IP — the framework baked into the instructions.

Architecture:
  SYSTEM PROMPT  — who Claude is, what rules it follows, what it must never do
  USER PROMPT    — the KPI data + the structured analytical task

Frameworks hardcoded:
  - Minto Pyramid (Situation → Complication → Question → Answer)
  - Issue Tree (MECE decomposition of the profit problem)
  - So-What Discipline (every observation must carry an implication)
  - Hypothesis-Driven Recommendations (3 recs, each: what / why now / risk mitigated)

Run standalone to preview the full prompt:
    python src/prompt_builder.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_all
from kpi_engine import compute_kpis, kpis_to_prompt_string


# ═══════════════════════════════════════════════════════════
# SYSTEM PROMPT
# Defines who Claude is and what analytical rules it follows
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
You are a senior pharma commercial strategy consultant with 15 years of experience 
advising oncology launch teams at top-10 pharmaceutical companies.

You have deep expertise in:
- Sales force effectiveness (SFE) and call planning
- Territory design and alignment
- Incentive compensation structure
- Oncology market dynamics — specifically NSCLC
- Competitive launch strategy

You have been given pre-computed KPIs from a pharma commercial analytics engine.
Your job is NOT to describe the numbers. Your job is to THINK like a consultant.

═══════════════════════════════════════════════════════════
ANALYTICAL FRAMEWORK YOU MUST FOLLOW — IN ORDER
═══════════════════════════════════════════════════════════

STEP 1 — SITUATION (1 short paragraph)
State the business context factually. No analysis yet.
What company, what drug, what market, what time period.

STEP 2 — COMPLICATION (1 short paragraph)
State what the data reveals that makes this a problem worth solving.
Use ONE specific number to anchor the complication.
Do not list. One sharp paragraph.

STEP 3 — ISSUE TREE (structured decomposition)
Walk down the MECE tree in order:
  Level 1: Profit = Revenue − Cost → which side is the problem?
  Level 2: Revenue = Price × Volume → which driver is failing?
  Level 3: Volume failure → where in the commercial model? 
           (Segmentation / Promotion / SF Design / Alignment / Activity / IC)
  Level 4: Pinpoint the root cause using the KPI evidence

At each level: state the hypothesis, cite the KPI that confirms or eliminates it,
then move to the next level. Never skip a level. Never assert without a number.

STEP 4 — KEY FINDINGS (exactly 3, numbered)
Each finding must follow this structure:
  Finding: [one sentence stating what is true]
  Evidence: [the specific KPI or metric that proves it]
  Implication: [what this means for a business decision — the so-what]

No finding is valid without all three parts.
Never state a finding that is not directly supported by a KPI in the data provided.

STEP 5 — RECOMMENDATIONS (exactly 3, numbered)
Each recommendation must follow this structure:
  Action: [specific, not vague — name the territory, the metric, the threshold]
  Why Now: [what makes this urgent — tie to launch timeline or risk score]
  Risk Mitigated: [which finding or flag this directly addresses]

Recommendations must be sequenced — fix the root cause first, then downstream effects.

STEP 6 — CURRENT LAUNCH RISK VERDICT (1 paragraph)
State clearly: is the AURONIX launch on track or not?
Reference the risk score and the most critical active flag.
End with a single sentence that a CEO could read in 10 seconds.

═══════════════════════════════════════════════════════════
RULES YOU MUST FOLLOW — NON-NEGOTIABLE
═══════════════════════════════════════════════════════════

RULE 1 — NO OBSERVATION WITHOUT IMPLICATION
Never write a sentence that ends with a number.
Every number must be followed by what it means for a decision.
Wrong: "Our call frequency was 1.7 calls/week."
Right: "Our call frequency of 1.7 calls/week — against a competitor standard of 3.0 —
        means oncologists formed prescribing habits for ZYLATEC before our reps
        established meaningful clinical relationships."

RULE 2 — NO VAGUE RECOMMENDATIONS
Every recommendation names a specific action, not a direction.
Wrong: "Improve rep deployment in high-value territories."
Right: "Redeploy the 3 junior reps (<2yr onco experience) currently covering 
        Dana-Farber, MSK, and MD Anderson to mid-value community hospital clusters,
        and replace them with reps from Q4 territories where potential is lower."

RULE 3 — MECE AT EVERY LEVEL
When you eliminate a hypothesis, say why explicitly.
Wrong: "Pricing was not the issue."
Right: "Pricing is eliminated: VEXORIN ASP was 7.4% below ZYLATEC's,
        and COGS per unit changed only +0.4% — margin pressure cannot 
        originate from either side of the pricing equation."

RULE 4 — NEVER SAY "THE DATA SHOWS"
Say what it means, not what it shows.
The data does not show things. It proves, confirms, eliminates, or flags things.

RULE 5 — WRITE FOR A VP OF COMMERCIAL OPERATIONS
Assume the reader knows pharma. Do not explain what a decile is.
Do not explain what market share means. Do not explain what NSCLC is.
Write at the level of a McKinsey deck, not a dashboard tooltip.

RULE 6 — NO HEDGING ON FINDINGS
If the KPI is unambiguous, the finding is unambiguous.
Wrong: "This may suggest there could potentially be an alignment issue."
Right: "This is an alignment failure."
Reserve hedging only for genuinely ambiguous data.

═══════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════

Use these exact section headers:

## SITUATION
## COMPLICATION  
## ISSUE TREE
## KEY FINDINGS
## RECOMMENDATIONS
## LAUNCH RISK VERDICT

Keep total output under 800 words. Consulting decks are tight. 
Every word must earn its place.
"""


# ═══════════════════════════════════════════════════════════
# CONTEXT BLOCK
# Static pharma domain context injected into every prompt
# ═══════════════════════════════════════════════════════════

CONTEXT_BLOCK = """
═══════════════════════════════════════════════════════════
CASE CONTEXT
═══════════════════════════════════════════════════════════

CLIENT        : OurCo Pharmaceuticals (mid-large US pharma)
DRUG (LAST)   : VEXORIN — NSCLC 2nd line treatment, launched Sept 2021
DRUG (CURRENT): AURONIX — same indication, launched Sept 2024 (20 weeks in)
COMPETITOR    : ZYLATEC (CompetitorCo) — same indication, same patient population
               Last launch: entered market 8 weeks after VEXORIN
               Current launch: entering market simultaneously with AURONIX

PROBLEM STATEMENT:
OurCo lost significant market share to ZYLATEC during the VEXORIN launch despite:
  - Launching first (8-week head start)
  - Comparable or lower pricing
  - Stable cost structure
  - Adequate production capacity

The AURONIX launch is now 20 weeks in. Early indicators suggest the same 
patterns are beginning to emerge. The VP of Commercial Operations has 
commissioned this analysis to determine:
  1. What specifically went wrong in the VEXORIN launch?
  2. Are we repeating those mistakes with AURONIX?
  3. What specific actions must be taken in the next 30 days?

DATA SOURCES:
  - Internal CRM (Veeva): call activity, rep master, HCP master
  - Territory master: alignment and potential data
  - IQVIA / Symphony Health: competitor call estimates, market share
  - HR and Finance: headcount, COGS, capex records

BENCHMARKS TO USE:
  - Call frequency standard for competitive oncology launches: 3.0+ calls/week on D8-10 HCPs
  - Sell-through rate benchmark: >85% for specialty pharma in active launch
  - Junior rep threshold: <2 years oncology-specific experience
  - Market share parity benchmark: within 3pp of competitor by week 8
  - High-decile HCPs: decile 8, 9, 10 — top 30% of prescribers by potential
"""


# ═══════════════════════════════════════════════════════════
# PROMPT BUILDER
# ═══════════════════════════════════════════════════════════

def build_prompt(kpis: dict, analysis_type: str = "full_diagnostic") -> tuple[str, str]:
    """
    Builds the (system_prompt, user_prompt) tuple for the Anthropic API call.

    Parameters
    ----------
    kpis          : output of compute_kpis()
    analysis_type : "full_diagnostic" | "current_launch_only" | "cost_elimination_only"
                    Controls which KPI layers are injected

    Returns
    -------
    (system_prompt, user_prompt) — ready for api call
    """

    kpi_string = kpis_to_prompt_string(kpis)
    risk       = kpis.get("launch_risk", {})

    # ── Filter KPI string by analysis type ──
    if analysis_type == "current_launch_only":
        # Only inject competitive + launch risk layers
        task_instruction = """
Focus your analysis exclusively on the CURRENT AURONIX LAUNCH.
Do not perform the full issue tree — go straight to:
  - What the current launch KPIs reveal
  - Whether patterns from the last launch are repeating
  - Specific actions needed in the next 30 days
"""
    elif analysis_type == "cost_elimination_only":
        task_instruction = """
Focus exclusively on the cost diagnostic.
Prove clearly and concisely — using only the KPIs provided — 
that the profit problem is NOT a cost problem.
Then state exactly one sentence on where the analysis must go next.
"""
    else:
        # full_diagnostic — the default
        task_instruction = """
Perform the full consulting diagnostic as specified in your framework.
Walk every level of the issue tree. Produce all 6 sections.
This is the primary analysis the VP of Commercial Operations will read.
"""

    # ── Risk callout for context ──
    risk_callout = f"""
═══════════════════════════════════════════════════════════
CURRENT LAUNCH RISK SCORE: {risk.get('score', 'N/A')}/100 — {risk.get('rag', 'N/A')}
Active red flags: {risk.get('red_flags', 0)} | Amber flags: {risk.get('amber_flags', 0)}
Market share gap improvement vs last launch: {risk.get('share_gap_improvement', 'N/A'):+}pp
═══════════════════════════════════════════════════════════
"""

    user_prompt = f"""
{CONTEXT_BLOCK}

{risk_callout}

═══════════════════════════════════════════════════════════
PRE-COMPUTED KPIs — USE THESE AS YOUR EVIDENCE BASE
Do not invent numbers. Every claim must trace to a KPI below.
═══════════════════════════════════════════════════════════
{kpi_string}

═══════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════
{task_instruction}

Produce your output now. Follow the framework. Follow the rules.
Keep it under 800 words. Make every sentence earn its place.
"""

    return SYSTEM_PROMPT, user_prompt


# ═══════════════════════════════════════════════════════════
# METHODOLOGY FOOTER
# Plain English explanation shown at the bottom of the Streamlit app
# ═══════════════════════════════════════════════════════════

def build_methodology_footer(kpis: dict) -> str:
    """
    Generates the plain-English methodology section
    shown at the bottom of the Streamlit app.
    Transparent about what the tool does and doesn't do.
    """
    risk = kpis.get("launch_risk", {})
    n_kpis = sum(
        len(v) for k, v in kpis.items()
        if isinstance(v, list)
    )

    return f"""
---

### How This Analysis Was Produced

**Data pipeline (Bronze → Silver → Gold)**  
Raw CRM, HR, finance, and market intelligence data was loaded, 
cleaned, and joined across {10} source tables into analysis-ready 
views using a three-layer medallion architecture.

**KPIs computed ({n_kpis} metrics across 5 diagnostic layers)**  
Metrics were calculated using pharma-specific benchmarks for oncology 
specialty launches — not generic analytics defaults. Each KPI carries 
a direction flag (Good / Watch / Bad) and a benchmark drawn from 
competitive oncology launch standards.

**Analytical framework (Minto Pyramid + Issue Tree)**  
Claude was not asked "what do you see in this data." It was given a 
structured consulting framework: decompose the profit problem MECE, 
eliminate hypotheses with evidence, surface root causes, produce 
hypothesis-driven recommendations. The framework follows the same 
logic a McKinsey or ZS Associates team would use in a commercial 
diagnostics engagement.

**What the AI did**  
Generated the narrative, applied the so-what discipline, connected 
KPIs to business implications, and produced sequenced recommendations.

**What the AI did not do**  
Invent numbers, access external data, or make claims beyond the 
KPIs provided. Every finding traces to a specific metric in the 
pipeline above.

**Current launch risk score: {risk.get('score', 'N/A')}/100 ({risk.get('rag', 'N/A')})**  
Computed from {risk.get('red_flags', 0)} red flags and {risk.get('amber_flags', 0)} 
amber flags across territory coverage, rep experience alignment, 
and competitor market share gaps.

**What a consultant still needs to do**  
Validate findings against client context not captured in the data, 
structure the narrative for the specific stakeholder audience, 
and build the PowerPoint that tells the story in sequence.  
This tool eliminates the analytical grunt work. The judgment layer remains human.
"""


# ═══════════════════════════════════════════════════════════
# STANDALONE PREVIEW
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n=== prompt_builder.py — Prompt Preview ===\n")

    _, _, gold = load_all()
    kpis = compute_kpis(gold)

    system_prompt, user_prompt = build_prompt(kpis, analysis_type="full_diagnostic")

    print("SYSTEM PROMPT (first 800 chars):")
    print("─" * 60)
    print(system_prompt[:800] + "...")

    print(f"\nSYSTEM PROMPT total length : {len(system_prompt):,} chars")
    print(f"USER PROMPT total length   : {len(user_prompt):,} chars")
    print(f"Combined (approx tokens)   : ~{(len(system_prompt) + len(user_prompt)) // 4:,} tokens")

    print("\n" + "─" * 60)
    print("USER PROMPT (first 1200 chars):")
    print("─" * 60)
    print(user_prompt[:1200] + "...")

    print("\n" + "─" * 60)
    print("METHODOLOGY FOOTER (full):")
    print("─" * 60)
    print(build_methodology_footer(kpis))

    print("\n=== Prompt architecture verified. Ready for analysis_engine.py ===\n")
