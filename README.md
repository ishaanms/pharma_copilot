# Pharma Analytics Copilot

A consulting-grade analytics tool for oncology commercial teams — built on a structured problem-solving framework, not generic AI output.

**Live demo → [pharma-analytics-copilot.streamlit.app](https://pharma-analytics-copilot.streamlit.app)**

---

## What it does

Most analytics tools describe data. This one diagnoses it.

Upload pharma commercial data and the copilot walks a full consulting framework — MECE issue tree, Minto Pyramid narrative, hypothesis-driven recommendations — to answer the question a VP of Commercial Operations actually cares about: *why are we losing, and what do we do about it in the next 30 days?*

Built around a real scenario: a mid-size pharma company lost market share to a competitor during their last oncology launch despite launching first. Their new drug is now 20 weeks into launch. The same patterns are beginning to emerge. This tool diagnoses what went wrong last time and flags where the current launch is at risk — before it's too late to fix.

---

## The framework (the real IP)

The AI doesn't get asked "what do you see in this data." It gets a structured consulting prompt:

```
1. SITUATION      — state the business context factually
2. COMPLICATION   — what the data reveals as the core problem
3. ISSUE TREE     — MECE decomposition: Profit → Revenue → Volume → Commercial model
4. KEY FINDINGS   — exactly 3, each with: Finding / Evidence / Implication
5. RECOMMENDATIONS — exactly 3, each with: Action / Why Now / Risk Mitigated
6. LAUNCH RISK VERDICT — current launch: on track or not, in one paragraph
```

Six non-negotiable rules are hardcoded into the system prompt — including: no observation without implication, no vague recommendations, no hedging on unambiguous findings, write for a VP not a dashboard user.

---

## Data architecture

```
BRONZE  →  10 raw CSV tables (HCP master, rep master, territory, call activity,
           competitor intel, product master, asset register, HR, production, promo spend)

SILVER  →  cleaned, typed, joined — dates parsed, experience flags set,
           categorical dtypes for memory efficiency

GOLD    →  6 analysis-ready views: rep×HCP coverage, call frequency by decile,
           territory performance, outcome by decile, launch comparison,
           risk flags (current launch early warning)

KPI ENGINE  →  23 metrics across 5 diagnostic layers, each with benchmark,
               direction flag (Good/Watch/Bad), and pre-computed so-what

PROMPT BUILDER  →  consulting framework + KPI evidence → structured Claude/Groq prompt

ANALYSIS ENGINE  →  multi-provider LLM call (Anthropic / Groq / Gemini), streaming,
                    section parsing, token tracking
```

---

## The diagnostic story

**The case:** OurCo Pharmaceuticals. VEXORIN launched Sept 2021 for NSCLC 2nd line treatment. ZYLATEC entered 8 weeks later. Despite the head start, VEXORIN lost market share and never recovered. Now AURONIX (same indication) is 20 weeks into launch. ZYLATEC is launching simultaneously.

**Planted findings the framework surfaces:**

| Finding | KPI Evidence |
|---|---|
| Call frequency on high-value oncologists was 1.7/week vs ZYLATEC's 3.1 in the critical weeks 1-8 window | SFE Layer — call frequency by decile |
| 38% of reps covering decile 8-10 HCPs had less than 2 years oncology experience | Rep Alignment Layer — experience vs decile mapping |
| Cancer center territories (Dana-Farber, MSK, MD Anderson) were covered by junior reps while 3 reps competed on the same mid-value cluster | Territory Layer — market potential vs resource allocation |
| Reps hit call volume targets by over-visiting low-decile HCPs — 52% of all calls went to decile 1-5 | IC Gaming signal — call distribution by decile |
| Current AURONIX launch is repeating the same pattern in 4 territories — launch risk score: 100/100 RED | Risk flags — current launch early warning |

---

## Project structure

```
pharma_copilot/
├── app.py                      # Streamlit frontend — 3 tabs, streaming output
├── generate_data.py            # Synthetic data generator (seeded, reproducible)
├── requirements.txt
│
├── data/synthetic/             # 10 CSVs + data dictionary
│   ├── hcp_master.csv
│   ├── rep_master.csv
│   ├── territory_master.csv
│   ├── call_activity.csv       # 188k rows — the core activity table
│   ├── competitor_intel.csv
│   ├── product_master.csv
│   ├── asset_register.csv
│   ├── hr_master.csv
│   ├── production_records.csv
│   ├── promo_spend.csv
│   └── data_dictionary.csv     # column-level documentation for all tables
│
└── src/
    ├── data_loader.py          # Bronze → Silver → Gold transformation
    ├── kpi_engine.py           # 23 pharma-specific KPIs across 5 layers
    ├── prompt_builder.py       # Consulting framework prompt architecture
    └── analysis_engine.py      # Multi-provider LLM (Anthropic / Groq / Gemini)
```

---

## Run locally

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/pharma-analytics-copilot.git
cd pharma-analytics-copilot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add API key (.env in project root)
echo "GROQ_API_KEY=gsk_your_key_here" > .env
# or ANTHROPIC_API_KEY / GEMINI_API_KEY — engine auto-detects

# 4. Generate synthetic data
python generate_data.py

# 5. Run
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## LLM provider support

The engine auto-detects which key is present. Set `LLM_PROVIDER` explicitly to override.

| Provider | Model | Cost | Quality |
|---|---|---|---|
| Anthropic | claude-sonnet-4 | ~$0.035/run | Best — prompt tuned for Claude |
| Groq | llama-3.3-70b-versatile | Free tier | Very good for structured output |
| Gemini | gemini-1.5-flash | Free tier | Good |

```
# .env
LLM_PROVIDER=groq          # or anthropic / gemini
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
```

---

## Deploy to Streamlit Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select repo
3. Main file: `app.py`
4. Advanced settings → Secrets → add your API key:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

5. Deploy → live in ~2 minutes

---

## What this is not

The data is synthetic. The framework is real.

This is a proof of concept showing the analytical logic a commercial analytics consultant would apply to actual pharma data — issue tree decomposition, hypothesis elimination, so-what discipline, sequenced recommendations. The tool eliminates the diagnostic grunt work. The judgment layer — validating findings against client context, building the narrative, presenting to stakeholders — remains human.

---

## Tech stack

- **Python** — pandas, numpy, faker
- **Streamlit** — frontend, streaming output, session state
- **Plotly** — interactive charts
- **Anthropic / Groq / Gemini** — LLM backend
- **Architecture** — medallion pattern (Bronze → Silver → Gold), prompt engineering, structured output parsing

---

*Built by a pharma commercial analyst who thinks in frameworks.*
