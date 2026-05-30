"""
analysis_engine.py
Multi-provider LLM layer — Anthropic, Groq, or Gemini.

Set in .env:
    LLM_PROVIDER=groq          # or anthropic / gemini
    GROQ_API_KEY=...
    ANTHROPIC_API_KEY=...
    GEMINI_API_KEY=...

Run standalone to test:
    python src/analysis_engine.py
"""

import os
import sys
import time
from pathlib import Path
from typing import Generator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_all
from kpi_engine import compute_kpis
from prompt_builder import build_prompt, build_methodology_footer

# ── Load .env ──
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Config ──
MAX_TOKENS  = 1500
TEMPERATURE = 0.3

SECTION_HEADERS = [
    "## SITUATION",
    "## COMPLICATION",
    "## ISSUE TREE",
    "## KEY FINDINGS",
    "## RECOMMENDATIONS",
    "## LAUNCH RISK VERDICT",
]

SECTION_META = {
    "SITUATION":           {"icon": "🏢", "color": "#1a1a2e", "label": "Situation",           "order": 1},
    "COMPLICATION":        {"icon": "⚠️",  "color": "#e63946", "label": "Complication",        "order": 2},
    "ISSUE TREE":          {"icon": "🌳", "color": "#457b9d", "label": "Issue Tree",           "order": 3},
    "KEY FINDINGS":        {"icon": "🔍", "color": "#2d6a4f", "label": "Key Findings",         "order": 4},
    "RECOMMENDATIONS":     {"icon": "🎯", "color": "#e76f51", "label": "Recommendations",      "order": 5},
    "LAUNCH RISK VERDICT": {"icon": "🚦", "color": "#6d2b3d", "label": "Launch Risk Verdict",  "order": 6},
}


def get_rag_color(rag: str) -> str:
    return {"RED": "#e63946", "AMBER": "#f4a261", "GREEN": "#2d6a4f"}.get(rag, "#888888")


# ═══════════════════════════════════════════════════════════
# PROVIDER DETECTION
# ═══════════════════════════════════════════════════════════

def _get_provider() -> str:
    provider = os.environ.get("LLM_PROVIDER", "").lower().strip()
    if provider in ("anthropic", "groq", "gemini"):
        return provider
    # Auto-detect from which key is present
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    raise EnvironmentError(
        "No LLM provider configured.\n"
        "Add one of these to your .env file:\n"
        "  ANTHROPIC_API_KEY=sk-ant-...\n"
        "  GROQ_API_KEY=gsk_...\n"
        "  GEMINI_API_KEY=AIza...\n"
        "Optionally set LLM_PROVIDER=anthropic|groq|gemini to be explicit."
    )


# ═══════════════════════════════════════════════════════════
# NON-STREAMING RUNNER
# ═══════════════════════════════════════════════════════════

def run_analysis(kpis: dict, analysis_type: str = "full_diagnostic") -> dict:
    system_prompt, user_prompt = build_prompt(kpis, analysis_type)
    provider = _get_provider()
    start    = time.time()

    try:
        if provider == "anthropic":
            full_text, usage = _run_anthropic(system_prompt, user_prompt)
        elif provider == "groq":
            full_text, usage = _run_groq(system_prompt, user_prompt)
        elif provider == "gemini":
            full_text, usage = _run_gemini(system_prompt, user_prompt)
        else:
            return _error_result(f"Unknown provider: {provider}")

        return {
            "full_text":   full_text,
            "sections":    _parse_sections(full_text),
            "usage":       usage,
            "elapsed_sec": round(time.time() - start, 2),
            "methodology": build_methodology_footer(kpis),
            "provider":    provider,
            "error":       None,
        }

    except Exception as e:
        return _error_result(str(e))


# ═══════════════════════════════════════════════════════════
# STREAMING RUNNER
# ═══════════════════════════════════════════════════════════

def stream_analysis(kpis: dict, analysis_type: str = "full_diagnostic") -> Generator[str, None, None]:
    system_prompt, user_prompt = build_prompt(kpis, analysis_type)
    provider = _get_provider()

    if provider == "anthropic":
        yield from _stream_anthropic(system_prompt, user_prompt)
    elif provider == "groq":
        yield from _stream_groq(system_prompt, user_prompt)
    elif provider == "gemini":
        yield from _stream_gemini(system_prompt, user_prompt)
    else:
        yield f"Error: unknown provider '{provider}'"


# ═══════════════════════════════════════════════════════════
# ANTHROPIC
# ═══════════════════════════════════════════════════════════

def _run_anthropic(system_prompt: str, user_prompt: str):
    import anthropic
    client   = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    full_text = response.content[0].text
    usage = _usage_anthropic(response.usage)
    return full_text, usage


def _stream_anthropic(system_prompt: str, user_prompt: str):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        for chunk in stream.text_stream:
            yield chunk


def _usage_anthropic(usage) -> dict:
    inp  = usage.input_tokens
    out  = usage.output_tokens
    cost = inp / 1_000_000 * 3.00 + out / 1_000_000 * 15.00
    return {
        "input_tokens":  inp,
        "output_tokens": out,
        "total_tokens":  inp + out,
        "cost_usd":      round(cost, 4),
        "cost_display":  f"${cost:.4f}",
    }


# ═══════════════════════════════════════════════════════════
# GROQ
# ═══════════════════════════════════════════════════════════

def _run_groq(system_prompt: str, user_prompt: str):
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Run: pip install groq")

    client   = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    full_text = response.choices[0].message.content
    usage     = _usage_groq(response.usage)
    return full_text, usage


def _stream_groq(system_prompt: str, user_prompt: str):
    try:
        from groq import Groq
    except ImportError:
        yield "Error: run pip install groq"
        return

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _usage_groq(usage) -> dict:
    inp = getattr(usage, "prompt_tokens",     0)
    out = getattr(usage, "completion_tokens", 0)
    return {
        "input_tokens":  inp,
        "output_tokens": out,
        "total_tokens":  inp + out,
        "cost_usd":      0.0,
        "cost_display":  "Free (Groq)",
    }


# ═══════════════════════════════════════════════════════════
# GEMINI
# ═══════════════════════════════════════════════════════════

def _run_gemini(system_prompt: str, user_prompt: str):
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Run: pip install google-generativeai")

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model    = genai.GenerativeModel(
        "gemini-1.5-flash",
        system_instruction=system_prompt,
    )
    response  = model.generate_content(user_prompt)
    full_text = response.text
    usage     = {
        "input_tokens":  getattr(response.usage_metadata, "prompt_token_count",     0),
        "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
        "total_tokens":  getattr(response.usage_metadata, "total_token_count",      0),
        "cost_usd":      0.0,
        "cost_display":  "Free (Gemini)",
    }
    return full_text, usage


def _stream_gemini(system_prompt: str, user_prompt: str):
    try:
        import google.generativeai as genai
    except ImportError:
        yield "Error: run pip install google-generativeai"
        return

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model  = genai.GenerativeModel(
        "gemini-1.5-flash",
        system_instruction=system_prompt,
    )
    stream = model.generate_content(user_prompt, stream=True)
    for chunk in stream:
        if chunk.text:
            yield chunk.text


# ═══════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════

def _parse_sections(text: str) -> dict:
    sections      = {}
    current_key   = None
    current_lines = []

    for line in text.split("\n"):
        matched = None
        for header in SECTION_HEADERS:
            if line.strip().startswith(header):
                matched = header.replace("## ", "").strip()
                break
        if matched:
            if current_key:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key   = matched
            current_lines = []
        elif current_key:
            current_lines.append(line)

    if current_key and current_lines:
        sections[current_key] = "\n".join(current_lines).strip()

    if not sections:
        sections["FULL OUTPUT"] = text.strip()

    return sections


def _error_result(message: str) -> dict:
    return {
        "full_text":   "",
        "sections":    {},
        "usage":       {"input_tokens": 0, "output_tokens": 0,
                        "total_tokens": 0, "cost_usd": 0, "cost_display": "$0"},
        "elapsed_sec": 0,
        "methodology": "",
        "provider":    "none",
        "error":       message,
    }


# ═══════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n=== analysis_engine.py — Round-Trip Test ===\n")

    provider = _get_provider()
    print(f"Provider detected: {provider.upper()}\n")

    print("Loading data...")
    _, _, gold = load_all()
    kpis       = compute_kpis(gold)
    print(f"KPIs ready: {sum(len(v) for k,v in kpis.items() if isinstance(v,list))} metrics\n")

    print("Calling API...")
    result = run_analysis(kpis, "full_diagnostic")

    if result["error"]:
        print(f"❌  Error: {result['error']}")
        sys.exit(1)

    print(f"✅  Done in {result['elapsed_sec']}s | "
          f"{result['usage']['total_tokens']} tokens | "
          f"{result['usage']['cost_display']}\n")

    print("─" * 60)
    for section, content in result["sections"].items():
        meta = SECTION_META.get(section, {})
        print(f"\n{meta.get('icon','•')}  {section}")
        print("─" * 40)
        print(content[:400] + ("..." if len(content) > 400 else ""))

    print(f"\n=== Ready for app.py ===\n")
