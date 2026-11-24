#!/usr/bin/env python3
"""
Company-Specific Event Study Analysis (Canonical UROP Phases).

Fixes single-company exclusive assignment bias by filtering each company
independently using its boolean flag (is_openai, is_anthropic, is_spacex) from
data/articles_with_phases.csv, so multi-company articles contribute to every
company they mention.

Groups each company's subset by its canonical company-specific phase:
  - baseline
  - rumor
  - confidential_filing
  - public_s1
  - roadshow
  - post_listing

Metrics evaluated per phase and company:
  1. Sample size N
  2. Company citation % (percentage of articles quoting or citing Company sources)
  3. Mean Gemini financial sentiment score (Negative=-1.0, Neutral=0.0, Positive=+1.0)
  4. Mean general VADER compound score (-1.0 to +1.0)
  5. Mean Risk/Benefit ratio (len(risks) / (len(benefits) + 1))

Outputs high-resolution (300 DPI) per-company plots:
  - data/plot_event_study_openai.png
  - data/plot_event_study_anthropic.png
  - data/plot_event_study_spacex.png

Usage:
    python src/16_company_event_study.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV    = PROJECT_ROOT / "data" / "articles_with_phases.csv"

OUT_OPENAI    = PROJECT_ROOT / "data" / "plot_event_study_openai.png"
OUT_ANTHROPIC = PROJECT_ROOT / "data" / "plot_event_study_anthropic.png"
OUT_SPACEX    = PROJECT_ROOT / "data" / "plot_event_study_spacex.png"

DPI = 300

# Canonical chronological phase order
CANONICAL_PHASES = [
    "baseline",
    "rumor",
    "confidential_filing",
    "public_s1",
    "roadshow",
    "post_listing",
]

PHASE_LABELS = {
    "baseline": "Baseline",
    "rumor": "Rumor",
    "confidential_filing": "Confidential\nFiling",
    "public_s1": "Public S-1",
    "roadshow": "Roadshow",
    "post_listing": "Post-Listing",
}

COMPANY_CONFIGS = [
    {
        "name": "OpenAI",
        "flag_col": "is_openai",
        "phase_col": "phase_openai",
        "color": "#00B894",
        "out_plot": OUT_OPENAI,
        "subtitle": "Confidential S-1 Filing: June 08, 2026",
    },
    {
        "name": "Anthropic",
        "flag_col": "is_anthropic",
        "phase_col": "phase_anthropic",
        "color": "#6C5CE7",
        "out_plot": OUT_ANTHROPIC,
        "subtitle": "Confidential S-1: June 01, 2026 | Public S-1: Oct 26, 2026",
    },
    {
        "name": "SpaceX",
        "flag_col": "is_spacex",
        "phase_col": "phase_spacex",
        "color": "#E07B39",
        "out_plot": OUT_SPACEX,
        "subtitle": "Public S-1: May 20, 2026 | Target Listing: June 12, 2026",
    },
]

# ---------------------------------------------------------------------------
# Styling setup
# ---------------------------------------------------------------------------

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_parse_json_list(val) -> list:
    """Parse JSON string arrays into Python lists safely."""
    if not isinstance(val, str) or not val.strip():
        return []
    try:
        parsed = json.loads(val)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


def load_dataset() -> pd.DataFrame:
    """Load canonical dataset and preprocess metrics."""
    print("=" * 80)
    print("LOADING CANONICAL PHASE DATASET")
    print("=" * 80)
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing required dataset: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Loaded {len(df):,} total rows from: {INPUT_CSV.name}")

    # Parse NLP lists
    df["sources_list"] = df["sources"].apply(safe_parse_json_list)
    df["risks_list"] = df["risks"].apply(safe_parse_json_list)
    df["benefits_list"] = df["benefits"].apply(safe_parse_json_list)

    # Metric 1: Company source citation
    df["has_company"] = df["sources_list"].apply(lambda s: "Company" in s)

    # Metric 2: Financial sentiment score mapping
    sent_map = {"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0}
    df["financial_score"] = df["financial_sentiment"].map(sent_map)

    # Metric 3: General VADER compound score
    df["vader_score"] = pd.to_numeric(df["vader_compound"], errors="coerce")

    # Metric 4: Risk / Benefit framing ratio
    df["risk_benefit_ratio"] = df["risks_list"].apply(len) / (df["benefits_list"].apply(len) + 1)

    return df


# ---------------------------------------------------------------------------
# Metrics Aggregation
# ---------------------------------------------------------------------------

def compute_company_metrics(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Calculate metrics for each company across its canonical phases."""
    results = {}

    for cfg in COMPANY_CONFIGS:
        comp_name = cfg["name"]
        flag_col = cfg["flag_col"]
        phase_col = cfg["phase_col"]

        # Filter independently using boolean flag and valid phase assignment
        comp_df = df[df[flag_col] & df[phase_col].notna()].copy()

        phase_rows = []
        for phase in CANONICAL_PHASES:
            sub = comp_df[comp_df[phase_col] == phase]
            n = len(sub)
            if n == 0:
                continue

            c_pct = sub["has_company"].mean() * 100.0
            fin_sent_avg = sub["financial_score"].dropna().mean()
            vader_avg = sub["vader_score"].dropna().mean()
            rb_ratio_avg = sub["risk_benefit_ratio"].mean()

            phase_rows.append({
                "phase": phase,
                "display_label": PHASE_LABELS.get(phase, phase.title()),
                "N": n,
                "company_citation_pct": c_pct,
                "financial_sentiment_score": fin_sent_avg,
                "vader_compound_score": vader_avg,
                "risk_benefit_ratio": rb_ratio_avg,
            })

        results[comp_name] = pd.DataFrame(phase_rows)

    return results


def print_summary_table(results: dict[str, pd.DataFrame]) -> None:
    """Print clean raw metrics table per company and phase."""
    print("\n" + "=" * 95)
    print("CANONICAL EVENT STUDY: RAW METRICS PER COMPANY & PHASE")
    print("=" * 95)

    for cfg in COMPANY_CONFIGS:
        comp_name = cfg["name"]
        print(f"\n--- {comp_name.upper()} ({cfg['subtitle']}) ---")
        header = (
            f"{'Phase':<22} | {'N':>6} | {'Company Cit%':>12} | "
            f"{'Fin Sent':>10} | {'VADER Comp':>10} | {'Risk/Ben Ratio':>14}"
        )
        print(header)
        print("-" * 88)

        summary_df = results[comp_name]
        for _, r in summary_df.iterrows():
            print(
                f"{r['phase']:<22} | {int(r['N']):>6,} | {r['company_citation_pct']:>11.1f}% | "
                f"{r['financial_sentiment_score']:>+10.3f} | {r['vader_compound_score']:>+10.3f} | "
                f"{r['risk_benefit_ratio']:>14.3f}"
            )

    print("\n" + "=" * 95 + "\n")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_company_event_study(cfg: dict, summary_df: pd.DataFrame) -> None:
    """Generate publication-ready 3-panel event study plot."""
    comp_name = cfg["name"]
    color = cfg["color"]
    out_plot = cfg["out_plot"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=DPI)

    x = np.arange(len(summary_df))
    x_labels = [
        f"{row['display_label']}\n(N={int(row['N']):,})"
        for _, row in summary_df.iterrows()
    ]

    # Panel 1: Sourcing (Company Citation Rate %)
    ax1 = axes[0]
    y1 = summary_df["company_citation_pct"].tolist()
    ax1.plot(x, y1, marker="o", color=color, linewidth=2.5, markersize=8, zorder=4)
    ax1.set_title("Company Sourcing (%)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_ylabel("Articles Citing Company (%)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=9.5)
    ax1.set_ylim(0, 105)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax1.xaxis.grid(False)

    for xi, yi in zip(x, y1):
        ax1.annotate(
            f"{yi:.1f}%",
            xy=(xi, yi),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            fontweight="semibold",
            color="#1B2631",
        )

    # Panel 2: Sentiment Comparison (Financial Sentiment vs General VADER)
    ax2 = axes[1]
    y2_fin = summary_df["financial_sentiment_score"].tolist()
    y2_vad = summary_df["vader_compound_score"].tolist()

    ax2.axhline(0.0, color="#7F8C8D", linestyle="--", linewidth=1.0, alpha=0.7)

    line_fin = ax2.plot(
        x, y2_fin,
        marker="s", color="#27AE60", linewidth=2.5, markersize=8, zorder=4,
        label="Gemini Financial (-1 to +1)"
    )
    line_vad = ax2.plot(
        x, y2_vad,
        marker="^", color="#2980B9", linewidth=2.2, markersize=8, linestyle="-.", zorder=4,
        label="VADER General (-1 to +1)"
    )

    ax2.set_title("Sentiment: Financial vs General", fontsize=11, fontweight="bold", pad=10)
    ax2.set_ylabel("Sentiment Score (-1.0 to +1.0)", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=9.5)
    ax2.set_ylim(-0.25, 1.1)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax2.xaxis.grid(False)
    ax2.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#BDC3C7", fontsize=8.5, loc="lower left")

    for xi, yi in zip(x, y2_fin):
        ax2.annotate(
            f"{yi:+.2f}",
            xy=(xi, yi),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=9.0,
            fontweight="bold",
            color="#145A32",
        )

    for xi, yi in zip(x, y2_vad):
        ax2.annotate(
            f"{yi:+.2f}",
            xy=(xi, yi),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=9.0,
            fontweight="bold",
            color="#1B4F72",
        )

    # Panel 3: Risk / Benefit Framing Ratio
    ax3 = axes[2]
    y3 = summary_df["risk_benefit_ratio"].tolist()
    ax3.axhline(1.0, color="#95A5A6", linestyle="--", linewidth=1.0, alpha=0.8, label="Parity (1.0)")
    ax3.plot(x, y3, marker="D", color="#C0392B", linewidth=2.5, markersize=7.5, zorder=4)
    ax3.set_title("Risk / Benefit Framing Ratio", fontsize=11, fontweight="bold", pad=10)
    ax3.set_ylabel("Risk / (Benefit + 1) Ratio", fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(x_labels, fontsize=9.5)
    ax3.set_ylim(0.4, max(max(y3) * 1.25, 1.4))
    ax3.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax3.xaxis.grid(False)

    for xi, yi in zip(x, y3):
        ax3.annotate(
            f"{yi:.3f}x",
            xy=(xi, yi),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            fontweight="semibold",
            color="#78281F",
        )

    plt.suptitle(
        f"{comp_name} Event Study: Coverage Dynamics Across Canonical IPO Phases",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
        color="#1A252F",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {comp_name} Event Study Plot: {out_plot}")


# ---------------------------------------------------------------------------
# Main CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("PER-COMPANY EVENT STUDY (MULTI-COMPANY INDEPENDENT FILTERING)")
    print("=" * 80)

    df = load_dataset()
    results = compute_company_metrics(df)

    print_summary_table(results)

    print("Regenerating per-company event study plots...")
    for cfg in COMPANY_CONFIGS:
        plot_company_event_study(cfg, results[cfg["name"]])

    print("\nEvent study completed successfully. All plots regenerated in data/ directory.")
    print("=" * 80)


if __name__ == "__main__":
    main()
