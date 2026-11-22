#!/usr/bin/env python3
"""
Week 3 Presentation Visualizations: Sourcing and Risk/Benefit Framing.

Generates two publication-quality plots from Gemini NLP analysis results:
  1. Plot 1 (Source Distribution):
     - Frequency of cited source categories (Company, Analyst, Journalist, Competitor, Anonymous, Other).
     - Saved to: data/plot_sources.png

  2. Plot 2 (Risk vs. Benefit Framing - Hypothesis H3):
     - Comparison of average risks vs benefits cited per article.
     - Saved to: data/plot_risk_benefit.png

Usage:
    python src/13_plot_sourcing_risk.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV    = PROJECT_ROOT / "data" / "nlp_analysis_results.csv"
OUT_SOURCES  = PROJECT_ROOT / "data" / "plot_sources.png"
OUT_RISK_BEN = PROJECT_ROOT / "data" / "plot_risk_benefit.png"

DPI = 300

# Standard source categories in preferred display order
SOURCE_CATEGORIES = [
    "Company",
    "Journalist",
    "Analyst",
    "Anonymous",
    "Competitor",
    "Other",
]

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")


# ---------------------------------------------------------------------------
# Data loading & parsing
# ---------------------------------------------------------------------------

def safe_parse_json_list(val) -> list:
    """Parse JSON array strings into Python lists, handling malformed values gracefully."""
    if not isinstance(val, str) or not val.strip():
        return []
    try:
        parsed = json.loads(val)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


def load_and_parse_data(path: Path) -> pd.DataFrame:
    """Load NLP dataset and parse JSON string columns."""
    print(f"Loading dataset: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    print(f"  Total records: {len(df):,}")

    # Exclude error records if any
    if "financial_sentiment" in df.columns:
        valid_mask = df["financial_sentiment"] != "ERROR"
        df = df[valid_mask].copy()
        print(f"  Valid classified records: {len(df):,}")

    # Parse JSON list columns
    df["sources_list"] = df["sources"].apply(safe_parse_json_list)
    df["risks_list"] = df["risks"].apply(safe_parse_json_list)
    df["benefits_list"] = df["benefits"].apply(safe_parse_json_list)

    df["risk_count"] = df["risks_list"].apply(len)
    df["benefit_count"] = df["benefits_list"].apply(len)

    return df


# ---------------------------------------------------------------------------
# Plot 1: Source Distribution
# ---------------------------------------------------------------------------

def plot_sources(df: pd.DataFrame, out_path: Path) -> None:
    """Generate and save bar chart of cited source categories."""
    source_counts = Counter()
    for sources in df["sources_list"]:
        source_counts.update(sources)

    # Sort categories by frequency
    categories = [cat for cat in SOURCE_CATEGORIES if cat in source_counts]
    # Include any extra category if present
    extra_cats = [c for c in source_counts if c not in categories]
    categories.extend(sorted(extra_cats, key=lambda c: source_counts[c], reverse=True))

    counts = [source_counts[cat] for cat in categories]
    total_articles = len(df)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=DPI)

    colors = ["#2B5B84", "#3D7EA6", "#56A3C8", "#7EBEC7", "#A5D6A7", "#D1E7DD"]
    bar_colors = (colors * 3)[:len(categories)]

    bars = ax.bar(
        categories,
        counts,
        color=bar_colors,
        edgecolor="#1C3144",
        linewidth=0.8,
        zorder=3,
        width=0.55,
    )

    # Annotate counts and percentages of articles
    for bar in bars:
        h = bar.get_height()
        pct = (h / total_articles) * 100 if total_articles > 0 else 0
        ax.annotate(
            f"{h:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="semibold",
            color="#2C3E50",
        )

    ax.set_ylabel("Total Citations Across Articles", fontsize=11, labelpad=10)
    ax.set_xlabel("Source Category", fontsize=11, labelpad=10)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.xaxis.grid(False)

    plt.suptitle(
        "Who is Cited in AI IPO Coverage?",
        fontsize=14,
        fontweight="bold",
        y=0.98,
        color="#1A252F",
    )
    ax.set_title(
        f"Frequency of sources quoted or cited across N={total_articles:,} analyzed articles.",
        fontsize=10.5,
        color="#566573",
        pad=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] Source distribution plot: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Risk vs. Benefit Framing (Hypothesis H3)
# ---------------------------------------------------------------------------

def plot_risk_benefit(df: pd.DataFrame, out_path: Path) -> None:
    """Generate and save comparison of average risks vs benefits per article."""
    avg_risks = df["risk_count"].mean()
    avg_benefits = df["benefit_count"].mean()
    total_articles = len(df)
    ratio = avg_risks / avg_benefits if avg_benefits > 0 else float("nan")

    labels = ["Average Risks\nper Article", "Average Benefits\nper Article"]
    averages = [avg_risks, avg_benefits]
    colors = ["#D9534F", "#2E86C1"]  # Coral red for risks, professional blue for benefits

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=DPI)

    bars = ax.bar(
        labels,
        averages,
        color=colors,
        edgecolor="#1C2833",
        linewidth=0.8,
        width=0.45,
        zorder=3,
    )

    # Annotate averages on top of bars
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#1C2833",
        )

    ax.set_ylabel("Average Mentions per Article", fontsize=11, labelpad=10)
    ax.set_ylim(0, max(averages) * 1.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.xaxis.grid(False)

    plt.suptitle(
        "Risk vs. Benefit Framing (Hypothesis H3)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
        color="#1A252F",
    )
    ax.set_title(
        f"Average framing counts across N={total_articles:,} articles. Risk/Benefit Ratio: {ratio:.2f}x",
        fontsize=10.5,
        color="#566573",
        pad=14,
    )

    # Informative annotation callout
    annotation_text = (
        f"Key Takeaway:\n"
        f"Articles mention an average of {avg_risks:.2f} risks\n"
        f"vs {avg_benefits:.2f} benefits ({ratio:.2f}x higher risk focus)."
    )
    ax.text(
        0.95, 0.85,
        annotation_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9F9", edgecolor="#BDC3C7", alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] Risk vs benefit plot: {out_path}")


# ---------------------------------------------------------------------------
# Main CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("WEEK 3 VISUALIZATIONS: SOURCING & RISK/BENEFIT FRAMING")
    print("=" * 70)

    df = load_and_parse_data(INPUT_CSV)
    plot_sources(df, OUT_SOURCES)
    plot_risk_benefit(df, OUT_RISK_BEN)

    print("\nAll plots generated successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()

