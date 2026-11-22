#!/usr/bin/env python3
"""
Plot sentiment comparison: VADER vs Gemini.

Generates a grouped bar chart comparing the percentage distribution of
general sentiment (VADER) and financial sentiment (Gemini) across the
common subset of analyzed articles.

Saved to:
    data/plot_sentiment_comparison.png

Usage:
    python src/12_plot_sentiment_comparison.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV    = PROJECT_ROOT / "data" / "merged_sentiment_analysis.csv"
OUTPUT_PNG   = PROJECT_ROOT / "data" / "plot_sentiment_comparison.png"

CATEGORIES   = ["Positive", "Neutral", "Negative"]

# Color palette: light blue for VADER, dark blue for Gemini
COLOR_VADER  = "#5DADE2"  # Light blue
COLOR_GEMINI = "#1B4F72"  # Dark blue
DPI          = 300


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")


# ---------------------------------------------------------------------------
# Plotting function
# ---------------------------------------------------------------------------

def generate_comparison_plot() -> None:
    print("=" * 70)
    print("GENERATING SENTIMENT COMPARISON PLOT")
    print("=" * 70)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded dataset: {INPUT_CSV}")
    print(f"Total articles in merged dataset: {len(df):,}")

    # Filter to valid categories if any unexpected values exist
    valid_df = df[
        df["vader_label"].isin(CATEGORIES)
        & df["financial_sentiment"].isin(CATEGORIES)
    ].copy()
    n_articles = len(valid_df)

    if n_articles == 0:
        raise ValueError("No valid rows found for sentiment comparison.")

    # Calculate percentage distributions
    vader_counts = valid_df["vader_label"].value_counts()
    gemini_counts = valid_df["financial_sentiment"].value_counts()

    vader_pct = [vader_counts.get(cat, 0) / n_articles * 100 for cat in CATEGORIES]
    gemini_pct = [gemini_counts.get(cat, 0) / n_articles * 100 for cat in CATEGORIES]

    print("\nSentiment Distributions (%):")
    for cat, v_val, g_val in zip(CATEGORIES, vader_pct, gemini_pct):
        print(f"  {cat:<8} -> VADER: {v_val:5.1f}% | Gemini: {g_val:5.1f}%")

    # Bar positions
    x = np.arange(len(CATEGORIES))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6), dpi=DPI)

    # Plot bars
    bars_vader = ax.bar(
        x - bar_width / 2,
        vader_pct,
        width=bar_width,
        label="VADER (General Sentiment)",
        color=COLOR_VADER,
        edgecolor="#34495E",
        linewidth=0.8,
        zorder=3,
    )

    bars_gemini = ax.bar(
        x + bar_width / 2,
        gemini_pct,
        width=bar_width,
        label="Gemini (Financial Sentiment)",
        color=COLOR_GEMINI,
        edgecolor="#154360",
        linewidth=0.8,
        zorder=3,
    )

    # Add percentage labels above each bar
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # 5 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="semibold",
                color="#2C3E50",
            )

    add_labels(bars_vader)
    add_labels(bars_gemini)

    # Axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES, fontsize=12, fontweight="medium")
    ax.set_ylabel("Percentage of Articles (%)", fontsize=12, labelpad=10)
    ax.set_xlabel("Sentiment Category", fontsize=12, labelpad=10)

    # Set y limits with comfortable headroom for top annotations
    max_pct = max(max(vader_pct), max(gemini_pct))
    ax.set_ylim(0, min(100, max_pct * 1.15))
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.xaxis.grid(False)

    # Title & Subtitle / Annotation
    plt.suptitle(
        "General vs Financial Sentiment: Divergence in AI IPO Coverage",
        fontsize=14,
        fontweight="bold",
        y=0.98,
        color="#1A252F",
    )
    ax.set_title(
        f"VADER (lexicon) vs Gemini (LLM-based financial reasoning). N={n_articles:,} articles.",
        fontsize=10.5,
        color="#566573",
        pad=14,
    )

    # Legend
    ax.legend(
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#D5D8DC",
        fontsize=10.5,
        loc="upper right",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save high-resolution PNG
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[Saved] Plot exported successfully to: {OUTPUT_PNG}")
    print("=" * 70)


def main() -> None:
    generate_comparison_plot()


if __name__ == "__main__":
    main()

