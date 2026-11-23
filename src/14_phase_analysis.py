#!/usr/bin/env python3
"""
Phase Analysis for UROP Hypotheses (H1, H2, H3).

Segments articles by IPO phase based on article publication dates:
  - Baseline: before 2026-01-01
  - Rumor: 2026-01-01 to 2026-04-30
  - Confidential Filing: 2026-05-01 to 2026-06-30
  - Public S1: 2026-07-01 to 2026-10-31
  - Post-Listing: after 2026-11-01

Generates publication-quality plots:
  - data/plot_h1_sourcing.png       (H1: Company Citations Drop Post-Filing?)
  - data/plot_h2_sentiment.png      (H2: Financial Sentiment Trend Across IPO Phases)
  - data/plot_h3_risk_benefit.png   (H3: Risk/Benefit Ratio Increases During Silence?)

Usage:
    python src/14_phase_analysis.py
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
SENTIMENT_CSV = PROJECT_ROOT / "data" / "merged_sentiment_analysis.csv"
GDELT_CSV     = PROJECT_ROOT / "data" / "gdelt_news_2025_2026_us_deep_dedup.csv"

OUT_H1 = PROJECT_ROOT / "data" / "plot_h1_sourcing.png"
OUT_H2 = PROJECT_ROOT / "data" / "plot_h2_sentiment.png"
OUT_H3 = PROJECT_ROOT / "data" / "plot_h3_risk_benefit.png"

DPI = 300

# Chronological IPO phases
PHASE_DEFINITIONS = [
    ("Baseline", "Before 2026-01-01"),
    ("Rumor", "2026-01-01 to 2026-04-30"),
    ("Confidential Filing", "2026-05-01 to 2026-06-30"),
    ("Public S1", "2026-07-01 to 2026-10-31"),
    ("Post-Listing", "After 2026-11-01"),
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
    """Parse JSON array strings into Python lists safely."""
    if not isinstance(val, str) or not val.strip():
        return []
    try:
        parsed = json.loads(val)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


def assign_ipo_phase(date_val: pd.Timestamp) -> str | None:
    """Assign article date to its respective IPO phase."""
    if pd.isna(date_val):
        return None
    if date_val < pd.Timestamp("2026-01-01"):
        return "Baseline"
    elif date_val <= pd.Timestamp("2026-04-30"):
        return "Rumor"
    elif date_val <= pd.Timestamp("2026-06-30"):
        return "Confidential Filing"
    elif date_val <= pd.Timestamp("2026-10-31"):
        return "Public S1"
    else:
        return "Post-Listing"


# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------

def load_and_prepare_data() -> pd.DataFrame:
    """Load sentiment analysis data and join with GDELT publication dates."""
    print("Loading data files...")
    if not SENTIMENT_CSV.exists():
        raise FileNotFoundError(f"Missing sentiment CSV: {SENTIMENT_CSV}")
    if not GDELT_CSV.exists():
        raise FileNotFoundError(f"Missing GDELT CSV: {GDELT_CSV}")

    df_sent = pd.read_csv(SENTIMENT_CSV, low_memory=False)
    df_gdelt = pd.read_csv(GDELT_CSV, usecols=["news_url", "date"], low_memory=False)

    print(f"  Loaded sentiment rows: {len(df_sent):,}")
    print(f"  Loaded GDELT date rows: {len(df_gdelt):,}")

    # Deduplicate GDELT URLs before merging
    df_gdelt = df_gdelt.drop_duplicates(subset=["news_url"])

    # Merge on URL
    merged = pd.merge(
        df_sent,
        df_gdelt,
        left_on="url",
        right_on="news_url",
        how="inner",
    )
    print(f"  Intersection rows: {len(merged):,}")

    # Parse and validate dates
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    valid_df = merged.dropna(subset=["date"]).copy()
    print(f"  Rows with valid dates: {len(valid_df):,}")

    # Assign phase
    valid_df["phase"] = valid_df["date"].apply(assign_ipo_phase)

    # Parse NLP fields
    valid_df["sources_list"] = valid_df["sources"].apply(safe_parse_json_list)
    valid_df["risks_list"] = valid_df["risks"].apply(safe_parse_json_list)
    valid_df["benefits_list"] = valid_df["benefits"].apply(safe_parse_json_list)

    # 1. H1 Metric: Company quoted/cited
    valid_df["has_company_source"] = valid_df["sources_list"].apply(
        lambda sources: "Company" in sources
    )

    # 2. H2 Metric: Financial sentiment score (-1, 0, +1)
    sentiment_map = {"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0}
    valid_df["sentiment_score"] = valid_df["financial_sentiment"].map(sentiment_map)

    # 3. H3 Metric: Risk/Benefit ratio = len(risks) / (len(benefits) + 1)
    valid_df["risk_count"] = valid_df["risks_list"].apply(len)
    valid_df["benefit_count"] = valid_df["benefits_list"].apply(len)
    valid_df["risk_benefit_ratio"] = valid_df["risk_count"] / (valid_df["benefit_count"] + 1)

    return valid_df


# ---------------------------------------------------------------------------
# Phase Aggregations
# ---------------------------------------------------------------------------

def compute_phase_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics across defined IPO phases."""
    rows = []
    for phase_name, date_range in PHASE_DEFINITIONS:
        sub = df[df["phase"] == phase_name]
        n_articles = len(sub)

        if n_articles > 0:
            company_citation_pct = sub["has_company_source"].mean() * 100.0
            sentiment_score_avg  = sub["sentiment_score"].dropna().mean()
            risk_benefit_avg     = sub["risk_benefit_ratio"].mean()
            avg_risks            = sub["risk_count"].mean()
            avg_benefits         = sub["benefit_count"].mean()
        else:
            company_citation_pct = np.nan
            sentiment_score_avg  = np.nan
            risk_benefit_avg     = np.nan
            avg_risks            = np.nan
            avg_benefits         = np.nan

        rows.append({
            "phase": phase_name,
            "date_range": date_range,
            "article_count": n_articles,
            "company_citation_pct": company_citation_pct,
            "sentiment_score_avg": sentiment_score_avg,
            "risk_benefit_avg": risk_benefit_avg,
            "avg_risks": avg_risks,
            "avg_benefits": avg_benefits,
        })

    summary_df = pd.DataFrame(rows)
    return summary_df


def print_summary_table(summary_df: pd.DataFrame) -> None:
    """Print clean raw metrics table to stdout."""
    print("\n" + "=" * 90)
    print("PHASE ANALYSIS: RAW METRICS PER IPO PHASE")
    print("=" * 90)
    header = (
        f"{'Phase':<20} | {'Date Range':<25} | {'N':>6} | "
        f"{'Company Cit%':>12} | {'Sent Score':>10} | {'Risk/Ben':>9}"
    )
    print(header)
    print("-" * 90)
    for _, r in summary_df.iterrows():
        n = int(r["article_count"])
        if n > 0:
            cit_str = f"{r['company_citation_pct']:>11.1f}%"
            sent_str = f"{r['sentiment_score_avg']:>+10.3f}"
            rb_str = f"{r['risk_benefit_avg']:>9.3f}"
        else:
            cit_str = f"{'N/A':>12}"
            sent_str = f"{'N/A':>10}"
            rb_str = f"{'N/A':>9}"

        line = (
            f"{r['phase']:<20} | {r['date_range']:<25} | {n:>6,} | "
            f"{cit_str} | {sent_str} | {rb_str}"
        )
        print(line)
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Plot 1: H1 Sourcing Plot
# ---------------------------------------------------------------------------

def plot_h1(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Generate H1 plot: Company citation rate across IPO phases."""
    plot_df = summary_df[summary_df["article_count"] > 0].copy()
    phases = plot_df["phase"].tolist()
    counts = plot_df["article_count"].tolist()
    citations = plot_df["company_citation_pct"].tolist()

    x_labels = [f"{p}\n(N={n:,})" for p, n in zip(phases, counts)]
    x = np.arange(len(phases))

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=DPI)

    # Bar chart with trend line overlay
    bars = ax.bar(
        x,
        citations,
        color="#3498DB",
        edgecolor="#1B4F72",
        width=0.5,
        zorder=3,
        label="Company Citation Rate (%)",
    )

    ax.plot(
        x,
        citations,
        color="#E67E22",
        marker="o",
        markersize=8,
        linewidth=2.2,
        zorder=4,
        label="Trend Line",
    )

    # Annotate bar values
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#1B4F72",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10.5)
    ax.set_ylabel("Articles Citing Company Source (%)", fontsize=11, labelpad=10)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.xaxis.grid(False)

    plt.suptitle(
        "H1: Company Citations Drop Post-Filing?",
        fontsize=14,
        fontweight="bold",
        y=0.98,
        color="#1A252F",
    )
    ax.set_title(
        "Percentage of articles quoting or citing official company spokespeople across IPO phases",
        fontsize=10,
        color="#566573",
        pad=14,
    )

    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#D5D8DC", loc="lower right")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] H1 Sourcing Plot: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: H2 Financial Sentiment Plot
# ---------------------------------------------------------------------------

def plot_h2(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Generate H2 plot: Financial sentiment trend across IPO phases."""
    plot_df = summary_df[summary_df["article_count"] > 0].copy()
    phases = plot_df["phase"].tolist()
    counts = plot_df["article_count"].tolist()
    scores = plot_df["sentiment_score_avg"].tolist()

    x_labels = [f"{p}\n(N={n:,})" for p, n in zip(phases, counts)]
    x = np.arange(len(phases))

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=DPI)

    # Reference neutral zero line
    ax.axhline(0.0, color="#7F8C8D", linestyle="--", linewidth=1.2, alpha=0.8, label="Neutral Baseline (0.0)")

    # Line plot with markers
    ax.plot(
        x,
        scores,
        color="#27AE60",
        marker="s",
        markersize=9,
        linewidth=2.5,
        zorder=4,
        label="Mean Financial Sentiment",
    )

    # Fill between zero and score
    ax.fill_between(x, 0, scores, color="#2ECC71", alpha=0.15, zorder=2)

    # Annotate score values
    for xi, yi in zip(x, scores):
        ax.annotate(
            f"{yi:+.3f}",
            xy=(xi, yi),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#145A32",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10.5)
    ax.set_ylabel("Financial Sentiment Score (-1 to +1)", fontsize=11, labelpad=10)
    ax.set_ylim(-0.2, 0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.xaxis.grid(False)

    plt.suptitle(
        "H2: Financial Sentiment Trend Across IPO Phases",
        fontsize=14,
        fontweight="bold",
        y=0.98,
        color="#1A252F",
    )
    ax.set_title(
        "Gemini Financial Sentiment score mapping: Negative=-1.0, Neutral=0.0, Positive=+1.0",
        fontsize=10,
        color="#566573",
        pad=14,
    )

    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#D5D8DC", loc="lower left")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] H2 Sentiment Plot: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: H3 Risk/Benefit Ratio Plot
# ---------------------------------------------------------------------------

def plot_h3(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Generate H3 plot: Risk/Benefit ratio trend across IPO phases."""
    plot_df = summary_df[summary_df["article_count"] > 0].copy()
    phases = plot_df["phase"].tolist()
    counts = plot_df["article_count"].tolist()
    ratios = plot_df["risk_benefit_avg"].tolist()

    x_labels = [f"{p}\n(N={n:,})" for p, n in zip(phases, counts)]
    x = np.arange(len(phases))

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=DPI)

    # Parity line at 1.0
    ax.axhline(1.0, color="#95A5A6", linestyle="--", linewidth=1.2, alpha=0.85, label="Risk/Benefit Parity (1.0)")

    # Line plot with markers
    ax.plot(
        x,
        ratios,
        color="#C0392B",
        marker="D",
        markersize=8,
        linewidth=2.5,
        zorder=4,
        label="Avg Risk/Benefit Ratio",
    )

    # Annotate ratio values
    for xi, yi in zip(x, ratios):
        ax.annotate(
            f"{yi:.3f}x",
            xy=(xi, yi),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#922B21",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10.5)
    ax.set_ylabel("Risk / (Benefit + 1) Ratio", fontsize=11, labelpad=10)
    ax.set_ylim(0.7, 1.45)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.xaxis.grid(False)

    plt.suptitle(
        "H3: Risk/Benefit Ratio Increases During Silence?",
        fontsize=14,
        fontweight="bold",
        y=0.98,
        color="#1A252F",
    )
    ax.set_title(
        "Average Risk-to-Benefit framing ratio across chronological IPO phases",
        fontsize=10,
        color="#566573",
        pad=14,
    )

    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#D5D8DC", loc="lower right")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] H3 Risk/Benefit Plot: {out_path}")


# ---------------------------------------------------------------------------
# Main CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 75)
    print("IPO PHASE ANALYSIS: TESTING HYPOTHESES H1, H2, H3")
    print("=" * 75)

    df = load_and_prepare_data()
    summary_df = compute_phase_metrics(df)

    print_summary_table(summary_df)

    print("Generating hypothesis visualization plots...")
    plot_h1(summary_df, OUT_H1)
    plot_h2(summary_df, OUT_H2)
    plot_h3(summary_df, OUT_H3)

    print("\nPhase analysis complete. All plots saved to data/ directory.")
    print("=" * 75)


if __name__ == "__main__":
    main()
