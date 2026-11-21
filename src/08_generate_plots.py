#!/usr/bin/env python3
"""
Week 2 Presentation Visualizations.

Generates two publication-quality plots from the GDELT US news dataset:

  Plot 1 — Bar chart: article count per company (OpenAI, Anthropic, SpaceX).
            An article mentioning multiple companies is counted once per company.
            Saved to: data/plot_articles_per_company.png

  Plot 2 — Line chart: total article volume per month with vertical dashed
            lines marking key IPO milestones.
            Saved to: data/plot_articles_over_time.png

Usage:
    python src/08_generate_plots.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV    = PROJECT_ROOT / "data" / "gdelt_news_2025_2026_us_deep_dedup.csv"
OUT_BAR      = PROJECT_ROOT / "data" / "plot_articles_per_company.png"
OUT_LINE     = PROJECT_ROOT / "data" / "plot_articles_over_time.png"

# ---------------------------------------------------------------------------
# Companies to track and their keyword variants
# ---------------------------------------------------------------------------

COMPANIES = {
    "OpenAI":    ["openai", "chatgpt", "Altman"],
    "Anthropic": ["anthropic", "claude", "Amodei"],
    "SpaceX":    ["spacex", "Musk", "Grok"],
}

# ---------------------------------------------------------------------------
# IPO milestones for Plot 2
# ---------------------------------------------------------------------------

MILESTONES = [
    {"date": "2026-05-20", "label": "SpaceX S-1\npublic",           "color": "#e07b39"},
    {"date": "2026-06-01", "label": "Anthropic\nconfidential S-1",  "color": "#6c5ce7"},
    {"date": "2026-06-08", "label": "OpenAI\nconfidential S-1",     "color": "#00b894"},
    {"date": "2026-10-26", "label": "Anthropic S-1\npublic",        "color": "#d63031"},
]

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

COMPANY_COLORS = {
    "OpenAI":    "#00b894",
    "Anthropic": "#6c5ce7",
    "SpaceX":    "#e07b39",
}

DPI = 300

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    # Fallback for older matplotlib versions
    plt.style.use("seaborn-whitegrid")


# ---------------------------------------------------------------------------
# 1. Load & clean data
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    """Load CSV, parse 'date' column, drop rows with unparseable dates."""
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Rows loaded: {len(df):,}")

    # Coerce 'date' to datetime; invalid entries become NaT and are dropped
    if "date" not in df.columns:
        raise ValueError("'date' column not found in dataset.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["date"])
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped:,} rows with invalid/missing dates.")

    # Ensure text column is a clean string
    text_col = "news_content" if "news_content" in df.columns else "title"
    if text_col not in df.columns:
        raise ValueError(f"Expected text column '{text_col}' not found.")
    df["_text"] = df[text_col].fillna("").str.lower()

    # Add month period for aggregation
    df["month"] = df["date"].dt.to_period("M")

    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Valid rows : {len(df):,}\n")
    return df


# ---------------------------------------------------------------------------
# 2. Company article counts
# ---------------------------------------------------------------------------

def count_per_company(df: pd.DataFrame) -> dict[str, int]:
    """
    For each company, count articles whose text contains at least one keyword.
    An article can be counted for multiple companies.
    """
    counts = {}
    for company, keywords in COMPANIES.items():
        pattern = "|".join(keywords)
        mask = df["_text"].str.contains(pattern, regex=True, na=False)
        counts[company] = int(mask.sum())
        print(f"  {company:<12}: {counts[company]:>6,} articles")
    return counts


# ---------------------------------------------------------------------------
# Plot 1 — Bar chart: articles per company
# ---------------------------------------------------------------------------

def plot_bar(counts: dict[str, int], out_path: Path) -> None:
    companies = list(counts.keys())
    values    = [counts[c] for c in companies]
    colors    = [COMPANY_COLORS[c] for c in companies]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(companies, values, color=colors, width=0.5, edgecolor="white", linewidth=1.2)

    # Value labels above bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.015,
            f"{val:,}",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#2d3436",
        )

    ax.set_title(
        "AI IPO Coverage: Article Count per Company\n(GDELT US News, Dec 2025 – Nov 2026)",
        fontsize=14, fontweight="bold", pad=16,
    )
    ax.set_xlabel("Company", fontsize=12, labelpad=8)
    ax.set_ylabel("Number of Articles", fontsize=12, labelpad=8)
    ax.set_ylim(0, max(values) * 1.15)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(axis="both", labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — Line chart: total articles over time with IPO milestones
# ---------------------------------------------------------------------------

def plot_timeline(df: pd.DataFrame, out_path: Path) -> None:
    # 1. Calcola il conteggio settimanale
    weekly_counts = df.set_index("date").resample("W").size()
    
    # 2. Applica una media mobile a 4 settimane (circa un mese) per lisciare i buchi
    smoothed_counts = weekly_counts.rolling(window=4, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(13, 5))

    # 3. Disegna la linea morbida
    ax.plot(
        smoothed_counts.index, smoothed_counts.values,
        color="#2980b9", linewidth=2.5,
        label="4-Week Rolling Average (Smoothed)",
    )
    ax.fill_between(
        smoothed_counts.index, smoothed_counts.values,
        alpha=0.15, color="#2980b9",
    )

    # Milestones (etichette scalate)
    label_offsets = {
        "SpaceX S-1\npublic": 0.97, 
        "Anthropic\nconfidential S-1": 0.84, 
        "OpenAI\nconfidential S-1": 0.71, 
        "Anthropic S-1\npublic": 0.97
    }
    y_max = smoothed_counts.max()
    
    for ms in MILESTONES:
        ms_date = pd.Timestamp(ms["date"])
        ax.axvline(ms_date, color=ms["color"], linestyle="--", linewidth=1.5, alpha=0.85)
        y_offset = label_offsets.get(ms["label"], 0.97)
        ax.text(
            ms_date, y_max * y_offset, ms["label"],
            color=ms["color"], fontsize=8, fontweight="bold",
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=ms["color"], alpha=0.88),
        )

    ax.set_title(
        "AI IPO News Volume Over Time (Smoothed)\n(GDELT US Articles, Dec 2025 – Nov 2026)",
        fontsize=14, fontweight="bold", pad=16,
    )
    ax.set_xlabel("Date", fontsize=12, labelpad=8)
    ax.set_ylabel("Smoothed Weekly Volume", fontsize=12, labelpad=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(axis="both", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(fontsize=9, loc="upper left", framealpha=0.92, edgecolor="#cccccc")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("WEEK 2 PRESENTATION — PLOT GENERATION")
    print("=" * 65)

    df = load_data(INPUT_CSV)

    # --- Plot 1: bar chart ---
    print("Plot 1: article count per company")
    counts = count_per_company(df)
    plot_bar(counts, OUT_BAR)

    # --- Plot 2: timeline ---
    print("\nPlot 2: monthly article volume + IPO milestones")
    plot_timeline(df, OUT_LINE)

    print("\nDone. Both plots saved to data/")
    print("=" * 65)


if __name__ == "__main__":
    main()

