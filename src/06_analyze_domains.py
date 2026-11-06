#!/usr/bin/env python3
"""
Domain Distribution Analyzer.

Reads data/gdelt_news_2025_2026_us_deep_dedup.csv, extracts the domain
from each news_url, counts article frequency per domain, prints the top 50,
and saves the full distribution to data/domain_distribution.csv.

Usage:
    python src/06_analyze_domains.py
"""

from pathlib import Path
from urllib.parse import urlparse
from collections import Counter

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH   = PROJECT_ROOT / "data" / "gdelt_news_2025_2026_us_deep_dedup.csv"
OUTPUT_PATH  = PROJECT_ROOT / "data" / "domain_distribution.csv"

TOP_N = 50  # Number of top domains to print


# ---------------------------------------------------------------------------
# Domain Extraction
# ---------------------------------------------------------------------------

def extract_domain(url: str) -> str:
    """
    Extract the bare domain (without 'www.' prefix) from a URL.

    Examples:
        https://www.reuters.com/article/...  →  reuters.com
        http://apnews.com/article/...        →  apnews.com
    """
    try:
        hostname = urlparse(url).netloc  # e.g. 'www.reuters.com'
        # Strip leading 'www.' for cleaner grouping
        if hostname.startswith("www."):
            hostname = hostname[4:]
        return hostname.lower() if hostname else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("DOMAIN DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # 1. Load dataset
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    total_articles = len(df)
    print(f"Total articles loaded: {total_articles:,}")

    # 2. Extract domain from each URL
    df["domain"] = df["news_url"].fillna("").apply(extract_domain)

    # 3. Count articles per domain and sort descending
    domain_counts = (
        df["domain"]
        .value_counts()
        .reset_index()
    )
    domain_counts.columns = ["domain", "article_count"]
    domain_counts["pct_of_total"] = (
        domain_counts["article_count"] / total_articles * 100
    ).round(2)

    total_domains = len(domain_counts)
    print(f"Unique domains found:  {total_domains:,}\n")

    # 4. Print top 50 domains
    print(f"{'RANK':<5} {'DOMAIN':<45} {'COUNT':>7}  {'%':>6}")
    print("-" * 70)
    for rank, row in enumerate(domain_counts.head(TOP_N).itertuples(), start=1):
        print(f"{rank:<5} {row.domain:<45} {row.article_count:>7,}  {row.pct_of_total:>5.2f}%")

    # 5. Save full list to CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    domain_counts.to_csv(OUTPUT_PATH, index=False)
    print("-" * 70)
    print(f"\nFull domain list ({total_domains:,} domains) saved to: {OUTPUT_PATH}")
    print(f"Top domain covers {domain_counts.iloc[0]['pct_of_total']:.2f}% of all articles.")
    print("=" * 70)


if __name__ == "__main__":
    main()
