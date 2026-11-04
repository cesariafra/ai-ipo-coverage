#!/usr/bin/env python3
"""
GDELT News Article Fetcher (2025-2026).

This script supplements historical news datasets with recent news articles
via the GDELT 2.0 DOC API (https://api.gdeltproject.org/api/v2/doc/doc).

Key features:
1. Full Date Storage: Extracts the full date (YYYY-MM-DD) from the GDELT `seendate`
   field and stores it in a dedicated 'date' column, while keeping 'year'.
2. Time Windowing: Iterates across the target date range in 7-day chunks.
3. Dynamic Granularity: If a 7-day window hits the 250 article limit, it
   saves the initial 250 articles immediately, then dynamically queries 1-day
   sub-windows (with a 5-second sleep) to capture any additional articles.
4. Fail-Fast Rate Limiting: Sleeps 3 seconds between requests, tries at most 2
   attempts per window, and skips failed windows quickly.
5. Enrichment Mode (--enrich): Matches fetched articles against existing CSV rows
   by news_url to fill in missing/empty dates in-place and appends new articles.
6. Dry-Run Mode (--dry-run): Previews enrichment and fetch counts without writing to disk.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None


# ---------------------------------------------------------------------------
# Default Configuration Constants
# ---------------------------------------------------------------------------

GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

DEFAULT_START_DATE = "2025-12-20"
DEFAULT_END_DATE = "2026-11-02"

DEFAULT_QUERY = (
    '("OpenAI" OR "Anthropic" OR "SpaceX") ("Artificial Intelligence" OR "ChatGPT" OR "Claude" OR "Generative AI" OR "GenAI" OR "Machine Learning" OR "Neural Network" OR "Large Language Model")'
)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

OUTPUT_COLUMNS = ["news_url", "news_content", "country", "year", "date"]
US_COUNTRY_VARIANTS = {'us', 'usa', 'united states', 'united states of america'}

RATE_LIMIT_SLEEP = 3.0   # Mandatory pause (seconds) between requests (Fail-Fast mode)
RETRY_SLEEP = 3.0        # Pause (seconds) before single retry
MAX_RETRIES = 2          # Maximum retries per window (try once, retry once, then fail fast)
GDELT_MAX_RECORDS = 250  # Hard limit per call in GDELT DOC API


# ---------------------------------------------------------------------------
# Article Formatting & Helper Functions
# ---------------------------------------------------------------------------

def format_article(article: dict) -> dict:
    """
    Format a raw GDELT JSON article object into the schema matching our dataset:
      - news_url: Article URL
      - news_content: Title + ' | ' + domain (GDELT API does not supply full text)
      - country: Source country lowercased
      - year: First 4 characters of seendate (e.g., '2026')
      - date: Full YYYY-MM-DD date derived from seendate[:8] (e.g., '2026-02-15')
    """
    url = (article.get("url") or "").strip()
    title = (article.get("title") or "").strip()
    domain = (article.get("domain") or "").strip()
    sourcecountry = (article.get("sourcecountry") or "").strip()
    seendate = str(article.get("seendate") or "").strip()

    # Construct composite content since raw body is not available via GDELT API
    if domain:
        news_content = f"{title} | {domain}" if title else domain
    else:
        news_content = title

    # Lowercase country code
    country = sourcecountry.lower()

    # Extract 4-digit year and full YYYY-MM-DD date from seendate
    year = seendate[:4] if len(seendate) >= 4 and seendate[:4].isdigit() else ""
    date = (
        f"{seendate[:4]}-{seendate[4:6]}-{seendate[6:8]}"
        if len(seendate) >= 8 and seendate[:8].isdigit()
        else ""
    )

    return {
        "news_url": url,
        "news_content": news_content,
        "country": country,
        "year": year,
        "date": date,
    }


def load_existing_dataset(csv_path: Path):
    """
    Load existing CSV into a list of row dicts and a URL index mapping: {news_url: row_index}.
    Ensures 'date' column is present in all loaded row dictionaries.
    """
    rows = []
    url_to_index = {}
    if not csv_path.exists():
        return rows, url_to_index

    with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if "date" not in row:
                row["date"] = ""
            u = (row.get("news_url") or "").strip()
            rows.append(row)
            if u and u not in url_to_index:
                url_to_index[u] = idx

    return rows, url_to_index


def save_articles_to_csv(rows: list, output_path: Path, file_exists: bool, dry_run: bool = False) -> bool:
    """
    Incrementally append rows to the CSV file. If the file does not exist,
    create it and write the header first.
    Returns True indicating the file now exists.
    """
    if not rows or dry_run:
        return file_exists

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not file_exists or not output_path.exists()

    with open(output_path, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    return True


# ---------------------------------------------------------------------------
# API Request & Retry Logic
# ---------------------------------------------------------------------------

def fetch_gdelt_window(
    start_dt: datetime,
    end_dt: datetime,
    query: str,
    max_retries: int = MAX_RETRIES,
    rate_limit_sleep: float = RATE_LIMIT_SLEEP,
    retry_sleep: float = RETRY_SLEEP,
) -> list:
    """
    Fetch articles for a specific [start_dt, end_dt] window using requests.

    Fail-Fast Error Handling & Retry Logic:
      - Formats timestamps to YYYYMMDDHHMMSS as required by GDELT DOC API.
      - Enforces a 3-second sleep after every attempt to maximize throughput.
      - If the first request fails (e.g. rate limit non-JSON), waits 3 seconds and tries ONE more time.
      - If it fails again, skips the window and logs the missing dates to patch later.
    """
    if requests is None:
        raise ImportError("The 'requests' package is required. Install with: pip install requests")

    # Format timestamps for GDELT (UTC YYYYMMDDHHMMSS)
    # End datetime is set to 23:59:59 of the final second to avoid overlapping boundaries
    start_str = start_dt.strftime("%Y%m%d%H%M%S")
    end_str = (end_dt - timedelta(seconds=1)).strftime("%Y%m%d%H%M%S")

    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "startdatetime": start_str,
        "enddatetime": end_str,
        "maxrecords": GDELT_MAX_RECORDS,
    }

    headers = {
        "User-Agent": DEFAULT_USER_AGENT
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                GDELT_ENDPOINT,
                params=params,
                headers=headers,
                timeout=30,
            )

            # Check if response body is empty (common when 0 matches exist)
            text_body = response.text.strip()
            if not text_body:
                time.sleep(rate_limit_sleep)
                return []

            # GDELT returns plain text/HTML on rate limits or errors
            # Attempt to decode as JSON
            try:
                data = response.json()
                articles = data.get("articles", [])

                # Successful request: wait mandatory rate-limit sleep before returning
                time.sleep(rate_limit_sleep)
                return articles

            except (json.JSONDecodeError, ValueError):
                # Rate limit plain text response detected
                preview = text_body[:100].replace("\n", " ")
                print(f"      [Warning] Non-JSON response on attempt {attempt}/{max_retries}: '{preview}'")

        except requests.RequestException as req_err:
            print(f"      [Warning] Network error on attempt {attempt}/{max_retries}: {req_err}")

        # If attempt failed, wait 3s and try ONE more time (MAX_RETRIES = 2)
        if attempt < max_retries:
            print(f"      [Retry] Sleeping {retry_sleep:.0f}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(retry_sleep)
        else:
            print(f"  [Skipped] Window {start_dt.date()} - {end_dt.date()}. We will patch this missing date later.")
            time.sleep(rate_limit_sleep)
            return []

    return []


# ---------------------------------------------------------------------------
# Main Time-Windowing Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    start_date_str: str = DEFAULT_START_DATE,
    end_date_str: str = DEFAULT_END_DATE,
    query: str = DEFAULT_QUERY,
    output_path: Path = None,
    chunk_days: int = 7,
    enrich: bool = False,
    dry_run: bool = False,
):
    """
    Execute time-windowing loop from start_date to end_date:
      1. Slices date range into chunk_days (7-day) windows.
      2. If a 7-day window returns exactly 250 articles (the API limit),
         immediately processes the initial 250 articles, then queries 1-day sub-windows.
      3. In enrichment mode (--enrich): updates missing dates in-place and appends new rows.
         Otherwise, appends newly discovered rows with deduplication by URL.
      4. In dry-run mode (--dry-run): previews counts without writing to disk.
    """
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")

    if output_path is None:
        project_root = Path(__file__).resolve().parent.parent
        output_path = project_root / "data" / "gdelt_news_2025_2026.csv"

    print("=" * 75)
    print("STARTING GDELT NEWS COLLECTION (2025-2026)")
    print("=" * 75)
    print(f"Target Date Range: {start_dt.date()} to {end_dt.date()}")
    print(f"Default Window:    {chunk_days} days (with dynamic 1-day fallback on 250 cap)")
    print(f"Output File:       {output_path}")
    print(f"Query:             {query}")
    print(f"Enrichment Mode:   {'ACTIVE (--enrich)' if enrich else 'Standard (append only)'}")
    print(f"Dry-Run Mode:      {'ACTIVE (--dry-run, no file writes)' if dry_run else 'Inactive'}")
    print("[!] US-ONLY FILTER ACTIVE: Non-US articles will be discarded.")
    print("=" * 75)

    file_exists = output_path.exists()

    # -----------------------------------------------------------------------
    # Setup State for Enrichment vs Standard Mode
    # -----------------------------------------------------------------------
    if enrich:
        existing_rows, url_to_index = load_existing_dataset(output_path)
        print(f"Loaded {len(existing_rows):,} existing rows from {output_path.name} for enrichment.\n")
        enriched_count = 0
        new_count = 0
        skipped_count = 0

        def process_articles(articles_list):
            nonlocal enriched_count, new_count, skipped_count
            w_enriched = 0
            w_new = 0
            w_skipped = 0
            for art in articles_list:
                # Enforce US-only filter: skip non-US articles
                country = (art.get("sourcecountry") or "").strip().lower()
                if country not in US_COUNTRY_VARIANTS:
                    continue

                formatted = format_article(art)
                url = formatted["news_url"]
                if not url:
                    continue

                if url in url_to_index:
                    idx = url_to_index[url]
                    row = existing_rows[idx]
                    existing_date = (row.get("date") or "").strip()
                    new_date = formatted["date"]
                    # If date is missing/empty and we have a new date, enrich in-place
                    if not existing_date and new_date:
                        row["date"] = new_date
                        enriched_count += 1
                        w_enriched += 1
                    else:
                        skipped_count += 1
                        w_skipped += 1
                else:
                    # New article not in existing CSV
                    url_to_index[url] = len(existing_rows)
                    existing_rows.append(formatted)
                    new_count += 1
                    w_new += 1

            return w_enriched, w_new, w_skipped

    else:
        # Standard mode: track seen URLs to avoid duplicate additions
        seen_urls = set()
        if file_exists:
            try:
                with open(output_path, mode="r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    existing_fieldnames = reader.fieldnames or []
                    existing_records = list(reader)

                for r in existing_records:
                    u = r.get("news_url")
                    if u:
                        seen_urls.add(u)

                # Upgrade existing CSV header to include 'date' if absent
                if "date" not in existing_fieldnames and not dry_run:
                    for r in existing_records:
                        if "date" not in r:
                            r["date"] = ""
                    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
                        writer.writeheader()
                        writer.writerows(existing_records)

                print(f"Loaded {len(seen_urls):,} existing URLs from {output_path.name} to avoid duplicates.\n")
            except Exception as e:
                print(f"Note: Could not preload existing URLs: {e}\n")

        cumulative_saved = len(seen_urls)

        def process_articles(articles_list):
            nonlocal cumulative_saved, file_exists
            new_rows = []
            for art in articles_list:
                # Enforce US-only filter: skip non-US articles
                country = (art.get("sourcecountry") or "").strip().lower()
                if country not in US_COUNTRY_VARIANTS:
                    continue

                url = (art.get("url") or "").strip()
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    new_rows.append(format_article(art))

            if new_rows and not dry_run:
                file_exists = save_articles_to_csv(new_rows, output_path, file_exists, dry_run=dry_run)
            cumulative_saved += len(new_rows)
            return len(new_rows)

    current_start = start_dt

    # -----------------------------------------------------------------------
    # Outer Time-Windowing Loop
    # -----------------------------------------------------------------------
    while current_start < end_dt:
        window_end = min(current_start + timedelta(days=chunk_days), end_dt)
        window_label = f"{current_start.date()} -> {window_end.date()}"

        print(f"\n[Window] Fetching {window_label} ({chunk_days}-day chunk)...")
        articles = fetch_gdelt_window(current_start, window_end, query=query)
        fetched_count = len(articles)

        # -------------------------------------------------------------------
        # Dynamic Window Reduction Logic:
        # If the window returned exactly 250 articles and is larger than 1 day,
        # it hit GDELT's maximum record limit. Save initial 250 articles first,
        # then subdivide into 1-day sub-windows to capture additional articles.
        # -------------------------------------------------------------------
        if fetched_count == GDELT_MAX_RECORDS and (window_end - current_start) > timedelta(days=1):
            print(f"  [!] Window {window_label} hit API limit of 250 articles.")
            print("      Step 1: Processing initial 250 weekly articles immediately to prevent data loss...")

            if enrich:
                w_enr, w_new, w_skip = process_articles(articles)
                print(
                    f"      -> Processed initial batch: {w_enr:3d} enriched, {w_new:3d} new, {w_skip:3d} skipped | "
                    f"Cumulative: {enriched_count:5,d} enriched, {new_count:5,d} new"
                )
            else:
                saved_count = process_articles(articles)
                print(
                    f"      -> Saved initial batch: {saved_count:3d} unique articles | "
                    f"Cumulative saved: {cumulative_saved:5,d}"
                )

            # Step 2: Proceed with 1-day sub-windows
            print("      Step 2: Subdividing into 1-day sub-windows to fetch any additional articles...")
            sub_start = current_start
            while sub_start < window_end:
                sub_end = min(sub_start + timedelta(days=1), window_end)
                sub_label = f"{sub_start.date()} -> {sub_end.date()}"

                # Pause at least 5 seconds before each 1-day request to give GDELT time to breathe
                time.sleep(5.0)

                print(f"    [1-Day Sub-Window] Fetching {sub_label}...")
                sub_articles = fetch_gdelt_window(sub_start, sub_end, query=query)

                if enrich:
                    s_enr, s_new, s_skip = process_articles(sub_articles)
                    print(
                        f"    -> Fetched: {len(sub_articles):3d} | "
                        f"Enriched: {s_enr:3d} | New: {s_new:3d} | Skipped: {s_skip:3d} | "
                        f"Cumulative: {enriched_count:5,d} enriched, {new_count:5,d} new"
                    )
                else:
                    sub_saved = process_articles(sub_articles)
                    print(
                        f"    -> Fetched: {len(sub_articles):3d} | "
                        f"New additional unique: {sub_saved:3d} | "
                        f"Cumulative saved: {cumulative_saved:5,d}"
                    )

                sub_start = sub_end

            current_start = window_end

        else:
            # Window returned fewer than 250 records (no truncation)
            if enrich:
                w_enr, w_new, w_skip = process_articles(articles)
                print(
                    f"  -> Fetched: {fetched_count:3d} | "
                    f"Enriched: {w_enr:3d} | New: {w_new:3d} | Skipped: {w_skip:3d} | "
                    f"Cumulative: {enriched_count:5,d} enriched, {new_count:5,d} new"
                )
            else:
                new_saved = process_articles(articles)
                print(
                    f"  -> Fetched: {fetched_count:3d} | "
                    f"New unique: {new_saved:3d} | "
                    f"Cumulative saved: {cumulative_saved:5,d}"
                )

            current_start = window_end

    # -----------------------------------------------------------------------
    # Final Output Summary & Writing
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("GDELT DATA COLLECTION COMPLETE")
    print("=" * 75)

    if enrich:
        print(f"Mode:                        ENRICHMENT (--enrich)")
        print(f"Total Rows in Dataset:       {len(existing_rows):,}")
        print(f"Rows Enriched with Date:     {enriched_count:,}")
        print(f"New Rows Added:              {new_count:,}")
        print(f"Rows Skipped (up to date):   {skipped_count:,}")

        if dry_run:
            print("\n[Dry-Run] Preview completed. No files were written to disk.")
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
                writer.writeheader()
                writer.writerows(existing_rows)
            print(f"Destination File Saved:      {output_path}")

    else:
        print(f"Mode:                        STANDARD (Append only)")
        print(f"Total Unique Articles Saved: {cumulative_saved:,}")
        if dry_run:
            print("[Dry-Run] Preview completed. No files were written to disk.")
        else:
            print(f"Destination File:            {output_path}")

    print("=" * 75)


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch news articles from GDELT 2.0 DOC API with dynamic time-windowing and enrichment."
    )
    project_root = Path(__file__).resolve().parent.parent
    default_output = project_root / "data" / "gdelt_news_2025_2026.csv"

    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE,
        help=f"Start date (YYYY-MM-DD), default: {DEFAULT_START_DATE}",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=DEFAULT_END_DATE,
        help=f"End date (YYYY-MM-DD), default: {DEFAULT_END_DATE}",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help=f"GDELT search query, default: {DEFAULT_QUERY}",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=default_output,
        help=f"Output CSV path, default: {default_output}",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=7,
        help="Initial window size in days (default: 7)",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Enrich existing CSV rows with missing dates in-place and append new rows",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview enrichment/fetch counts without modifying or writing to disk",
    )

    args = parser.parse_args()

    run_pipeline(
        start_date_str=args.start_date,
        end_date_str=args.end_date,
        query=args.query,
        output_path=args.output,
        chunk_days=args.chunk_days,
        enrich=args.enrich,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
