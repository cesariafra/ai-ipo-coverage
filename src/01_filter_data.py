#!/usr/bin/env python3
"""
Filter Global News Dataset for AI IPO Coverage (Version 2).

This script processes large news CSV files (e.g. data/global_news_private.csv)
using pandas chunking (default chunksize=100,000) to keep memory usage low.

Key features:
1. US Country Filtering: Filters to keep only articles from the US ('US', 'USA', 'United States').
2. Expanded AI Keywords & Relaxed Matching:
   - Target Companies: OpenAI, Anthropic, SpaceX.
   - Expanded AI Keywords: AI, Artificial Intelligence, LLM(s), Large Language Model(s),
     ChatGPT, Claude, Generative AI, GenAI, Sam Altman, Dario Amodei, Elon Musk.
   - Matching rule: If the text mentions a target company, an AI keyword is not strictly
     required; rows matching either a target company or an expanded AI keyword are kept.
3. Date Extraction from URL:
   - Regex function extracts date (YYYY-MM-DD or YYYY/MM/DD) from news_url.
   - Standardizes to YYYY-MM-DD in a new 'date' column.
   - Falls back to the 'year' column if no date is found in the URL.
4. Output & Reporting:
   - Saves filtered rows to data/filtered_companies_v2.csv.
   - Prints country distribution before filtering, final dataset shape, and min/max date.
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None


# ---------------------------------------------------------------------------
# Configuration & Keywords
# ---------------------------------------------------------------------------

DEFAULT_CHUNKSIZE = 100_000

TARGET_COMPANIES = [
    'OpenAI',
    'Anthropic',
    'SpaceX',
]

AI_KEYWORDS = [
    'AI',
    'Artificial Intelligence',
    'LLM',
    'LLMs',
    'Large Language Model',
    'Large Language Models',
    'ChatGPT',
    'Claude',
    'Generative AI',
    'GenAI',
    'Sam Altman',
    'Dario Amodei',
    'Elon Musk',
]

US_COUNTRY_VARIANTS = {'us', 'usa', 'united states', 'united states of america'}

URL_DATE_REGEX = re.compile(
    r'\b((?:19|20)\d{2}[/-](?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01]))\b'
)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def build_regex_pattern(terms):
    """
    Build a compiled regular expression with word boundaries (\\b)
    to match terms case-insensitively. Longer terms are ordered first
    so compound phrases match before shorter subterms.
    """
    sorted_terms = sorted(terms, key=len, reverse=True)
    escaped_terms = [re.escape(term) for term in sorted_terms]
    pattern_str = r'\b(?:' + '|'.join(escaped_terms) + r')\b'
    return re.compile(pattern_str, re.IGNORECASE)


def extract_date_from_url(url, fallback_year=""):
    """
    Extract date (YYYY-MM-DD or YYYY/MM/DD) from news_url using regex.
    Normalizes the extracted date to YYYY-MM-DD.
    If no date is found in the URL, falls back to fallback_year.
    """
    if isinstance(url, str):
        match = URL_DATE_REGEX.search(url)
        if match:
            return match.group(1).replace('/', '-')

    if fallback_year is not None:
        y_str = str(fallback_year).strip()
        # Clean up float representations like 2023.0 -> 2023
        if y_str.endswith('.0'):
            y_str = y_str[:-2]
        return y_str

    return ""


def detect_columns(columns, preferred_date_col=None, preferred_text_col=None, preferred_country_col=None, preferred_url_col=None):
    """
    Detect relevant columns (date/year, text, country, url) from available headers.
    """
    col_map = {c.strip().lower(): c for c in columns}

    # Date / Year column
    date_col = preferred_date_col if preferred_date_col in columns else None
    if not date_col:
        for cand in ['year', 'date', 'seendate', 'publish_date', 'datetime']:
            if cand in col_map:
                date_col = col_map[cand]
                break

    # Text / Content column
    text_col = preferred_text_col if preferred_text_col in columns else None
    if not text_col:
        for cand in ['news_content', 'body', 'text', 'article_text', 'content', 'title', 'headline']:
            if cand in col_map:
                text_col = col_map[cand]
                break

    # Country column
    country_col = preferred_country_col if preferred_country_col in columns else None
    if not country_col:
        for cand in ['country', 'sourcecountry', 'country_code', 'nation', 'region']:
            if cand in col_map:
                country_col = col_map[cand]
                break

    # URL column
    url_col = preferred_url_col if preferred_url_col in columns else None
    if not url_col:
        for cand in ['news_url', 'url', 'documentidentifier', 'link']:
            if cand in col_map:
                url_col = col_map[cand]
                break

    return date_col, text_col, country_col, url_col


# ---------------------------------------------------------------------------
# Processing Pipeline (pandas implementation)
# ---------------------------------------------------------------------------

def process_file_pandas(
    input_path: Path,
    output_path: Path,
    chunksize: int = DEFAULT_CHUNKSIZE,
    preferred_date_col: str = None,
    preferred_text_col: str = None,
    preferred_country_col: str = None,
    preferred_url_col: str = None,
):
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    company_pattern = build_regex_pattern(TARGET_COMPANIES)
    ai_pattern = build_regex_pattern(AI_KEYWORDS)

    print("=" * 70)
    print("STARTING DATA FILTERING (v2)")
    print("=" * 70)
    print(f"Input file:       {input_path}")
    print(f"Output file:      {output_path}")
    print(f"Chunk size:       {chunksize:,} rows")
    print(f"Target companies: {TARGET_COMPANIES}")
    print(f"AI keywords:      {AI_KEYWORDS}")
    print(f"US country filter:{sorted(US_COUNTRY_VARIANTS)}")
    print("=" * 70)

    total_rows_scanned = 0
    total_rows_saved = 0
    country_counter = Counter()
    first_write = True

    date_col = None
    text_col = None
    country_col = None
    url_col = None

    chunk_iterator = pd.read_csv(
        input_path,
        chunksize=chunksize,
        low_memory=False,
        encoding='utf-8',
        on_bad_lines='skip',
    )

    for chunk_idx, chunk in enumerate(chunk_iterator, start=1):
        if chunk_idx == 1:
            date_col, text_col, country_col, url_col = detect_columns(
                chunk.columns,
                preferred_date_col=preferred_date_col,
                preferred_text_col=preferred_text_col,
                preferred_country_col=preferred_country_col,
                preferred_url_col=preferred_url_col,
            )

            print("\n" + "-" * 70)
            print("STEP 1: INSPECTING FIRST CHUNK & DETECTING COLUMNS")
            print("-" * 70)
            print(f"Available columns ({len(chunk.columns)}): {list(chunk.columns)}")
            print(f"Detected country column: '{country_col}'")
            print(f"Detected text column:    '{text_col}'")
            print(f"Detected URL column:     '{url_col}'")
            print(f"Detected year/date col:  '{date_col}'")
            print("-" * 70 + "\n")

            if text_col is None:
                raise ValueError(f"Could not detect text column from: {list(chunk.columns)}")

        chunk_rows = len(chunk)
        total_rows_scanned += chunk_rows

        # Track country distribution before filtering
        if country_col and country_col in chunk.columns:
            cleaned_countries = chunk[country_col].fillna("UNKNOWN").astype(str).str.strip().str.lower()
            country_counter.update(cleaned_countries)

        # 1. Filter by US country
        if country_col and country_col in chunk.columns:
            us_mask = cleaned_countries.isin(US_COUNTRY_VARIANTS)
            us_chunk = chunk[us_mask].copy()
        else:
            us_chunk = chunk.copy()

        if us_chunk.empty:
            print(f"Chunk {chunk_idx:3d}: {chunk_rows:7,d} rows scanned | 0 US matches | Cumulative saved: {total_rows_saved:7,d}")
            continue

        # 2. Relaxed matching: company name present OR expanded AI keyword present
        text_series = us_chunk[text_col].fillna("").astype(str)
        company_mask = text_series.str.contains(company_pattern, na=False)
        ai_mask = text_series.str.contains(ai_pattern, na=False)
        matched_mask = company_mask | ai_mask

        filtered_chunk = us_chunk[matched_mask].copy()
        matched_count = len(filtered_chunk)

        if matched_count > 0:
            # 3. Extract date from URL (fallback to date_col/year)
            if url_col and url_col in filtered_chunk.columns:
                fallback_series = filtered_chunk[date_col] if (date_col and date_col in filtered_chunk.columns) else [""] * matched_count
                filtered_chunk['date'] = [
                    extract_date_from_url(u, y)
                    for u, y in zip(filtered_chunk[url_col], fallback_series)
                ]
            else:
                filtered_chunk['date'] = filtered_chunk[date_col].astype(str) if (date_col and date_col in filtered_chunk.columns) else ""

            # 4. Save to output CSV
            if first_write:
                filtered_chunk.to_csv(output_path, mode='w', index=False, header=True)
                first_write = False
            else:
                filtered_chunk.to_csv(output_path, mode='a', index=False, header=False)

            total_rows_saved += matched_count

        print(
            f"Chunk {chunk_idx:3d}: {chunk_rows:7,d} rows scanned | "
            f"{matched_count:5,d} matches | "
            f"Cumulative saved: {total_rows_saved:7,d}"
        )

    # If no rows matched across entire dataset, create an empty CSV with headers
    if first_write:
        headers = list(chunk.columns)
        if 'date' not in headers:
            headers.append('date')
        pd.DataFrame(columns=headers).to_csv(output_path, index=False)

    return total_rows_scanned, total_rows_saved, country_counter


# ---------------------------------------------------------------------------
# Fallback Pipeline (standard library csv for minimal environments)
# ---------------------------------------------------------------------------

def process_file_csv(
    input_path: Path,
    output_path: Path,
    preferred_date_col: str = None,
    preferred_text_col: str = None,
    preferred_country_col: str = None,
    preferred_url_col: str = None,
):
    import csv
    csv.field_size_limit(sys.maxsize)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    company_pattern = build_regex_pattern(TARGET_COMPANIES)
    ai_pattern = build_regex_pattern(AI_KEYWORDS)

    country_counter = Counter()
    total_rows_scanned = 0
    filtered_rows = []

    with open(input_path, mode='r', encoding='utf-8', errors='replace') as f:
        clean_lines = (line.replace('\0', '') for line in f)
        reader = csv.DictReader(clean_lines)
        fieldnames = list(reader.fieldnames)

        date_col, text_col, country_col, url_col = detect_columns(
            fieldnames,
            preferred_date_col=preferred_date_col,
            preferred_text_col=preferred_text_col,
            preferred_country_col=preferred_country_col,
            preferred_url_col=preferred_url_col,
        )

        out_fieldnames = fieldnames if 'date' in fieldnames else fieldnames + ['date']

        for row in reader:
            total_rows_scanned += 1
            raw_country = (row.get(country_col) or '').strip() if country_col else ''
            country_lower = raw_country.lower()
            country_counter[raw_country if raw_country else 'UNKNOWN'] += 1

            if country_col and country_lower not in US_COUNTRY_VARIANTS:
                continue

            content = (row.get(text_col) or '') if text_col else ''
            has_comp = bool(company_pattern.search(content))
            has_ai = bool(ai_pattern.search(content))

            if has_comp or has_ai:
                url = row.get(url_col) or '' if url_col else ''
                year = row.get(date_col) or '' if date_col else ''
                row['date'] = extract_date_from_url(url, year)
                filtered_rows.append(row)

    with open(output_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    return total_rows_scanned, len(filtered_rows), country_counter


# ---------------------------------------------------------------------------
# CLI & Execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Filter news articles for US target companies and expanded AI keywords."
    )
    project_root = Path(__file__).resolve().parent.parent
    default_input = project_root / "data" / "global_news_private.csv"
    default_output = project_root / "data" / "filtered_companies_v2.csv"

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=default_input,
        help=f"Path to input CSV file (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=default_output,
        help=f"Path to output CSV file (default: {default_output})",
    )
    parser.add_argument(
        "--chunksize",
        "-c",
        type=int,
        default=DEFAULT_CHUNKSIZE,
        help=f"Pandas chunk size (default: {DEFAULT_CHUNKSIZE})",
    )
    parser.add_argument(
        "--country-col",
        type=str,
        default=None,
        help="Explicit country column name (overrides auto-detection)",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default=None,
        help="Explicit text/content column name (overrides auto-detection)",
    )
    parser.add_argument(
        "--url-col",
        type=str,
        default=None,
        help="Explicit URL column name (overrides auto-detection)",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default=None,
        help="Explicit date/year column name (overrides auto-detection)",
    )

    args = parser.parse_args()

    # Use pandas when installed, otherwise fall back to standard streaming reader
    if pd is not None:
        total_scanned, total_saved, country_counter = process_file_pandas(
            input_path=args.input,
            output_path=args.output,
            chunksize=args.chunksize,
            preferred_date_col=args.date_col,
            preferred_text_col=args.text_col,
            preferred_country_col=args.country_col,
            preferred_url_col=args.url_col,
        )
    else:
        print("Note: pandas not found in current environment, using streaming csv reader.")
        total_scanned, total_saved, country_counter = process_file_csv(
            input_path=args.input,
            output_path=args.output,
            preferred_date_col=args.date_col,
            preferred_text_col=args.text_col,
            preferred_country_col=args.country_col,
            preferred_url_col=args.url_col,
        )

    # Inspect final file to get new shape and min/max date
    import csv
    csv.field_size_limit(sys.maxsize)
    num_cols = 0
    dates = []
    with open(args.output, mode='r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        num_cols = len(reader.fieldnames) if reader.fieldnames else 0
        for r in reader:
            d = (r.get('date') or '').strip()
            if d:
                dates.append(d)

    min_date = min(dates) if dates else "N/A"
    max_date = max(dates) if dates else "N/A"

    # Display Required Information
    print("\n" + "=" * 70)
    print("COUNTRY DISTRIBUTION BEFORE FILTERING (Top 25)")
    print("=" * 70)
    for code, count in country_counter.most_common(25):
        pct = (count / total_scanned * 100) if total_scanned else 0
        print(f"  {code:10s}: {count:7,d} ({pct:5.2f}%)")
    print(f"  Total unique countries before filtering: {len(country_counter)}")

    print("\n" + "=" * 70)
    print("FILTERED DATASET METRICS")
    print("=" * 70)
    print(f"Output File:     {args.output}")
    print(f"New Shape:       ({total_saved}, {num_cols})  [Rows: {total_saved:,}, Columns: {num_cols}]")
    print(f"Min Date (new):  {min_date}")
    print(f"Max Date (new):  {max_date}")
    print("=" * 70)


if __name__ == "__main__":
    main()
