#!/usr/bin/env python3
"""
NLP Analysis via Gemini API (Batch Mode).

Processes articles in batches (default: 10 per API call) to drastically reduce
request counts and stay within free-tier quota limits (1500 req/day, 15 req/min).

For each article, extracts:
  - sources cited        (Company, Analyst, Journalist, Competitor, Anonymous, Other)
  - financial_sentiment  (Negative | Neutral | Positive)
  - general_sentiment    (Negative | Neutral | Positive)
  - risks                (up to 5 specific phrases)
  - benefits             (up to 5 specific phrases)

Features:
  - Batching: Sends 10 articles per Gemini API request.
  - Resume: Skips URLs already present in data/nlp_analysis_results.csv.
  - Incremental save: Flushes to CSV after every batch.
  - Rate limiting: Sleep 4s between batch calls (≤ 15 req/min).
  - Robust JSON parsing: Handles JSON array responses with ID mapping.
  - Exponential backoff: Retries 503 / 429 up to 3 times (10s, 30s, 60s).

Usage:
    export GEMINI_API_KEY="your-key"
    python src/09_nlp_analysis.py

    # Test with first 20 articles (2 batches)
    python src/09_nlp_analysis.py --limit 20
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    raise SystemExit("ERROR: pandas required. Run: pip install pandas")

try:
    from google import genai
except ImportError:
    raise SystemExit(
        "ERROR: google-genai required.\n"
        "Run: pip install google-genai"
    )

# Suppress noisy SDK warnings (e.g., thought_signature)
logging.getLogger("google.generativeai").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV    = PROJECT_ROOT / "data" / "scraped_articles_full.csv"
OUTPUT_CSV   = PROJECT_ROOT / "data" / "nlp_analysis_results.csv"

MODEL_NAME        = "gemini-3.5-flash-lite"
BATCH_SIZE        = 10      # articles per API call
RATE_LIMIT_SLEEP  = 4.0     # seconds between batch API calls
MAX_ARTICLE_CHARS = 8_000   # max characters per article to keep token usage efficient

OUTPUT_COLUMNS = [
    "url",
    "sources",
    "financial_sentiment",
    "general_sentiment",
    "risks",
    "benefits",
]

VALID_SENTIMENTS = {"Negative", "Neutral", "Positive"}
VALID_SOURCES    = {"Company", "Analyst", "Journalist", "Competitor", "Anonymous", "Other"}


# ---------------------------------------------------------------------------
# Gemini prompt template
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
Analyze the following {count} news articles.
Return a JSON ARRAY of objects, where each object corresponds to an article (in order) and has EXACTLY this schema:
[
  {{
    "id": 0,
    "sources": ["List of only categories actually found in the text"],
    "financial_sentiment": "Negative" | "Neutral" | "Positive",
    "general_sentiment": "Negative" | "Neutral" | "Positive",
    "risks": ["list", "of", "specific", "risks"],
    "benefits": ["list", "of", "specific", "benefits"]
  }},
  ...
]

Rules:
- `sources`: Allowed categories to choose from are: "Company", "Analyst", "Journalist", "Competitor", "Anonymous", "Other". CRITICAL: Only include a source category in the list if the text explicitly quotes, cites, or references someone from that specific category. If an article only cites a journalist, return ['Journalist']. DO NOT return categories that are not present. If a company spokesperson is cited, include "Company".
- `risks` / `benefits`: Extract specific mentions (e.g., "job displacement", "regulatory scrutiny", "revenue growth"). Do not include generic words. Return empty lists if none are found.
- Return ONLY valid JSON array, no markdown.

Articles:
{articles_text}
"""


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers Gemini sometimes adds."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _sanitize_list(value, valid_set: set | None = None) -> list:
    """Ensure a value is a list of strings, optionally filtered by a valid set."""
    if not isinstance(value, list):
        return []
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    if valid_set:
        cleaned = [item for item in cleaned if item in valid_set]
    return cleaned


def _sanitize_sentiment(value: str) -> str:
    """Return one of the three valid sentiments, defaulting to Neutral."""
    if isinstance(value, str) and value.strip() in VALID_SENTIMENTS:
        return value.strip()
    return "Neutral"


def parse_batch_gemini_response(raw_text: str, batch_size: int) -> list[dict]:
    """
    Parse Gemini's JSON array response for a batch of articles.
    Matches returned items by 'id' (0 to batch_size - 1) or positional index.
    Returns a list of exactly `batch_size` sanitized result dictionaries.
    """
    default_record = {
        "sources":             [],
        "financial_sentiment": "Neutral",
        "general_sentiment":   "Neutral",
        "risks":               [],
        "benefits":            [],
    }

    raw_items = []
    try:
        cleaned = _strip_markdown_fences(raw_text)
        data = json.loads(cleaned)
        if isinstance(data, list):
            raw_items = data
        elif isinstance(data, dict):
            # If wrapped in a dictionary like {"articles": [...]}
            for val in data.values():
                if isinstance(val, list):
                    raw_items = val
                    break
            if not raw_items:
                raw_items = [data]
    except (json.JSONDecodeError, ValueError):
        # Fallback: regex search for outer JSON array [...]
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    raw_items = data
            except (json.JSONDecodeError, ValueError):
                pass

    # Map raw items by ID or index
    id_map: dict[int, dict] = {}
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        try:
            target_id = int(item_id) if item_id is not None else idx
        except (ValueError, TypeError):
            target_id = idx
        id_map[target_id] = item

    # Assemble sanitized list for each expected ID 0..batch_size-1
    results: list[dict] = []
    for i in range(batch_size):
        item = id_map.get(i)
        if not item and i < len(raw_items) and isinstance(raw_items[i], dict):
            item = raw_items[i]

        if item:
            results.append({
                "sources":             _sanitize_list(item.get("sources", []), VALID_SOURCES),
                "financial_sentiment": _sanitize_sentiment(item.get("financial_sentiment", "")),
                "general_sentiment":   _sanitize_sentiment(item.get("general_sentiment", "")),
                "risks":               _sanitize_list(item.get("risks", []))[:5],
                "benefits":            _sanitize_list(item.get("benefits", []))[:5],
            })
        else:
            results.append(dict(default_record))

    return results


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_already_done(path: Path) -> set[str]:
    """Return the set of URLs already present in the output CSV."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path, usecols=["url"])
        return set(df["url"].dropna().astype(str).tolist())
    except Exception as exc:
        print(f"[Warning] Could not read existing output for resume: {exc}")
        return set()


def save_batch(rows: list[dict], path: Path, write_header: bool) -> None:
    """Append a list of result dicts to the output CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({
                "url":                 row["url"],
                "sources":             json.dumps(row["sources"]),
                "financial_sentiment": row["financial_sentiment"],
                "general_sentiment":   row["general_sentiment"],
                "risks":               json.dumps(row["risks"]),
                "benefits":            json.dumps(row["benefits"]),
            })


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_nlp_analysis(api_key: str, limit: int | None) -> None:
    print("=" * 70)
    print("NLP ANALYSIS — Gemini Batch Mode")
    print("=" * 70)

    # 1. Configure the Gemini client
    client = genai.Client(api_key=api_key)
    print(f"Model       : {MODEL_NAME}")
    print(f"Batch Size  : {BATCH_SIZE} articles / request")

    # 2. Load input
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Rows in scraped dataset: {len(df):,}")

    # 3. Filter to successful articles with non-empty text
    df = df[
        (df["category"] == "SUCCESS") &
        df["article_text"].notna() &
        (df["article_text"].str.strip() != "")
    ].copy()
    print(f"Eligible rows (SUCCESS + non-empty text): {len(df):,}")

    # 4. Resume: skip already-analysed URLs
    already_done = load_already_done(OUTPUT_CSV)
    if already_done:
        print(f"Resuming: {len(already_done):,} URLs already analysed — skipping them.")
        df = df[~df["url"].astype(str).isin(already_done)]

    if limit:
        df = df.head(limit)

    total_articles = len(df)
    if total_articles == 0:
        print("Nothing left to analyse. Exiting.")
        return

    num_batches = (total_articles + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Articles to analyse this run : {total_articles:,} across {num_batches:,} batches")
    print("=" * 70 + "\n")

    # 5. Output file setup
    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
    success_count = 0
    error_count   = 0
    start_time    = time.time()

    # 6. Iterate in batches
    for batch_num in range(num_batches):
        batch_start = batch_num * BATCH_SIZE
        batch_df = df.iloc[batch_start : batch_start + BATCH_SIZE]
        curr_batch_size = len(batch_df)

        # Build formatted articles text
        articles_formatted = []
        batch_urls = []
        for local_id, (_, row) in enumerate(batch_df.iterrows()):
            url = str(row["url"])
            text = str(row["article_text"]).strip()[:MAX_ARTICLE_CHARS]
            batch_urls.append(url)
            articles_formatted.append(f"--- ARTICLE {local_id} ---\n{text}\n")

        articles_text = "\n".join(articles_formatted)
        prompt = PROMPT_TEMPLATE.format(count=curr_batch_size, articles_text=articles_text)

        # Retry loop for batch (3 retries: 10s, 30s, 60s)
        max_retries = 3
        backoff_seconds = [10, 30, 60]
        attempt = 0
        batch_processed_rows: list[dict] = []

        while True:
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                )
                raw_text = response.text
                parsed_list = parse_batch_gemini_response(raw_text, curr_batch_size)

                for local_id, parsed_row in enumerate(parsed_list):
                    parsed_row["url"] = batch_urls[local_id]
                    batch_processed_rows.append(parsed_row)

                success_count += curr_batch_size
                print(
                    f"[Batch {batch_num + 1:>3}/{num_batches}] OK  "
                    f"Processed {curr_batch_size} articles "
                    f"({batch_start + curr_batch_size}/{total_articles})"
                )
                break  # success, exit retry loop

            except Exception as exc:
                # Check for temporary 503 / 429 errors
                is_temp = False
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if code in (503, 429):
                    is_temp = True

                status_str = str(getattr(exc, "status", "")).upper()
                if any(tag in status_str for tag in ("UNAVAILABLE", "RESOURCE_EXHAUSTED", "TOO_MANY_REQUESTS")):
                    is_temp = True

                err_msg = str(exc).upper()
                if any(tag in err_msg for tag in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "TOO_MANY_REQUESTS")):
                    is_temp = True

                if is_temp and attempt < max_retries:
                    wait = backoff_seconds[attempt]
                    attempt += 1
                    print(
                        f"[Batch {batch_num + 1:>3}/{num_batches}] RETRY {attempt}/{max_retries} "
                        f"after {wait}s due to {type(exc).__name__}: {exc}"
                    )
                    time.sleep(wait)
                    continue
                else:
                    error_count += curr_batch_size
                    print(f"[Batch {batch_num + 1:>3}/{num_batches}] ERR {type(exc).__name__}: {exc}")
                    for url in batch_urls:
                        batch_processed_rows.append({
                            "url":                 url,
                            "sources":             [],
                            "financial_sentiment": "ERROR",
                            "general_sentiment":   "ERROR",
                            "risks":               [],
                            "benefits":            [],
                        })
                    break

        # Save batch results immediately to disk
        if batch_processed_rows:
            save_batch(batch_processed_rows, OUTPUT_CSV, write_header)
            write_header = False

        # Rate limit between batches (skip after final batch)
        if batch_num + 1 < num_batches:
            time.sleep(RATE_LIMIT_SLEEP)

    # 7. Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total processed : {total_articles:,}")
    print(f"Successful calls: {success_count:,}")
    print(f"Errors          : {error_count:,}")
    print(f"Duration        : {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")
    print(f"Output saved to : {OUTPUT_CSV}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gemini NLP analysis on scraped articles in batches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=None,
        help="Gemini API key. Falls back to GEMINI_API_KEY env var.",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Max articles to process this run (omit = all pending).",
    )
    args = parser.parse_args()

    # Resolve API key: CLI arg → env var → error
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "ERROR: Gemini API key not found.\n"
            "Pass it with --api-key or set the GEMINI_API_KEY environment variable."
        )

    run_nlp_analysis(api_key=api_key, limit=args.limit)


if __name__ == "__main__":
    main()
