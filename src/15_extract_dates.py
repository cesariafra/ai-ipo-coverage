#!/usr/bin/env python3
"""
Multi-task extraction and classification via Gemini (Batch Mode).

Supports two tasks via `--task`:
  1. `--task dates` (default):
     - Recovers missing publication dates for articles lacking dates in GDELT.
     - Saves to: data/recovered_dates.csv (columns: url, recovered_date)

  2. `--task sources`:
     - Stratifies across all canonical IPO phases and target companies (OpenAI, Anthropic, SpaceX).
     - Resolves attribution bias by strictly separating target-company sources from Third Party companies (e.g., Microsoft, Google, Nvidia).
     - Saves to: data/validation_sourcing_refined.csv (columns: url, specific_sources)

Features:
  - Batching: 10 articles per Gemini API request.
  - Rate limiting: 4-second sleep between batch calls.
  - Exponential backoff: Retries 503 / 429 up to 3 times (10s, 30s, 60s).
  - Resume capability: Skips URLs already present in the target CSV.

Usage:
    export GEMINI_API_KEY="your-key"
    python src/15_extract_dates.py --task dates
    python src/15_extract_dates.py --task sources --sample-size 300
    python src/15_extract_dates.py --task sources --limit 20
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

# Suppress noisy SDK warnings
logging.getLogger("google.generativeai").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRAPED_CSV  = PROJECT_ROOT / "data" / "scraped_articles_full.csv"
GDELT_CSV    = PROJECT_ROOT / "data" / "gdelt_news_2025_2026_us_deep_dedup.csv"
PHASES_CSV   = PROJECT_ROOT / "data" / "articles_with_phases.csv"

OUT_DATES_CSV   = PROJECT_ROOT / "data" / "recovered_dates.csv"
OUT_SOURCES_CSV = PROJECT_ROOT / "data" / "validation_sourcing_refined.csv"

MODEL_NAME        = "gemini-3.5-flash-lite"
BATCH_SIZE        = 10      # articles per API call
RATE_LIMIT_SLEEP  = 4.0     # seconds between batch API calls
MAX_ARTICLE_CHARS = 5_000   # keep token context efficient

ALLOWED_REFINED_SOURCES = {
    "Company Spokesperson",
    "Company Executive",
    "Company Document",
    "Third Party",
    "Competitor",
    "Analyst",
    "Journalist",
    "Anonymous",
}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROMPT_DATES = """\
Extract the publication date of the following news articles. 
Return a JSON array where each object has the article's `id` and the `date` in YYYY-MM-DD format. 
If a date is not explicitly found, return null for that date.
Articles: {articles_text}
"""

PROMPT_SOURCES_REFINED = """\
Analyze the following news articles about AI/tech IPOs (target companies: OpenAI, Anthropic, SpaceX).
Classify the information sources explicitly cited or quoted in each article.
Return a JSON array where each object has `id` and `sources` (a list of categories actually present).

Allowed categories:
- "Company Spokesperson": Official spokesperson, PR representative, or official press statement ONLY from OpenAI, Anthropic, or SpaceX.
- "Company Executive": Named executive, CEO, or founder (e.g., Sam Altman, Dario Amodei, Elon Musk) ONLY from OpenAI, Anthropic, or SpaceX.
- "Company Document": Official S-1 filing, financial disclosure, or corporate blog/report ONLY from OpenAI, Anthropic, or SpaceX.
- "Third Party": Executives, spokespersons, or financial reports from OTHER companies (e.g., Google, Microsoft, Nvidia, customers, partners).
- "Competitor": Rival AI/space companies framed in direct competition.
- "Analyst": Financial analysts, VC investors, or academic/industry experts.
- "Journalist": Independent investigative reporting or editorial analysis by the journalist/outlet.
- "Anonymous": Unnamed insiders or "people familiar with the matter".

Rules:
- Do NOT label statements or earnings reports from Google or Microsoft as "Company Executive" or "Company Document"; those belong to "Third Party".
- Only include categories explicitly supported by the text.
- Return ONLY valid JSON array, no markdown formatting.

Articles:
{articles_text}
"""


# ---------------------------------------------------------------------------
# General parsing helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers Gemini sometimes adds."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _parse_json_array(raw_text: str) -> list[dict]:
    """Parse JSON array from raw Gemini text response."""
    cleaned = _strip_markdown_fences(raw_text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            for val in data.values():
                if isinstance(val, list):
                    return val
            return [data]
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return data
            except (json.JSONDecodeError, ValueError):
                pass
    return []


def normalize_date(date_val) -> str | None:
    """Validate and normalize extracted date string to YYYY-MM-DD format."""
    if not date_val or not isinstance(date_val, str):
        return None
    date_str = date_val.strip()

    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str

    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", date_str)
    if iso_match:
        return iso_match.group(1)

    try:
        dt = pd.to_datetime(date_str, errors="coerce")
        if pd.notna(dt):
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass

    return None


def parse_batch_dates(raw_text: str, batch_size: int) -> list[str | None]:
    """Parse batch date extraction response into list of YYYY-MM-DD or None."""
    raw_items = _parse_json_array(raw_text)
    id_map = {}
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        try:
            target_id = int(item_id) if item_id is not None else idx
        except (ValueError, TypeError):
            target_id = idx
        id_map[target_id] = item

    results = []
    for i in range(batch_size):
        item = id_map.get(i)
        if not item and i < len(raw_items) and isinstance(raw_items[i], dict):
            item = raw_items[i]
        date_val = item.get("date") if item else None
        results.append(normalize_date(date_val))
    return results


def parse_batch_sources(raw_text: str, batch_size: int) -> list[list[str]]:
    """Parse batch sources response into list of sanitized category lists."""
    raw_items = _parse_json_array(raw_text)
    id_map = {}
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        try:
            target_id = int(item_id) if item_id is not None else idx
        except (ValueError, TypeError):
            target_id = idx
        id_map[target_id] = item

    results = []
    for i in range(batch_size):
        item = id_map.get(i)
        if not item and i < len(raw_items) and isinstance(raw_items[i], dict):
            item = raw_items[i]

        sources_raw = item.get("sources", []) if item else []
        if not isinstance(sources_raw, list):
            sources_raw = []

        cleaned_sources = [
            str(s).strip() for s in sources_raw
            if str(s).strip() in ALLOWED_REFINED_SOURCES
        ]
        results.append(cleaned_sources)
    return results


def load_already_processed(path: Path) -> set[str]:
    """Load set of URLs already present in target CSV for resume capability."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path, usecols=["url"])
        return set(df["url"].dropna().astype(str).tolist())
    except Exception as exc:
        print(f"[Warning] Could not read existing output for resume: {exc}")
        return set()


def save_batch(rows: list[dict], path: Path, fieldnames: list[str], write_header: bool) -> None:
    """Append a batch of rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Data loading for Task 1: Dates
# ---------------------------------------------------------------------------

def prepare_dates_dataset() -> pd.DataFrame:
    """Identify successfully scraped articles that lack valid dates in GDELT."""
    print("Loading datasets for Task 'dates'...")
    df_gdelt = pd.read_csv(GDELT_CSV, usecols=["news_url", "date"], low_memory=False)
    valid_gdelt_urls = set(df_gdelt.dropna(subset=["date"])["news_url"].dropna().astype(str))
    print(f"  Valid dates in GDELT: {len(valid_gdelt_urls):,}")

    df_scraped = pd.read_csv(SCRAPED_CSV, usecols=["url", "category", "article_text"], low_memory=False)
    df_success = df_scraped[
        (df_scraped["category"] == "SUCCESS")
        & df_scraped["article_text"].notna()
        & (df_scraped["article_text"].str.strip() != "")
    ].copy()
    print(f"  Successfully scraped articles: {len(df_success):,}")

    df_missing = df_success[~df_success["url"].astype(str).isin(valid_gdelt_urls)].copy()
    print(f"  Articles needing date extraction: {len(df_missing):,}")
    return df_missing


# ---------------------------------------------------------------------------
# Data loading for Task 2: Sources (Stratified across IPO phases & companies)
# ---------------------------------------------------------------------------

def prepare_stratified_sources_dataset(sample_size: int = 300) -> pd.DataFrame:
    """
    Create a stratified sample of articles across all canonical IPO phases and all
    three target companies (OpenAI, Anthropic, SpaceX).
    Merges data/articles_with_phases.csv with data/scraped_articles_full.csv to obtain article_text.
    """
    print(f"Loading datasets for Task 'sources' (Target Sample Size: {sample_size:,})...")
    if not PHASES_CSV.exists():
        raise FileNotFoundError(f"Missing canonical phases dataset: {PHASES_CSV}. Run src/18_assign_ipo_phase.py first.")
    if not SCRAPED_CSV.exists():
        raise FileNotFoundError(f"Missing scraped articles dataset: {SCRAPED_CSV}")

    df_phases = pd.read_csv(PHASES_CSV, low_memory=False)
    df_text = pd.read_csv(SCRAPED_CSV, usecols=["url", "article_text"], low_memory=False)

    merged = pd.merge(df_phases, df_text, on="url", how="inner")
    merged = merged[
        merged["article_text"].notna()
        & (merged["article_text"].str.strip() != "")
        & merged["final_date"].notna()
    ].copy()

    company_strata_defs = [
        ("OpenAI", "is_openai", "phase_openai"),
        ("Anthropic", "is_anthropic", "phase_anthropic"),
        ("SpaceX", "is_spacex", "phase_spacex"),
    ]

    # Collect available subsets per (company, phase) stratum
    strata_subsets: list[tuple[str, str, pd.DataFrame]] = []
    for comp, flag, p_col in company_strata_defs:
        comp_df = merged[merged[flag] & merged[p_col].notna()]
        phases = comp_df[p_col].unique()
        for p in phases:
            sub = comp_df[comp_df[p_col] == p]
            if len(sub) > 0:
                strata_subsets.append((comp, p, sub))

    num_strata = len(strata_subsets)
    print(f"  Identified {num_strata} active company-phase strata across eligible articles.")

    target_per_stratum = max(1, sample_size // max(num_strata, 1))

    sampled_frames = []
    for comp, p, sub in strata_subsets:
        k = min(len(sub), target_per_stratum)
        sampled_frames.append(sub.sample(k, random_state=42))

    stratified_sample = pd.concat(sampled_frames).drop_duplicates(subset=["url"])
    print(f"  Primary stratified sample: {len(stratified_sample):,} unique articles.")

    # Fill up to sample_size if some small strata had fewer articles than target
    if len(stratified_sample) < sample_size:
        remaining_pool = merged[~merged["url"].isin(stratified_sample["url"])]
        fill_count = min(len(remaining_pool), sample_size - len(stratified_sample))
        if fill_count > 0:
            fill_sample = remaining_pool.sample(fill_count, random_state=42)
            stratified_sample = pd.concat([stratified_sample, fill_sample]).drop_duplicates(subset=["url"])

    stratified_sample = stratified_sample.reset_index(drop=True)
    print(f"  Final balanced sample: {len(stratified_sample):,} articles ready for classification.")
    return stratified_sample


# ---------------------------------------------------------------------------
# Unified batch processing pipeline
# ---------------------------------------------------------------------------

def run_extraction_pipeline(task: str, api_key: str, sample_size: int = 300, limit: int | None = None) -> None:
    print("=" * 75)
    print(f"GEMINI BATCH EXTRACTION — TASK: {task.upper()}")
    print("=" * 75)

    client = genai.Client(api_key=api_key)
    print(f"Model       : {MODEL_NAME}")
    print(f"Batch Size  : {BATCH_SIZE} articles / request")

    if task == "dates":
        target_csv = OUT_DATES_CSV
        fieldnames = ["url", "recovered_date"]
        df_target = prepare_dates_dataset()
        prompt_template = PROMPT_DATES
    elif task == "sources":
        target_csv = OUT_SOURCES_CSV
        fieldnames = ["url", "specific_sources"]
        df_target = prepare_stratified_sources_dataset(sample_size=sample_size)
        prompt_template = PROMPT_SOURCES_REFINED
    else:
        raise ValueError(f"Unknown task: {task}")

    # Resume capability: skip URLs already in target CSV
    already_done = load_already_processed(target_csv)
    if already_done:
        print(f"  Resuming: {len(already_done):,} URLs already in {target_csv.name} — skipping.")
        df_target = df_target[~df_target["url"].astype(str).isin(already_done)]

    if limit is not None:
        df_target = df_target.head(limit)

    total_articles = len(df_target)
    if total_articles == 0:
        print("\nNo articles left to process. Exiting.")
        return

    num_batches = (total_articles + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Articles to process this run : {total_articles:,} across {num_batches:,} batches")
    print(f"  Target output file           : {target_csv}")
    print("=" * 75 + "\n")

    write_header = not target_csv.exists() or target_csv.stat().st_size == 0
    success_count = 0
    error_count   = 0
    start_time    = time.time()

    for batch_num in range(num_batches):
        batch_start = batch_num * BATCH_SIZE
        batch_df = df_target.iloc[batch_start : batch_start + BATCH_SIZE]
        curr_batch_size = len(batch_df)

        articles_formatted = []
        batch_urls = []
        for local_id, (_, row) in enumerate(batch_df.iterrows()):
            url = str(row["url"])
            text = str(row["article_text"]).strip()[:MAX_ARTICLE_CHARS]
            batch_urls.append(url)
            articles_formatted.append(f"--- ARTICLE {local_id} ---\n{text}\n")

        articles_text = "\n".join(articles_formatted)
        prompt = prompt_template.format(articles_text=articles_text)

        # Retry logic: 3 retries (10s, 30s, 60s) for 503/429
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

                if task == "dates":
                    parsed_results = parse_batch_dates(raw_text, curr_batch_size)
                    for local_id, rec_date in enumerate(parsed_results):
                        batch_processed_rows.append({
                            "url": batch_urls[local_id],
                            "recovered_date": rec_date or "",
                        })
                    found_dates = sum(1 for d in parsed_results if d)
                    print(
                        f"[Batch {batch_num + 1:>3}/{num_batches}] OK  "
                        f"Processed {curr_batch_size} articles ({found_dates}/{curr_batch_size} dates found) | "
                        f"Total: {batch_start + curr_batch_size}/{total_articles}"
                    )

                elif task == "sources":
                    parsed_results = parse_batch_sources(raw_text, curr_batch_size)
                    for local_id, sources_list in enumerate(parsed_results):
                        batch_processed_rows.append({
                            "url": batch_urls[local_id],
                            "specific_sources": json.dumps(sources_list),
                        })
                    print(
                        f"[Batch {batch_num + 1:>3}/{num_batches}] OK  "
                        f"Classified sources for {curr_batch_size} articles | "
                        f"Total: {batch_start + curr_batch_size}/{total_articles}"
                    )

                success_count += curr_batch_size
                break

            except Exception as exc:
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
                        if task == "dates":
                            batch_processed_rows.append({"url": url, "recovered_date": ""})
                        else:
                            batch_processed_rows.append({"url": url, "specific_sources": "[]"})
                    break

        # Flush batch to CSV immediately
        if batch_processed_rows:
            save_batch(batch_processed_rows, target_csv, fieldnames, write_header)
            write_header = False

        # Rate limit between batches (skip after final batch)
        if batch_num + 1 < num_batches:
            time.sleep(RATE_LIMIT_SLEEP)

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print("PIPELINE SUMMARY")
    print("=" * 75)
    print(f"Task executed            : {task}")
    print(f"Total articles processed : {total_articles:,}")
    print(f"Successful items         : {success_count:,}")
    print(f"Errors encountered       : {error_count:,}")
    print(f"Execution time           : {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")
    print(f"Saved to                 : {target_csv}")
    print("=" * 75)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-task Gemini extraction (dates recovery & granular source classification).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        choices=["dates", "sources"],
        default="dates",
        help="Task to run: 'dates' to recover missing dates, or 'sources' to classify granular sources across IPO phases.",
    )
    parser.add_argument(
        "--sample-size", "-s",
        type=int,
        default=300,
        help="Target sample size for stratified sampling across IPO phases in '--task sources' (default: 300).",
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=None,
        help="Gemini API key. Falls back to GEMINI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Max articles to process this run (default: all pending).",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "ERROR: Gemini API key not found.\n"
            "Pass it with --api-key or set the GEMINI_API_KEY environment variable."
        )

    run_extraction_pipeline(
        task=args.task,
        api_key=api_key,
        sample_size=args.sample_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
