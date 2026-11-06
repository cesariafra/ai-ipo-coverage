#!/usr/bin/env python3
"""
Production Full-Text Article Scraper.

Scrapes full article text for all URLs in data/gdelt_news_2025_2026_us_deep_dedup.csv
using 3 parallel Playwright workers with stealth anti-bot evasion.

Features:
  - Parallelism: 3 concurrent Playwright browser instances via ThreadPoolExecutor.
  - Resume: skips URLs already present in data/scraped_articles_full.csv.
  - Article validation: checks trafilatura metadata (title + date/author).
  - Domain blacklist: filters known aggregators / non-article domains.
  - Batch saving: flushes to CSV every 50 articles to prevent data loss.
  - Full stealth: random User-Agent, viewport, mouse simulation, image blocking.

Categories:
  SUCCESS        - clean article text extracted (> 200 chars)
  PAYWALL/BLOCKED - HTTP 401/403, or 200 with too-short text
  NOT_AN_ARTICLE  - text extracted but trafilatura metadata lacks title+date/author
  ERROR           - timeout, connection error, or unexpected status
  BLACKLISTED     - domain on the aggregator/garbage blacklist

Usage:
    python src/07_scrape_production.py
    python src/07_scrape_production.py --limit 500   # test run
    python src/07_scrape_production.py --workers 5 --headless
"""
from __future__ import annotations
import argparse
import csv
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

try:
    import pandas as pd
except ImportError:
    raise SystemExit("ERROR: pandas is required. Run: pip install pandas")

try:
    import trafilatura
except ImportError:
    raise SystemExit("ERROR: trafilatura is required. Run: pip install trafilatura")

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    from playwright_stealth import Stealth
except ImportError:
    raise SystemExit(
        "ERROR: playwright and playwright-stealth are required.\n"
        "Run: pip install playwright playwright-stealth && playwright install chromium"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT  = PROJECT_ROOT / "data" / "gdelt_news_2025_2026_us_deep_dedup.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "scraped_articles_full.csv"

DEFAULT_WORKERS    = 3      # Parallel Playwright instances
BATCH_SAVE_EVERY   = 50     # Flush results to disk every N articles
SUCCESS_MIN_CHARS  = 200    # Minimum text length to count as SUCCESS
PAGE_TIMEOUT_MS    = 20_000 # Playwright page.goto() timeout

# Domains using aggressive bot detection (DataDome / Cloudflare) —
# skip them entirely to protect the IP from rate-ban.
HARDCODED_BLOCKED = {
    "forbes.com", "nytimes.com", "bloomberg.com",
    "wsj.com", "reuters.com", "techcrunch.com",
    "zerohedge.com", "nypost.com",
}

# Known aggregator / SEO-spam / non-article domains to filter out.
AGGREGATOR_BLACKLIST = {
    "article.wn.com", "wn.com", "geneamusings.com",
    "paperity.org", "readability.com", "alltop.com",
    "flipboard.com", "smartnews.com", "newsblur.com",
    "feedburner.com", "feedproxy.google.com",
}

BLOCKED_DOMAINS = HARDCODED_BLOCKED | AGGREGATOR_BLACKLIST

# Realistic browser User-Agents (Mac + Windows, Chrome + Firefox + Safari)
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]

# Common desktop screen resolutions
VIEWPORTS = [
    {"width": 1920, "height": 1080},
    {"width": 1440, "height": 900},
    {"width": 1536, "height": 864},
    {"width": 1366, "height": 768},
]

# CSV columns written to the output file
OUTPUT_COLUMNS = ["url", "domain", "status", "category", "text_length", "article_text"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_domain(url: str) -> str:
    """Return bare domain (no www.) from a URL, or empty string on failure."""
    try:
        netloc = urlparse(url).netloc
        return netloc[4:].lower() if netloc.startswith("www.") else netloc.lower()
    except Exception:
        return ""


def is_blocked(domain: str) -> bool:
    """Return True if the domain matches any entry in the blacklist."""
    return any(blocked in domain for blocked in BLOCKED_DOMAINS)


def validate_article_metadata(html: str) -> bool:
    """
    Use trafilatura metadata extraction to verify this is a real article.
    Requires at minimum: a title AND (a date OR an author).
    Returns True if it passes as a real article, False otherwise.
    """
    try:
        meta = trafilatura.extract_metadata(html)
        if meta is None:
            return False
        has_title  = bool(meta.title and meta.title.strip())
        has_date   = bool(meta.date and meta.date.strip())
        has_author = bool(meta.author and meta.author.strip())
        return has_title and (has_date or has_author)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-thread scraping logic (one Playwright browser per thread)
# ---------------------------------------------------------------------------

def scrape_worker(
    worker_id: int,
    urls: list[str],
    total_articles: int,
    start_offset: int,
    result_queue: list,
    lock: threading.Lock,
    counter: dict,
    headless: bool,
) -> None:
    """
    Worker function executed in a dedicated thread.
    Launches its own Playwright browser and scrapes all URLs in its shard.

    Args:
        worker_id    : integer identifier for logging (1, 2, 3, …)
        urls         : the slice of URLs assigned to this worker
        total_articles: total articles being scraped (for display only)
        start_offset : index offset of this worker's first URL in the global list
        result_queue : shared list; each worker appends its result dicts here
        lock         : threading.Lock protecting result_queue and counter writes
        counter      : shared dict with key "done" tracking global progress
        headless     : whether to run Chromium in headless mode
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",  # hides navigator.webdriver
                "--no-sandbox",
            ],
        )

        for local_idx, url in enumerate(urls):
            global_idx = start_offset + local_idx + 1
            domain = extract_domain(url)

            # --- Blacklist check ---
            if is_blocked(domain):
                result = {
                    "url": url,
                    "domain": domain,
                    "status": "BLACKLISTED",
                    "category": "BLACKLISTED",
                    "text_length": 0,
                    "article_text": "",
                }
                with lock:
                    result_queue.append(result)
                    counter["done"] += 1
                    _log(global_idx, total_articles, "BLACKLISTED", domain, 0)
                continue

            # --- Spin up a fresh randomised browser context per URL ---
            context = browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport=random.choice(VIEWPORTS),
                ignore_https_errors=True,
            )
            page = context.new_page()

            # Apply playwright-stealth to mask automation fingerprints
            Stealth().use_sync(page)

            # Block images / media / fonts — faster loads & less bot-like bandwidth
            page.route(
                "**/*",
                lambda route: (
                    route.abort()
                    if route.request.resource_type in {"image", "media", "font", "stylesheet"}
                    else route.continue_()
                ),
            )

            status_str   = "UNKNOWN"
            category     = "ERROR"
            text_length  = 0
            article_text = ""

            try:
                resp = page.goto(url, timeout=PAGE_TIMEOUT_MS, wait_until="domcontentloaded")
                http_status = resp.status if resp else 0

                if http_status in (401, 403):
                    status_str = str(http_status)
                    category   = "PAYWALL/BLOCKED"

                elif http_status == 404:
                    status_str = "404"
                    category   = "ERROR"

                elif http_status == 200:
                    # --- Simulate human behaviour ---
                    # 1. Random scroll to trigger lazy-load content
                    page.mouse.wheel(0, random.randint(300, 800))
                    # 2. Pause as a human would while reading
                    page.wait_for_timeout(random.randint(1000, 2000))
                    # 3. Move mouse to a random position
                    page.mouse.move(
                        random.randint(100, 500),
                        random.randint(100, 500),
                    )
                    # 4. Brief pause after mouse movement
                    page.wait_for_timeout(random.randint(500, 1500))

                    html = page.content()

                    # Extract clean article text
                    extracted = trafilatura.extract(
                        html,
                        include_comments=False,
                        include_tables=False,
                        no_fallback=False,
                    )

                    if extracted:
                        text_length  = len(extracted.strip())
                        article_text = extracted.strip()

                        if text_length <= SUCCESS_MIN_CHARS:
                            # Extracted something but very short → paywall / cookie wall
                            status_str = "200_SHORT"
                            category   = "PAYWALL/BLOCKED"
                        else:
                            # Validate it's a real article, not an aggregator page
                            if validate_article_metadata(html):
                                status_str = "200"
                                category   = "SUCCESS"
                            else:
                                status_str = "200_NO_META"
                                category   = "NOT_AN_ARTICLE"
                    else:
                        # 200 but trafilatura returned nothing
                        status_str = "200_EMPTY"
                        category   = "PAYWALL/BLOCKED"

                else:
                    # 5xx, 429, redirects that land on something unexpected, etc.
                    status_str = str(http_status)
                    category   = "ERROR"

            except PlaywrightTimeoutError:
                status_str = "TIMEOUT"
                category   = "ERROR"
            except Exception as exc:
                status_str = type(exc).__name__
                category   = "ERROR"
            finally:
                page.close()
                context.close()

            result = {
                "url": url,
                "domain": domain,
                "status": status_str,
                "category": category,
                "text_length": text_length,
                "article_text": article_text,
            }

            with lock:
                result_queue.append(result)
                counter["done"] += 1
                _log(global_idx, total_articles, category, domain, text_length)

            # Polite inter-request delay to avoid triggering rate limiters
            time.sleep(random.uniform(1.0, 2.5))

        browser.close()


def _log(global_idx: int, total: int, category: str, domain: str, length: int) -> None:
    """Print a formatted progress line to stdout."""
    width = len(str(total))
    print(f"[{global_idx:{width}d}/{total}] {category:<16} - {domain:<40} Length: {length:>7,}")


# ---------------------------------------------------------------------------
# Batch saving
# ---------------------------------------------------------------------------

def save_batch(rows: list[dict], output_path: Path, write_header: bool) -> None:
    """Append a list of result dicts to the output CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_production_scraper(
    input_path: Path,
    output_path: Path,
    max_workers: int,
    limit: int | None,
    headless: bool,
) -> None:
    print("=" * 75)
    print("PRODUCTION ARTICLE SCRAPER")
    print("=" * 75)
    print(f"Input:    {input_path}")
    print(f"Output:   {output_path}")
    print(f"Workers:  {max_workers}")
    print(f"Headless: {headless}")
    print(f"Limit:    {limit or 'ALL'}")
    print("=" * 75)

    # 1. Load input dataset
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    url_col = "news_url" if "news_url" in df.columns else df.columns[0]
    all_urls: list[str] = df[url_col].dropna().unique().tolist()
    print(f"\nTotal URLs in dataset: {len(all_urls):,}")

    # 2. Resume: collect already-scraped URLs from existing output
    already_done: set[str] = set()
    output_exists = output_path.exists() and output_path.stat().st_size > 0
    if output_exists:
        try:
            done_df = pd.read_csv(output_path, usecols=["url"])
            already_done = set(done_df["url"].dropna().tolist())
            print(f"Resuming: {len(already_done):,} URLs already scraped — skipping them.")
        except Exception as e:
            print(f"[Warning] Could not read existing output for resume: {e}")

    # 3. Build the work list (deduplicated, minus already-done)
    pending_urls = [u for u in all_urls if u not in already_done]
    if limit:
        pending_urls = pending_urls[:limit]

    total = len(pending_urls)
    if total == 0:
        print("Nothing left to scrape. Exiting.")
        return

    print(f"Articles to scrape this run: {total:,}\n")

    # 4. Partition URLs across workers (round-robin sharding)
    shards: list[list[str]] = [[] for _ in range(max_workers)]
    for i, url in enumerate(pending_urls):
        shards[i % max_workers].append(url)

    # Compute each worker's global start offset for display
    offsets = []
    running = 0
    for shard in shards:
        offsets.append(running)
        running += len(shard)

    # 5. Shared state for threads
    result_queue: list[dict] = []
    lock         = threading.Lock()
    counter      = {"done": 0}
    pending_flush: list[dict] = []   # accumulates results between batch saves
    write_header = not output_exists  # only write CSV header if file is new

    start_time = time.time()

    # 6. Launch workers via ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                scrape_worker,
                worker_id   = wid + 1,
                urls        = shards[wid],
                total_articles = total,
                start_offset   = offsets[wid],
                result_queue   = result_queue,
                lock           = lock,
                counter        = counter,
                headless       = headless,
            )
            for wid in range(max_workers)
        ]

        # 7. Batch-save loop: flush every BATCH_SAVE_EVERY articles
        last_saved = 0
        while any(not f.done() for f in futures):
            time.sleep(2)
            with lock:
                new_results = result_queue[last_saved:]

            if len(new_results) >= BATCH_SAVE_EVERY:
                save_batch(new_results, output_path, write_header)
                write_header = False          # header written once
                last_saved += len(new_results)
                print(f"\n  ↳ [SAVED] Flushed {last_saved:,} / {total:,} results to {output_path.name}\n")

        # Wait for all futures to finish cleanly
        for f in futures:
            f.result()   # re-raises any worker exception

    # 8. Final flush of any remaining results
    with lock:
        remaining = result_queue[last_saved:]
    if remaining:
        save_batch(remaining, output_path, write_header)
        last_saved += len(remaining)

    # 9. Summary report
    total_duration = time.time() - start_time
    results_df = pd.read_csv(output_path)
    counts = results_df["category"].value_counts()

    print("\n" + "=" * 75)
    print("PRODUCTION SCRAPE — FINAL SUMMARY")
    print("=" * 75)
    print(f"Total processed this run:  {last_saved:,}")
    print(f"Total duration:            {total_duration / 60:.1f} minutes")
    print(f"Average per article:       {total_duration / max(last_saved, 1):.2f} s")
    print("-" * 75)
    for cat in ["SUCCESS", "PAYWALL/BLOCKED", "NOT_AN_ARTICLE", "ERROR", "BLACKLISTED"]:
        n = counts.get(cat, 0)
        pct = n / max(len(results_df), 1) * 100
        print(f"  {cat:<20} {n:>7,}  ({pct:.1f}%)")
    print("-" * 75)
    success_df = results_df[results_df["category"] == "SUCCESS"]
    if not success_df.empty:
        avg_len = success_df["text_length"].mean()
        print(f"  Avg text length (SUCCESS): {avg_len:,.0f} chars")
    print(f"\nFull results saved to: {output_path}")
    print("=" * 75)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production-grade parallel article scraper with stealth evasion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",   "-i", type=Path, default=DEFAULT_INPUT,
                        help="Path to input CSV with news_url column.")
    parser.add_argument("--output",  "-o", type=Path, default=DEFAULT_OUTPUT,
                        help="Path to output CSV (append-safe, resume-aware).")
    parser.add_argument("--workers", "-w", type=int,  default=DEFAULT_WORKERS,
                        help="Number of parallel Playwright browser instances.")
    parser.add_argument("--limit",   "-l", type=int,  default=None,
                        help="Max articles to scrape this run (omit = all pending).")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run Chromium in headless mode (default: True).")
    parser.add_argument("--no-headless", dest="headless", action="store_false",
                        help="Show the browser UI (useful for debugging).")

    args = parser.parse_args()

    run_production_scraper(
        input_path  = args.input,
        output_path = args.output,
        max_workers = args.workers,
        limit       = args.limit,
        headless    = args.headless,
    )


if __name__ == "__main__":
    main()

