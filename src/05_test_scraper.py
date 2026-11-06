#!/usr/bin/env python3
"""
Advanced Stealth Test Scraper using Playwright + playwright-stealth.
"""

import argparse
import time
import random
from pathlib import Path
from urllib.parse import urlparse

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas required.")
    exit(1)

try:
    import trafilatura
except ImportError:
    print("ERROR: trafilatura required.")
    exit(1)

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    from playwright_stealth import Stealth
except ImportError:
    print("ERROR: playwright and playwright-stealth required.")
    exit(1)

DEFAULT_SAMPLE_SIZE = 100
DEFAULT_RANDOM_STATE = 42
SUCCESS_MIN_CHARS = 200

# Domini noti per usare DataDome/Cloudflare aggressivo che ci bloccano sempre.
# Li saltiamo a piè pari per non farci bannare l'IP.
BLOCKED_DOMAINS = []

# Lista di User-Agent reali (Mac, Windows, iPhone)
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0"
]

# Dimensioni schermo comuni
VIEWPORTS = [
    {"width": 1920, "height": 1080},
    {"width": 1440, "height": 900},
    {"width": 1536, "height": 864},
    {"width": 1366, "height": 768}
]

def extract_domain(url):
    try:
        hostname = urlparse(url).netloc
        if hostname.startswith("www."):
            hostname = hostname[4:]
        return hostname.lower()
    except:
        return ""

def run_test_scraper(input_path: Path, output_path: Path, sample_size: int, headless: bool):
    print("=" * 75)
    print("ADVANCED STEALTH SCRAPING TEST")
    print("=" * 75)

    df = pd.read_csv(input_path)
    url_col = "news_url" if "news_url" in df.columns else df.columns[0]
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=DEFAULT_RANDOM_STATE)
    urls_to_test = sample_df[url_col].dropna().tolist()
    
    print(f"Testing {len(urls_to_test)} URLs...\n")

    results = []
    start_time = time.time()

    with sync_playwright() as p:
        browser = p.chromium.launch(
                headless=headless,
                args=[
                    "--disable-blink-features=AutomationControlled",  # hides navigator.webdriver flag
                    "--no-sandbox",                                    # required in some CI/container envs
                ]
            )
        
        for idx, url in enumerate(urls_to_test, start=1):
            domain = extract_domain(url)
            
            # 1. Salta i domini impossibili
            if any(bd in domain for bd in BLOCKED_DOMAINS):
                print(f"[{idx:3d}/{len(urls_to_test)}] BLACKLISTED    - {domain}")
                results.append({"url": url, "status": "BLACKLISTED", "category": "BLACKLISTED", "text_length": 0, "article_text": ""})
                continue

            # 2. Crea un contesto randomizzato per ogni richiesta
            context = browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport=random.choice(VIEWPORTS),
                ignore_https_errors=True
            )
            page = context.new_page()
            
            # 3. Applica Stealth
            Stealth().use_sync(page)

            # 4. Blocca immagini e CSS per essere più veloci e sembrare "umani"
            page.route("**/*", lambda route: route.abort() if route.request.resource_type in ["image", "media", "font"] else route.continue_())

            status, text_length, category = "UNKNOWN", 0, "ERROR"

            try:
                # Aspetta il network idle (il sito ha finito di caricare le chiamate API)
                page.goto(url, timeout=20000, wait_until="domcontentloaded")
                
                # Simulate human: scroll down a random amount first
                page.mouse.wheel(0, random.randint(300, 800))

                # Wait a short random time (1-2s) as a human would while reading
                page.wait_for_timeout(random.randint(1000, 2000))

                # Move the mouse smoothly to a random coordinate on the page
                page.mouse.move(random.randint(100, 500), random.randint(100, 500))

                # Small additional pause after mouse movement
                page.wait_for_timeout(random.randint(500, 1500))

                html_content = page.content()
                extracted_text = trafilatura.extract(html_content, include_comments=False, include_tables=False)

                if extracted_text:
                    text_length = len(extracted_text.strip())
                    if text_length > SUCCESS_MIN_CHARS:
                        category, status = "SUCCESS", "200"
                    else:
                        category, status = "PAYWALL/BLOCKED", "200_SHORT"
                else:
                    extracted_text = ""
                    category, status = "PAYWALL/BLOCKED", "200_EMPTY"
                    
            except PlaywrightTimeoutError:
                status, category = "TIMEOUT", "ERROR"
                extracted_text = ""
            except Exception as e:
                status, category = type(e).__name__, "ERROR"
                extracted_text = ""
            finally:
                page.close()
                context.close()

            results.append({"url": url, "status": status, "category": category, "text_length": text_length, "article_text": extracted_text or ""})
            print(f"[{idx:3d}/{len(urls_to_test)}] {category:<15} (Length: {text_length:5d}) - {domain}")
            
            time.sleep(random.uniform(1.0, 2.5))

        browser.close()

    results_df = pd.DataFrame(results)[["url", "status", "category", "text_length", "article_text"]]
    results_df.to_csv(output_path, index=False)

    total_duration = time.time() - start_time
    counts = results_df["category"].value_counts()
    
    print("\n" + "=" * 75)
    print("SUMMARY REPORT")
    print("=" * 75)
    print(f"Total Attempted:  {len(results_df)}")
    print(f"Duration:         {total_duration:.1f} seconds")
    print(f"SUCCESS:          {counts.get('SUCCESS', 0)} ({counts.get('SUCCESS', 0)/len(results_df)*100:.1f}%)")
    print(f"BLACKLISTED:      {counts.get('BLACKLISTED', 0)}")
    print(f"PAYWALL/BLOCKED:  {counts.get('PAYWALL/BLOCKED', 0)}")
    print(f"ERROR:            {counts.get('ERROR', 0)}")
    print(f"\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--input", type=Path, default=project_root / "data" / "gdelt_news_2025_2026_us_deep_dedup.csv")
    parser.add_argument("--output", type=Path, default=project_root / "data" / "scraped_articles_test.csv")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    run_test_scraper(args.input, args.output, args.sample_size, args.headless)