#!/usr/bin/env python3
"""
Deep deduplication using three signals:
1. Exact normalized title match
2. URL slug match (catches syndication across domains)
3. Fuzzy title match (catches minor variations)
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import pandas as pd
    from rapidfuzz import fuzz, process
except ImportError as e:
    print(f"Missing dependency: {e}. Install with: pip install pandas rapidfuzz")
    sys.exit(1)


def normalize_title(text):
    t = re.sub(r'\s*\|\s*[^\s|]+$', '', str(text))
    t = re.sub(r'[^\w\s]', ' ', t.lower())
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def extract_slug(url):
    """Extract the last meaningful path segment from a URL, stripping dates."""
    if not isinstance(url, str):
        return ''
    # Remove query string and fragment
    u = url.split('?')[0].split('#')[0].rstrip('/')
    # Take last path segment
    parts = [p for p in u.split('/') if p and p not in ('http:', 'https:', '')]
    if not parts:
        return ''
    slug = parts[-1].lower()
    # Strip leading/trailing date patterns
    slug = re.sub(r'^\d{4}[-/]\d{2}[-/]\d{2}[-_]', '', slug)
    slug = re.sub(r'[-_]\d{4}[-/]\d{2}[-/]\d{2}$', '', slug)
    slug = re.sub(r'^\d{8}[-_]', '', slug)
    slug = re.sub(r'[-_]\d{8}$', '', slug)
    # Remove .html, .htm, .php extensions
    slug = re.sub(r'\.(html?|php|aspx?)$', '', slug)
    return slug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('data/gdelt_news_2025_2026_us_clean.csv'))
    parser.add_argument('--output', type=Path, default=Path('data/gdelt_news_2025_2026_us_deep_dedup.csv'))
    parser.add_argument('--fuzzy-threshold', type=int, default=90, help='Fuzzy match threshold (0-100)')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Righe input: {len(df):,}")

    df['title_norm'] = df['news_content'].apply(normalize_title)
    df['slug'] = df['news_url'].apply(extract_slug)

    # --- LIVELLO 1: dedup titolo esatto ---
    print(f"\n[Livello 1] Dedup titolo esatto...")
    df['is_dup_l1'] = df.duplicated(subset='title_norm', keep='first')
    l1_removed = df['is_dup_l1'].sum()
    print(f"  Rimossi: {l1_removed:,} ({l1_removed/len(df)*100:.1f}%)")
    df_l1 = df[~df['is_dup_l1']].copy()
    print(f"  Rimanenti: {len(df_l1):,}")

    # --- LIVELLO 2: dedup slug (syndication cross-domain) ---
    print(f"\n[Livello 2] Dedup per slug URL...")
    # Considera solo slug non vuoti e non troppo corti (evita falsi positivi)
    valid_slug_mask = (df_l1['slug'].str.len() >= 15)
    df_with_slug = df_l1[valid_slug_mask].copy()
    df_no_slug = df_l1[~valid_slug_mask].copy()

    # Conta quanti domini diversi hanno lo stesso slug
    slug_domain_counts = df_with_slug.groupby('slug')['news_url'].apply(
        lambda s: len(set(re.search(r'https?://([^/]+)', u).group(1) for u in s if re.search(r'https?://([^/]+)', u)))
    )
    syndicated_slugs = slug_domain_counts[slug_domain_counts >= 2].index

    df_with_slug['is_dup_l2'] = df_with_slug['slug'].isin(syndicated_slugs) & df_with_slug.duplicated(subset='slug', keep='first')
    l2_removed = df_with_slug['is_dup_l2'].sum()
    print(f"  Slug syndacati (>=2 domini): {len(syndicated_slugs):,}")
    print(f"  Rimossi: {l2_removed:,} ({l2_removed/len(df_l1)*100:.1f}%)")
    df_l2 = pd.concat([df_with_slug[~df_with_slug['is_dup_l2']], df_no_slug])
    print(f"  Rimanenti: {len(df_l2):,}")

    # --- LIVELLO 3: fuzzy matching sui rimanenti ---
    print(f"\n[Livello 3] Fuzzy matching (threshold={args.fuzzy_threshold})...")
    titles = df_l2['title_norm'].tolist()
    keep_mask = [True] * len(titles)

    # Per ogni titolo, cerca duplicati fuzzy tra i successivi
    for i in range(len(titles)):
        if not keep_mask[i]:
            continue
        # Cerca tra i successivi (non tra i precedenti già processati)
        matches = process.extract(
            titles[i],
            {j: titles[j] for j in range(i + 1, len(titles)) if keep_mask[j]},
            scorer=fuzz.ratio,
            score_cutoff=args.fuzzy_threshold,
            limit=None,
        )
        for _, _, idx in matches:
            keep_mask[idx] = False

    l3_removed = sum(1 for k in keep_mask if not k)
    print(f"  Rimossi: {l3_removed:,} ({l3_removed/len(df_l2)*100:.1f}%)")
    df_final = df_l2[keep_mask].copy()
    print(f"  Rimanenti: {len(df_final):,}")

    # --- RIEPILOGO ---
    total_removed = len(df) - len(df_final)
    print(f"\n{'=' * 60}")
    print(f"RIEPILOGO DEDUPLICAZIONE")
    print(f"{'=' * 60}")
    print(f"  Input:             {len(df):,}")
    print(f"  Livello 1 (titolo):  -{l1_removed:,}")
    print(f"  Livello 2 (slug):    -{l2_removed:,}")
    print(f"  Livello 3 (fuzzy):   -{l3_removed:,}")
    print(f"  Totale rimosso:    -{total_removed:,} ({total_removed/len(df)*100:.1f}%)")
    print(f"  Output:            {len(df_final):,}")

    # Mostra esempi di fuzzy match per audit
    print(f"\nEsempi di duplicati fuzzy rimossi (primi 5):")
    shown = 0
    for i in range(len(titles)):
        if keep_mask[i]:
            continue
        # Trova il "keeper" (il primo titolo simile che è stato tenuto)
        for j in range(i - 1, -1, -1):
            if keep_mask[j] and fuzz.ratio(titles[i], titles[j]) >= args.fuzzy_threshold:
                print(f"  KEPT:   {titles[j][:90]}")
                print(f"  REMOVED:{titles[i][:90]}")
                print()
                shown += 1
                break
        if shown >= 5:
            break

    # Salva output
    drop_cols = ['is_dup_l1', 'is_dup_l2', 'title_norm', 'slug']
    df_final = df_final.drop(columns=[c for c in drop_cols if c in df_final.columns], errors='ignore')
    df_final.to_csv(args.output, index=False)
    print(f"File salvato: {args.output}")


if __name__ == '__main__':
    main()
