#!/usr/bin/env python3
"""
Clean the US-only GDELT dataset by removing SpaceX-about-rockets noise.
Strategy: keep every article EXCEPT those that have space terms but no AI/IPO terms.
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas required. Install with: pip install pandas")
    sys.exit(1)


AI_TERMS = [
    'artificial intelligence', 'chatgpt', 'claude', 'generative',
    'genai', 'llm', 'language model', 'neural network', 'machine learning',
    'openai', 'anthropic',
]

IPO_TERMS = [
    'ipo', 'public offering', 's-1', 'listing', 'filing',
    'go public', 'flotation', 'valuation',
]

SPACE_TERMS = [
    'rocket', 'launch', 'starship', 'falcon', 'starlink', 'orbit',
    'nasa', 'satellite', 'mars', 'lunar', 'spaceflight', 'spacecraft',
    'space station', 'interplanetary', 'space tourism',
]


def count_terms(text: str, terms: list) -> int:
    t = text.lower()
    return sum(1 for term in terms if re.search(rf'\b{re.escape(term)}\b', t))


def classify(text: str) -> tuple:
    t = text.lower()
    ai = count_terms(t, AI_TERMS)
    ipo = count_terms(t, IPO_TERMS)
    space = count_terms(t, SPACE_TERMS)
    if ai + ipo > 0:
        return 'signal', ai, ipo, space
    if space > 0:
        return 'space_only', ai, ipo, space
    return 'generic', ai, ipo, space


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('data/gdelt_news_2025_2026_us.csv'))
    parser.add_argument('--output', type=Path, default=Path('data/gdelt_news_2025_2026_us_clean.csv'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Righe input: {len(df):,}")

    results = df['news_content'].apply(classify)
    df['category'] = [r[0] for r in results]
    df['ai_score'] = [r[1] for r in results]
    df['ipo_score'] = [r[2] for r in results]
    df['space_score'] = [r[3] for r in results]

    print("\nDistribuzione categorie (prima del filtro):")
    for cat, count in df['category'].value_counts().items():
        print(f"  {cat:12s}: {count:6,d} ({count/len(df)*100:5.1f}%)")

    # -----------------------------------------------------------------------
    # REGOLA DI ESCLUSIONE: rimuovi solo gli articoli "space-only"
    # (hanno termini spaziali ma ZERO termini AI o IPO)
    # -----------------------------------------------------------------------
    noise_mask = (df['space_score'] > 0) & (df['ai_score'] + df['ipo_score'] == 0)
    df_clean = df[~noise_mask].copy()
    # -----------------------------------------------------------------------

    print(f"\nArticoli rimossi (rumore spaziale): {noise_mask.sum():,}")
    print(f"Righe dopo pulizia: {len(df_clean):,} ({len(df_clean)/len(df)*100:.1f}%)")

    print("\nBreakdown per azienda (dopo pulizia):")
    for company in ['OpenAI', 'Anthropic', 'SpaceX']:
        n = df_clean['news_content'].str.contains(company, case=False, na=False).sum()
        print(f"  {company}: {n:,}")

    if args.dry_run:
        print("\n[DRY RUN] Nessun file scritto.")
        # Mostra i 10 titoli rimossi come campione
        print("\nCampione di 10 titoli rimossi:")
        for t in df[noise_mask]['news_content'].head(10).tolist():
            print(f"  - {t[:120]}")
        return

    # Rimuovi le colonne di scoring, tieni solo i dati originali + categoria
    df_clean = df_clean.drop(columns=['ai_score', 'ipo_score', 'space_score'])
    df_clean.to_csv(args.output, index=False)
    print(f"\nFile salvato: {args.output}")


if __name__ == '__main__':
    main()