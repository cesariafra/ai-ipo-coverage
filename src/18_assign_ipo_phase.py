from __future__ import annotations

import os
import pandas as pd

MILESTONES_PATH = "data/ipo_milestones.csv"
NLP_PATH = "data/nlp_analysis_results.csv"
VADER_PATH = "data/vader_sentiment_results.csv"
GDELT_PATH = "data/gdelt_news_2025_2026_us_deep_dedup.csv"
RECOVERED_DATES_PATH = "data/recovered_dates.csv"
TEXT_PATH = "data/scraped_articles_full.csv"
OUTPUT_PATH = "data/articles_with_phases.csv"

# 1. Lookup table canonica delle milestone per azienda (allineata alle 6 fasi UROP)
MILESTONES_DATA = [
    # SpaceX (Public S-1: 20 Maggio 2026, Listing: 12 Giugno 2026)
    {"company": "SpaceX", "phase": "baseline", "start_date": "1900-01-01", "end_date": "2025-11-30"},
    {"company": "SpaceX", "phase": "rumor", "start_date": "2025-12-01", "end_date": "2026-05-19"},
    {"company": "SpaceX", "phase": "public_s1", "start_date": "2026-05-20", "end_date": "2026-05-31"},
    {"company": "SpaceX", "phase": "roadshow", "start_date": "2026-06-01", "end_date": "2026-06-11"},
    {"company": "SpaceX", "phase": "post_listing", "start_date": "2026-06-12", "end_date": "2099-12-31"},

    # Anthropic (Confidential S-1: 1 Giugno 2026, Public S-1: 26 Ottobre 2026)
    {"company": "Anthropic", "phase": "baseline", "start_date": "1900-01-01", "end_date": "2025-11-30"},
    {"company": "Anthropic", "phase": "rumor", "start_date": "2025-12-01", "end_date": "2026-05-31"},
    {"company": "Anthropic", "phase": "confidential_filing", "start_date": "2026-06-01", "end_date": "2026-10-25"},
    {"company": "Anthropic", "phase": "public_s1", "start_date": "2026-10-26", "end_date": "2026-11-15"},
    {"company": "Anthropic", "phase": "roadshow", "start_date": "2026-11-16", "end_date": "2099-12-31"},

    # OpenAI (Confidential S-1: 8 Giugno 2026)
    {"company": "OpenAI", "phase": "baseline", "start_date": "1900-01-01", "end_date": "2025-11-30"},
    {"company": "OpenAI", "phase": "rumor", "start_date": "2025-12-01", "end_date": "2026-06-07"},
    {"company": "OpenAI", "phase": "confidential_filing", "start_date": "2026-06-08", "end_date": "2099-12-31"},
]


def build_lookup_table() -> pd.DataFrame:
    df_m = pd.DataFrame(MILESTONES_DATA)
    df_m.to_csv(MILESTONES_PATH, index=False)
    print(f"[OK] Lookup table salvata in {MILESTONES_PATH}")
    return df_m


def assign_company_phase(pub_date: pd.Timestamp, company_rules: list[tuple[pd.Timestamp, pd.Timestamp, str]]) -> str | None:
    if pd.isna(pub_date):
        return None
    for start, end, phase in company_rules:
        if start <= pub_date <= end:
            return phase
    return None


def main():
    df_milestones = build_lookup_table()

    # Caricamento dataset
    df_nlp = pd.read_csv(NLP_PATH)
    df_vader = pd.read_csv(VADER_PATH)
    df_text = pd.read_csv(TEXT_PATH, usecols=["url", "article_text"])
    df_gdelt = pd.read_csv(GDELT_PATH, usecols=["news_url", "date"]).rename(columns={"news_url": "url", "date": "gdelt_date"})

    # Merge centrale
    df = df_nlp.merge(df_vader, on="url", how="left")
    df = df.merge(df_text, on="url", how="left")
    df = df.merge(df_gdelt.drop_duplicates(subset=["url"]), on="url", how="left")

    # Integrazione date recuperate da Gemini se il file esiste
    if os.path.exists(RECOVERED_DATES_PATH):
        df_rec = pd.read_csv(RECOVERED_DATES_PATH).drop_duplicates(subset=["url"])
        df = df.merge(df_rec, on="url", how="left")
        df["final_date"] = df["gdelt_date"].fillna(df["recovered_date"])
    else:
        df["final_date"] = df["gdelt_date"]

    df["final_date"] = pd.to_datetime(df["final_date"], errors="coerce")

    # Flag booleani multi-azienda (risolve la fallacia dell'assegnazione esclusiva)
    text_series = df["article_text"].fillna("")
    df["is_openai"] = text_series.str.contains(r"\bOpenAI\b", case=False, regex=True)
    df["is_anthropic"] = text_series.str.contains(r"\bAnthropic\b", case=False, regex=True)
    df["is_spacex"] = text_series.str.contains(r"\bSpaceX\b", case=False, regex=True)

    # Prepara regole per lookup rapido
    rules_by_company = {}
    for company, group in df_milestones.groupby("company"):
        rules_by_company[company] = [
            (pd.to_datetime(r["start_date"]), pd.to_datetime(r["end_date"]), r["phase"])
            for _, r in group.iterrows()
        ]

    # Assegna fase specifica per ciascuna azienda
    df["phase_openai"] = df["final_date"].apply(lambda d: assign_company_phase(d, rules_by_company["OpenAI"]))
    df["phase_anthropic"] = df["final_date"].apply(lambda d: assign_company_phase(d, rules_by_company["Anthropic"]))
    df["phase_spacex"] = df["final_date"].apply(lambda d: assign_company_phase(d, rules_by_company["SpaceX"]))

    # Rimuovi article_text prima di salvare per mantenere il CSV leggero
    df_out = df.drop(columns=["article_text"])
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"\n[OK] Dataset unificato salvato in {OUTPUT_PATH} (Righe: {len(df_out):,})")
    print(f"Articoli con data valida: {df_out['final_date'].notna().sum():,}")
    print(f"Menzioni -> OpenAI: {df_out['is_openai'].sum():,} | Anthropic: {df_out['is_anthropic'].sum():,} | SpaceX: {df_out['is_spacex'].sum():,}")
    print(f"Articoli che citano >=2 aziende contemporaneamente: {(df_out[['is_openai', 'is_anthropic', 'is_spacex']].sum(axis=1) >= 2).sum():,}")


if __name__ == "__main__":
    main()