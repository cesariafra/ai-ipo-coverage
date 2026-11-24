#!/usr/bin/env python3
"""
Recupera le date di pubblicazione native da GDELT 2.0 su Google BigQuery
(gdelt-bq.gdeltv2.gkg_partitioned, rif. Mouchel et al. 2026, Appendice B.1)
per tutti gli URL presenti nel dataset.
"""

import argparse
import json
import os
import re
from pathlib import Path
import pandas as pd
from google.cloud import bigquery


def normalize_url(u: str) -> str:
    if not isinstance(u, str):
        return ""
    u = u.strip().lower()
    u = re.sub(r"^https?://(www\.)?", "", u)
    u = u.rstrip("/")
    return u


def main():
    key_path = Path("gcp_key.json")
    default_project = None
    if key_path.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path.resolve())
        with open(key_path, "r", encoding="utf-8") as f:
            default_project = json.load(f).get("project_id")

    parser = argparse.ArgumentParser(description="Recupera date GDELT via BigQuery JOIN.")
    parser.add_argument("--project", default=default_project, help="Google Cloud Project ID")
    parser.add_argument("--input", default="data/articles_with_phases.csv", help="Dataset di input")
    parser.add_argument("--output", default="data/gdelt_bq_dates.csv", help="File CSV con le date recuperate")
    parser.add_argument("--start-date", default="2025-01-01", help="Inizio partizione GDELT")
    parser.add_argument("--end-date", default="2026-11-30", help="Fine partizione GDELT")
    args = parser.parse_args()

    if not args.project:
        raise SystemExit("Errore: --project non specificato e gcp_key.json non trovato nella cartella corrente.")

    input_path = Path(args.input)
    df = pd.read_csv(input_path, low_memory=False)
    url_col = "url" if "url" in df.columns else "news_url"
    print(f"[1/4] Project ID: {args.project} | Caricati {len(df):,} articoli da {input_path}")

    urls_df = df[[url_col]].rename(columns={url_col: "url"}).dropna().drop_duplicates().copy()
    urls_df["norm_url"] = urls_df["url"].apply(normalize_url)
    urls_df = urls_df[urls_df["norm_url"] != ""].drop_duplicates(subset=["norm_url"])
    print(f"      URL unici normalizzati da cercare su GDELT: {len(urls_df):,}")

    client = bigquery.Client(project=args.project)

    # 1. Crea dataset temporaneo e carica la tabella degli URL
    dataset_id = f"{args.project}.tmp_ipo_lookup"
    table_id = f"{dataset_id}.target_urls"

    dataset = bigquery.Dataset(dataset_id)
    dataset.location = "US"
    client.create_dataset(dataset, exists_ok=True)

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    print("[2/4] Caricamento URL su tabella temporanea BigQuery...")
    load_job = client.load_table_from_dataframe(urls_df[["url", "norm_url"]], table_id, job_config=job_config)
    load_job.result()

    # 2. Query SQL su gdelt-bq.gdeltv2.gkg_partitioned (Appendice B.1)
    sql = f"""
    WITH gdelt_matched AS (
      SELECT
        REGEXP_REPLACE(
          REGEXP_REPLACE(LOWER(TRIM(DocumentIdentifier)), r'^https?://(www\\.)?', ''),
          r'/$', ''
        ) AS norm_url,
        MIN(DATE) AS min_gdelt_date,
        ANY_VALUE(SourceCommonName) AS gdelt_source
      FROM `gdelt-bq.gdeltv2.gkg_partitioned`
      WHERE DATE(_PARTITIONTIME) >= DATE('{args.start_date}')
        AND DATE(_PARTITIONTIME) <= DATE('{args.end_date}')
      GROUP BY norm_url
    )
    SELECT
      t.url,
      g.min_gdelt_date,
      g.gdelt_source
    FROM `{table_id}` t
    INNER JOIN gdelt_matched g
      ON t.norm_url = g.norm_url
    """

    print("[3/4] Esecuzione JOIN su gdelt-bq.gdeltv2.gkg_partitioned...")
    res_df = client.query(sql).to_dataframe()

    # Pulizia tabella temporanea
    client.delete_table(table_id, not_found_ok=True)

    # Converte YYYYMMDDHHMMSS -> YYYY-MM-DD
    def parse_gdelt_ts(val):
        s = str(val).strip()
        if len(s) >= 8 and s[:8].isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return None

    res_df["bq_date"] = res_df["min_gdelt_date"].apply(parse_gdelt_ts)
    res_df = res_df.dropna(subset=["bq_date"]).drop_duplicates(subset=["url"])
    res_df.to_csv(args.output, index=False)
    print(f"[4/4] Match trovati su GDELT BigQuery: {len(res_df):,} / {len(urls_df):,} ({len(res_df)/len(urls_df)*100:.1f}%)")
    print(f"      Salvato in {args.output}")

    # Aggiorna direttamente data/recovered_dates.csv mantenendo lo schema esistente
    rec_path = Path("data/recovered_dates.csv")
    if rec_path.exists():
        old_rec = pd.read_csv(rec_path)
        u_col = "url" if "url" in old_rec.columns else old_rec.columns[0]
        d_col = "recovered_date" if "recovered_date" in old_rec.columns else ("date" if "date" in old_rec.columns else old_rec.columns[1])
        new_rec = res_df[["url", "bq_date"]].rename(columns={"url": u_col, "bq_date": d_col})
        if "date_source" in old_rec.columns:
            new_rec["date_source"] = "gdelt_bigquery"
        combined = pd.concat([new_rec, old_rec], ignore_index=True).drop_duplicates(subset=[u_col], keep="first")
    else:
        combined = res_df[["url", "bq_date"]].rename(columns={"bq_date": "recovered_date"})
        combined["date_source"] = "gdelt_bigquery"

    combined.to_csv(rec_path, index=False)
    print(f"[OK] Aggiornato {rec_path} (totale date recuperate: {len(combined):,})")


if __name__ == "__main__":
    main()
