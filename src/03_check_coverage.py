import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None

# Percorso del file (compatibile sia da root che da sottocartelle)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "gdelt_news_2025_2026.csv"

if not DATA_PATH.exists():
    print(f"Errore: File non trovato in {DATA_PATH}")
    sys.exit(1)


def extract_date(url):
    """
    Estrae la data YYYY-MM-DD dall'URL tramite pattern multipli,
    validando mese (01-12) e giorno (01-31) per evitare falsi positivi da ID articolo.
    """
    if not isinstance(url, str):
        return None
    patterns = [
        r'/((?:20\d{2}))[/-](0[1-9]|1[0-2])[/-](0[1-9]|[12]\d|3[01])/',       # /2026/10/26/ o /2026-10-26/
        r'[/-]((?:20\d{2}))[/-](0[1-9]|1[0-2])[/-](0[1-9]|[12]\d|3[01])',     # -2026-10-26 o /2026-10-26
        r'[?&]date=((?:20\d{2}))-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])',      # ?date=2026-10-26
        r'/((?:20\d{2}))(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])/',               # /20261026/
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


# Lettura dati
if pd is not None:
    df = pd.read_csv(DATA_PATH)
    df["date"] = df["news_url"].apply(extract_date)
    covered_days = set(df["date"].dropna().unique())
else:
    import csv
    covered_days = set()
    with open(DATA_PATH, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = extract_date(row.get("news_url"))
            if d:
                covered_days.add(d)

# Definizione intervallo temporale
start = datetime(2025, 12, 1)
end = datetime(2026, 11, 2)

# Include anche l'ultimo giorno (end)
total_days_count = (end - start).days + 1
all_days = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(total_days_count)]

# Calcolo giorni coperti e mancanti all'interno del range
covered_in_range = set(d for d in covered_days if d in set(all_days))
missing_days = [d for d in all_days if d not in covered_in_range]

# Output metriche
print(f"Giorni totali nel range: {len(all_days)}")
print(f"Giorni con almeno un articolo: {len(covered_in_range)}")
print(f"Giorni completamente vuoti: {len(missing_days)}")
print(f"Copertura: {len(covered_in_range) / len(all_days) * 100:.1f}%")

if missing_days:
    print(f"\nElenco dei {len(missing_days)} giorni mancanti:")
    print(missing_days)

