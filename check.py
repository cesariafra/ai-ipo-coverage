import pandas as pd

PATH = "data/global_news_private.csv"
US = {"us", "usa", "united states", "united states of america"}

print("=== Prime 5 righe ===")
# Leggiamo solo le prime righe per stampa (skippando eventuali righe corrotte)
df_head = pd.read_csv(PATH, nrows=5, on_bad_lines='skip')
print(df_head[['country', 'year']].to_string(index=False))

print("\n=== Scansione primi 500.000 record ===")
us_count = 0
countries_seq = []
total_rows = 0

# Usiamo chunksize per non saturare la RAM e on_bad_lines per ignorare i NUL
for chunk in pd.read_csv(PATH, chunksize=100000, on_bad_lines='skip'):
    if total_rows >= 500000:
        break
    
    # Pulizia colonna country
    chunk['country_clean'] = chunk['country'].astype(str).str.strip().str.lower()
    
    # Conteggio US
    us_count += chunk['country_clean'].isin(US).sum()
    
    # Salvataggio sequenza
    if len(countries_seq) < 50:
        rem = 50 - len(countries_seq)
        countries_seq.extend(chunk['country_clean'].head(rem).tolist())
        
    total_rows += len(chunk)

print(f"country=us nei primi 500k: {us_count}")
print(f"Sequenza primi 50 country: {countries_seq}")
