import pandas as pd
import re

PATH = "data/global_news_private.csv"
US = {"us", "usa", "united states", "united states of america"}
PATTERN = re.compile(
    r"\b(AI|Artificial Intelligence|LLM|LLMs|Large Language Models?|ChatGPT|Claude|"
    r"Generative AI|GenAI|Sam Altman|Dario Amodei|Elon Musk|OpenAI|Anthropic|SpaceX)\b",
    re.IGNORECASE,
)

n_us = 0
n_match = 0

print("Scansione in corso (potrebbe volerci un po' su 600k righe)...")
for chunk in pd.read_csv(PATH, chunksize=100000, on_bad_lines='skip'):
    chunk['country_clean'] = chunk['country'].astype(str).str.strip().str.lower()
    us_chunk = chunk[chunk['country_clean'].isin(US)].copy()
    
    n_us += len(us_chunk)
    
    # Assicuriamoci che news_content sia trattato come stringa
    us_chunk['news_content'] = us_chunk['news_content'].astype(str)
    matches = us_chunk['news_content'].str.contains(PATTERN, na=False)
    n_match += matches.sum()

print(f"Articoli US totali: {n_us}")
print(f"Articoli US con keyword: {n_match}")
