import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pysentiment2 as ps

INPUT_PATH = "data/scraped_articles_full.csv"
OUTPUT_PATH = "data/finbert_lm_results.csv"
MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return -1


def main():
    df = pd.read_csv(INPUT_PATH)
    df = df[(df["category"] == "SUCCESS") & (df["article_text"].notna())].copy()
    df = df[df["article_text"].str.strip() != ""].drop_duplicates(subset=["url"])

    # Resume logic
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        done_urls = set(existing["url"].dropna())
        df = df[~df["url"].isin(done_urls)].copy()
        print(f"[Resume] Già analizzati: {len(done_urls):,}. Rimanenti: {len(df):,}")
    else:
        print(f"[Start] Articoli da analizzare: {len(df):,}")

    if df.empty:
        print("Tutti gli articoli sono già stati analizzati.")
        return

    # Inizializza Loughran-McDonald
    lm = ps.LM()

    # Inizializza FinBERT
    device = get_device()
    print(f"Caricamento {MODEL_NAME} su device: {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    finbert = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=512,
    )

    urls = df["url"].tolist()
    texts = df["article_text"].astype(str).tolist()

    for i in range(0, len(texts), BATCH_SIZE):
        batch_urls = urls[i : i + BATCH_SIZE]
        # FinBERT usa i primi ~2000 caratteri (max 512 token); LM analizza fino a 15.000 caratteri
        batch_texts_fb = [t[:2500] for t in texts[i : i + BATCH_SIZE]]
        batch_texts_lm = [t[:15000] for t in texts[i : i + BATCH_SIZE]]

        fb_preds = finbert(batch_texts_fb)

        rows = []
        for url, pred, full_txt in zip(batch_urls, fb_preds, batch_texts_lm):
            label = pred["label"].capitalize()  # Positive, Negative, Neutral
            conf = round(float(pred["score"]), 4)
            signed_score = conf if label == "Positive" else (-conf if label == "Negative" else 0.0)

            tokens = lm.tokenize(full_txt)
            lm_scores = lm.get_score(tokens)
            # In pysentiment2 get_score restituisce Positive, Negative, Polarity, Subjectivity
            rows.append({
                "url": url,
                "finbert_label": label,
                "finbert_confidence": conf,
                "finbert_signed_score": round(signed_score, 4),
                "lm_positive": lm_scores.get("Positive", 0),
                "lm_negative": lm_scores.get("Negative", 0),
                "lm_polarity": round(lm_scores.get("Polarity", 0.0), 4),
                "lm_subjectivity": round(lm_scores.get("Subjectivity", 0.0), 4),
            })

        batch_df = pd.DataFrame(rows)
        header = not os.path.exists(OUTPUT_PATH)
        batch_df.to_csv(OUTPUT_PATH, mode="a", header=header, index=False)

        processed = min(i + BATCH_SIZE, len(texts))
        if (i // BATCH_SIZE) % 10 == 0 or processed == len(texts):
            print(f"Progress: {processed:,}/{len(texts):,} articoli salvati...")

    # Riepilogo finale
    final_df = pd.read_csv(OUTPUT_PATH)
    print("\n=== DISTRIBUZIONE FINBERT ===")
    print(final_df["finbert_label"].value_counts(normalize=True).mul(100).round(2))


if __name__ == "__main__":
    main()