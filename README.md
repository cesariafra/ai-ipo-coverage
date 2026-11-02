# AI IPO Coverage Analysis: The "Quiet Period" Effect

This repository contains the code, data pipelines, and materials for my bachelor's thesis, conducted under the supervision of Prof. Daniele Quercia and Damiano Giallongo at Politecnico di Torino.

## Project Overview

**Context:** 2026 is expected to be the strongest year for IPOs since 2021. AI companies such as OpenAI, Anthropic, and SpaceX have already filed (or are expected to file) for IPOs. Once a company files, securities law forbids it from promoting itself to the press (the "Quiet Period"). We know that AI news is typically shaped by the companies themselves—over 60% of it traces back to company press announcements.

**Research Question:** *How do AI companies sound once they can't speak for themselves?* 

**Proposal:** To address this, we will build a longitudinal dataset of AI-related news coverage and analyze shifts in sourcing and framing across different IPO phases.

## Methodology

The pipeline is structured into the following phases:

1.  **Data Collection:** Pull time-stamped news articles mentioning "OpenAI", "Anthropic", and "SpaceX" from the GDELT Project, retaining only those related to AI.
2.  **IPO Phase Tagging:** Auto-label each article based on its publish date relative to known IPO milestones (e.g., baseline >6 months pre-filing, rumor, confidential-filing, public S-1, roadshow, post-listing).
3.  **Source Tagging:** Code each article for the quoted or cited source (e.g., Company representative, Analyst, Journalist, Competitor, Anonymous source).
4.  **Sentiment Scoring:** Evaluate sentiment along two dimensions:
    *   *Financial sentiment:* Using FinBERT or Loughran-McDonald dictionary (risk, uncertainty, growth language).
    *   *General sentiment:* Using VADER or standard NLP pipelines (positive/neutral/negative tone).
5.  **Risk/Benefit Extraction:** Use an LLM to extract specific risks (e.g., job displacement) and benefits (e.g., revenue growth) per article. Compute a risk-to-benefit ratio: `risk_count / (benefit_count + 1)`.
6.  **Analysis & Visualization:** Plot sentiment and risk-to-benefit ratios over time per company, marking phase boundaries.

## Hypotheses

*   **H1 (Sourcing):** The share of articles quoting company representatives will drop sharply once a company enters confidential filing, replaced by other sources.
*   **H2 (Sentiment):** Sentiment scores will become more negative during the confidential filing and pre-S-1 window compared to the baseline, since positive company framing is no longer available to counterbalance critical coverage.
*   **H3 (Risk/Benefit Framing):** The ratio of risk-framed language (safety, regulation, job loss) to benefit-framed language (growth, capability) will rise during the quiet period and fall again once the public S-1 is released.

## Repository Structure

*   `data/` - Local directory for raw and filtered datasets (Ignored by git due to size).
*   `src/` - Python scripts for data processing, NLP pipelines, and analysis.
*   `notebooks/` - Jupyter notebooks for exploratory data analysis and prototyping.
*   `slides/` - PDF slides used for progress meetings.

## Data Disclaimer

The raw dataset (`global_news_private.csv`) is over 2GB and is not tracked in this repository. Because of its size, the Python scripts utilize `pandas` chunking to process the data efficiently without exhausting RAM. The `data/` folder is excluded via `.gitignore`.

## Setup & Requirements

To replicate the analysis, create a Python environment and install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn transformers torch vaderSentiment
