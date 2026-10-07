# Pax Silica Media Monitor

A rule-based news monitoring pipeline that tracks themes and stance media coverage of the Pax Silica initiative in the Philippines. It ingests articles several times a day, classifies them with transparent keyword dictionaries, stores results in Google Sheets, and serves a live Streamlit dashboard.

**[Live dashboard](https://newsmonitoringnlp.streamlit.app/)** (best viewed on a laptop at 90% browser zoom)

---

## Overview

Pax Silica is a U.S.-led initiative on technology, AI infrastructure, and critical-mineral supply chains, with the Philippines positioned to play a role. Coverage moves quickly and each stakeholder group frames it differently, which makes manual tracking difficult.

This project is a topic-specific monitor built to answer four exploratory questions:

1. What themes appear in coverage, and which receive the most attention?
2. Is coverage supportive, critical, or neutral?
3. How has the story changed over time?
4. How do state-owned local, independent local, and international outlets differ in their reporting?

Business value: Journalists, researchers, and interested individuals without access to commercial media-intelligence platforms.

| | |
|---|---|
| **Corpus** | News articles, columns, editorials, press releases, organizational statements, and explainer articles (grows daily) |
| **Refresh rate** | 6 times per day |
| **Classifier** | Rule-based keyword matching (no trained model) |
| **Stack** | Python, pandas, GitHub Actions, Google Sheets API, Streamlit |

---
## Repository Structure
Path	Purpose
fetch_and_append.py	Pipeline: collect, filter, classify, deduplicate, append to Google Sheets
reclassify_sheet.py	Re-applies the classifier to existing sheet rows after keyword changes
streamlit_app.py	Dashboard reading the sheet as its live data source
.github/workflows/daily_fetch.yml	Scheduled workflow that runs the pipeline
requirements.txt	Python dependencies

---

## Architecture

```
Sources ──► Collect ──► Clean ──► Classify ──► Store ──► Visualize
(Google News RSS,       (relevance  (themes +   (Google    (Streamlit
 World News API,         filter,     stance)     Sheets)    dashboard)
 outlet RSS feeds)       normalize,
                         dedupe)
                  └──────── GitHub Actions (cron, 6x/day) ────────┘
```

---

## Methodology

### 1. Collection

A scheduled GitHub Actions workflow (cron, six runs per day) pulls candidate articles from three source types:

- Google News RSS
- World News API
- RSS feeds of individual media outlets

Each record carries the headline, summary or snippet, URL, outlet, and publication date. Outlets are grouped into three categories (state-owned local, independent local, international) to support the comparison in research question 4.

### 2. Cleaning and preprocessing

Operates on each article's **headline and summary**:

1. **Relevance filter:** drops results that do not concern Pax Silica.
2. **Normalization:** text is cleaned and lowercased.
3. **Link decoding:** Google News redirect links are decoded to the canonical article URL.
4. **Deduplication:** removes repeat articles across sources and across runs.

The classifier matches directly against the lowercased string, so there is no tokenization or stopword removal.

### 3. Classification

Classification is **article-level**: each article receives one or more themes and a single stance.

**Themes.** Each article is matched against seven keyword dictionaries (for example, Economic Development and Geopolitical Security). An article may match multiple themes. The dictionaries mix official or policy vocabulary (e.g., "bilateral agreement") with civil-society vocabulary (e.g., "moratorium") so that different stakeholder framings are captured.

**Stance.** Two custom word lists, supportive and critical, are counted per article. The side with the stronger signal wins:

| Stance | Rule | Typical signals |
|---|---|---|
| Supportive | More supportive than critical terms | benefits, opportunities, growth, investment, jobs, backing or welcoming of developments |
| Critical | More critical than supportive terms | concerns, risks, opposition, protests, displacement, scrutiny, backlash |
| Neutral | Tie, or no clear signal | straightforward reporting, explainers, balanced or keyword-free text |

**Why not generic sentiment?** An early version used TextBlob, but general-purpose scoring was too blunt for policy coverage (for instance, it could not distinguish "critical of the project" from "critical infrastructure"). Domain-specific lists replaced it.

**Why rules instead of a trained or embedding-based model?** The research questions center on tracking specific terms and entities, which is a lexical problem. A rule-based approach is sufficient, and every label can be traced to the exact keywords that produced it.

### 4. Validation and maintenance

There is no labeled ground-truth set. Quality control is manual and iterative: classifications are periodically reviewed, misclassifications corrected, and the keyword lists refined. Keyword lists were expanded over time based on what the classifier missed, and this maintenance is ongoing as new coverage and terminology appear.

### 5. Storage and visualization

Classified records are appended to **Google Sheets** through the Sheets API, which serves as the data store. The **Streamlit** dashboard reads the sheet as its live data source, so charts and takeaway titles update as new articles are ingested.

---

## Limitations

- **Small corpus.** Sufficient to demonstrate the pipeline, not to generalize about all coverage.
- **Lexical only.** Cannot detect sarcasm, implied meaning, vernacular phrasing, or anything absent from the keyword lists.
- **Article-level labels.** Mixed viewpoints within a single article are not separated.
- **Uneven text depth.** Google News and RSS entries often provide little beyond the headline. Shorter text has fewer chances to match keywords and is more likely to be labeled Neutral, which can bias stance by source.
- **Single-pass classification.** No embeddings or model-based second pass.
- **Ongoing keyword upkeep** is required for accuracy.

**Planned next step:** full-text analysis.

---

## Getting Started

> Adjust the commands and secret names below to match your repository.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
streamlit run app.py
```

**Configuration.** The pipeline expects credentials for the World News API and a Google service account with access to the target Sheet. In GitHub Actions these are stored as repository secrets; locally, supply them via environment variables or Streamlit secrets.

---

## Tech Stack

Python · pandas · GitHub Actions · Google Sheets API · Streamlit

---

## Context

Individual capstone project for the Eskwelabs Data Analytics program, presented at Demo Day.
