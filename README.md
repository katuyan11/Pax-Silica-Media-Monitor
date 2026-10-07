# Pax Silica Monitor

A rule-based news monitoring pipeline that tracks themes and stance in Philippine and international media coverage of the Pax Silica initiative. It ingests articles six times a day, classifies them with transparent keyword dictionaries, stores results in Google Sheets, and serves a live Streamlit dashboard.

**[Live dashboard](https://newsmonitoringnlp.streamlit.app/)** (best viewed on a laptop at 90% browser zoom)

---

## Overview

Pax Silica is a U.S.-led initiative on technology, AI infrastructure, and critical-mineral supply chains, with the Philippines positioned to play a role. Coverage moves quickly and each stakeholder group frames it differently, which makes manual tracking difficult.

This is a topic-specific monitor built to answer four exploratory questions:

1. What themes appear in coverage, and which receive the most attention?
2. Is coverage supportive, critical, or neutral?
3. How has the story changed over time?
4. How do state-owned local, independent local, and international outlets differ in their reporting?

Business value: Journalists, researchers, and interested individuals without access to commercial media-intelligence platforms.

| | |
|---|---|
| **Corpus** | News articles, columns, editorials, press releases, organizational statements, and explainer articles (grows daily) |
| **Refresh rate** | 6 runs per day via GitHub Actions |
| **Classifier** | Rule-based keyword matching (no trained model) |
| **Stack** | Python, pandas, feedparser, gspread, GitHub Actions, Google Sheets, Streamlit |

---

## Repository Structure

| Path | Purpose |
|---|---|
| `fetch_and_append.py` | Pipeline: collect, filter, classify, deduplicate, append to Google Sheets |
| `reclassify_sheet.py` | Re-applies the classifier to existing sheet rows after keyword changes |
| `streamlit_app.py` | Dashboard reading the sheet as its live data source |
| `.github/workflows/daily_fetch.yml` | Scheduled workflow that runs the pipeline |
| `requirements.txt` | Python dependencies |

---

## Architecture

```
Google News RSS ─┐
Outlet RSS feeds ─┼─► Relevance filter ─► Classify ─► Deduplicate ─► Google Sheets ─► Streamlit
World News API ──┘   (anchor terms,      (themes +    (URL, source+   (Clean_Data)     dashboard
                      date, exclusions)    stance)      title, title)
        └──────────── GitHub Actions (scheduled, 6x/day) ────────────┘
```

---

## Methodology

### 1. Collection

Three source types are pulled on each run:

- **Google News RSS** (Philippine edition, `hl=en-PH&gl=PH`): one search per query in a list of 14 topic queries (e.g. "Pax Silica", "New Clark City", "BCDA", "Aeta ancestral domain", "AI data center Philippines"), chosen to capture both official and civil-society angles.
- **Outlet RSS feeds:** ten feeds from GMA, Inquirer (four sections), Manila Bulletin, Philstar, Rappler, PNA, and Abante. Entries are pre-screened against an RSS keyword list before further filtering.
- **World News API:** same topic queries, restricted to English-language Philippine sources. It is quota-limited, so it is **off by default** and only runs when `RUN_WORLD_NEWS=true`, which allows an RSS-only schedule for most runs.

Each record stores: `topic`, `title`, `description`, `source`, `url`, `published_at`, `themes`, `stance`, `fetched_at`. Publication timestamps from all sources are normalized to `Asia/Manila` (`YYYY-MM-DD HH:MM:SS`); feeds without a timezone offset are assumed to be UTC.

### 2. Relevance filtering

Classification only runs on the article's **headline and summary**. An article is kept if, after an exclusion check (e.g. job listings, property ads, entertainment, sports), it either:

- contains the anchor term **"pax silica"**, or
- contains **two or more** secondary terms (e.g. "new clark city", "economic security zone", "aeta ancestral domain").

Google News RSS is stricter: because its keyword search is loose and returns unrelated BCDA/New Clark City stories, it requires the literal anchor term. It also drops anything published before **December 1, 2025**, since no genuine Pax Silica coverage can predate it.

### 3. Text handling (Google News)

- **Link decoding:** Google News returns redirect tokens rather than publisher URLs, so each is decoded with `googlenewsdecoder`. Filtering on raw text happens *before* decoding because decoding is the slow, rate-limited step. Decoded links are cached within a run, and entries that fail to decode are skipped and picked up on a later run. Keeping an unresolved link previously caused duplicate appends.
- **Title cleanup:** the trailing " - Source Name" is stripped from Google News titles so outlet names don't pollute term analysis.
- **HTML stripping:** Google News summaries contain markup, which is removed.
- **Outlet names:** taken from the feed's source field where available, otherwise derived from the URL domain.

### 4. Classification

Classification is **article-level**: each article receives zero or more themes and exactly one stance. Keywords are compiled once into whole-word regular expressions:

- **Whole-word boundaries** (`(?<!\w)term(?!\w)`) prevent false hits such as "dict" matching "predict". Lookarounds are used instead of `\b` so terms ending in punctuation (e.g. "Rep.", "Gov't") still match.
- **Case-insensitive** by default. The one exception is "US", which is case-sensitive so it doesn't match the pronoun "us".
- **Duplicate keywords are removed** before compiling so a repeated entry can't count twice.
- Text is whitespace-collapsed (not lowercased) before matching. There is no tokenization or stopword removal.

**Themes.** Each article is checked against seven dictionaries. Theme matching is boolean per theme, so an article can match several, and theme patterns also accept plural forms (e.g. "data center" matches "data centers"). Articles matching none are labeled `Uncategorized`.

| Theme | Example vocabulary |
|---|---|
| Economic Development | investment, economic zone, manufacturing, industrial corridor |
| Environmental & Resource Impact | water table, energy demand, displacement, ancestral domain |
| Technological Advancement | AI infrastructure, data center, hyperscaler, chip design |
| Human Capital & Employment | workforce, upskilling, skilled workers, job creation |
| Supply-Chain Resilience | critical minerals, diversification, trusted partners |
| Institutional Governance | bilateral agreement, oversight, civil society, moratorium, named agencies and officials |
| Geopolitical Security | national security, sovereignty, strategic dependence, economic security |

The Institutional Governance dictionary combines official vocabulary with civil-society and protest terms, so that different stakeholder framings are captured.

**Stance.** Two custom lists, supportive and critical, are scored by counting how many *distinct* listed terms appear in the article (repeated occurrences of one term count once). Stance lists use exact whole-word matches only, with variants listed explicitly ("protest", "protests", "protested"), to avoid double-counting.

| Stance | Rule |
|---|---|
| Supportive | More supportive than critical terms matched |
| Critical | More critical than supportive terms matched |
| Neutral | Tie, or no matches |

**Why not generic sentiment?** An early version used TextBlob, but general-purpose scoring was too blunt for policy coverage (it could not distinguish "critical of the project" from "critical infrastructure"). Domain-specific lists replaced it.

**Why rules instead of a trained or embedding-based model?** The research questions center on tracking specific terms and entities, which is a lexical problem. A rule-based approach is sufficient, and every label can be traced to the exact keywords that produced it.

### 5. Deduplication

The same story can surface through several queries and sources, sometimes with different URLs or outlet labels. Within a run, rows are deduplicated by URL. Against the existing sheet, a new article is dropped if **any** of three checks matches:

1. identical URL
2. identical normalized `source + title`
3. identical normalized `title` alone (catches outlet-name variations between fetches)

### 6. Storage and visualization

New rows are appended to a **Google Sheets** tab (`Clean_Data`) through the Sheets API via `gspread`, which acts as the data store. `streamlit_app.py` reads that sheet, so charts and takeaway titles update as articles arrive.

### 7. Validation and maintenance

There is no labeled ground-truth set. Quality control is manual and iterative: classifications are periodically reviewed, misclassifications corrected, and keyword lists refined as new coverage and terminology appear. After list changes, `reclassify_sheet.py` is used to bring existing rows in line with the current rules.

---

## Limitations

- **Small corpus.** Sufficient to demonstrate the pipeline, not to generalize about all coverage.
- **Lexical only.** Cannot detect sarcasm, implied meaning, vernacular phrasing, or anything absent from the keyword lists.
- **Article-level labels.** Mixed viewpoints within a single article are not separated.
- **Uneven text depth.** Google News and RSS entries often provide little beyond the headline. Shorter text has fewer chances to match keywords and is more likely to be labeled Neutral, which can bias stance by source.
- **Single-pass classification.** No embeddings or model-based second pass.
- **Ongoing keyword upkeep** is required for accuracy.

---

## Getting Started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

**Environment variables** (stored as GitHub Actions secrets in production):

| Variable | Purpose |
|---|---|
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Service account credentials (JSON string) with access to the target sheet |
| `GOOGLE_SHEET_ID` | ID of the Google Sheet used as the data store |
| `WORLD_NEWS_API_KEY` | World News API key (only needed if enabled) |
| `RUN_WORLD_NEWS` | Set to `true` to include World News API; defaults to off |

**Run the pipeline and dashboard locally:**

```bash
python fetch_and_append.py
streamlit run streamlit_app.py
```

---

## Context

Individual capstone project for the Eskwelabs Data Analytics program, presented at Demo Day.
