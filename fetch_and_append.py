import os
import json
import requests
import pandas as pd
import gspread
import feedparser
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta

# ---------------------------------------------------------
# CREDENTIALS (from GitHub Actions Secrets, not a local file)
# ---------------------------------------------------------
WORLD_NEWS_API_KEY = os.environ["WORLD_NEWS_API_KEY"].strip()

print(
    f"DEBUG: key length = {len(WORLD_NEWS_API_KEY)}, "
    f"first 4 chars = {WORLD_NEWS_API_KEY[:4]}, "
    f"last 4 chars = {WORLD_NEWS_API_KEY[-4:]}"
)

GOOGLE_SERVICE_ACCOUNT_JSON = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
SHEET_ID = os.environ["GOOGLE_SHEET_ID"]

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

creds_dict = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
creds = Credentials.from_service_account_info(
    creds_dict,
    scopes=SCOPES
)

gc = gspread.authorize(creds)
sheet = gc.open_by_key(SHEET_ID).sheet1

# ---------------------------------------------------------
# API / RSS SOURCES
# ---------------------------------------------------------
WORLD_NEWS_URL = "https://api.worldnewsapi.com/search-news"

GMA_RSS_URL = "https://data.gmanews.tv/gno/rss/news/feed.xml"

# ---------------------------------------------------------
# SEARCH TOPICS
# ---------------------------------------------------------
# Trimmed to 12 topics to fit the free tier's 50 points/day budget
# (1 point per search call), while keeping thematic variety.

default_topics = [
    # Core identity
    "Pax Silica",
    "Pax Silica Philippines",
    "Pax Silica Summit",

    # Location & project terms
    "New Clark City",
    "Clark Freeport",
    "Tarlac AI hub",

    # Government / policy
    "BCDA",
    "Economic Security Zone",
    "Henry Aguda",

    # Civil society & environment
    "Kalikasan",
    "Aeta ancestral domain",
    "hyperscaler moratorium",
]

# ---------------------------------------------------------
# THEME CLASSIFICATION
# ---------------------------------------------------------
THEME_KEYWORDS = {
    "Economic / Investment": [
        "investment", "jobs", "gdp", "economic zone", "supply chain",
        "semiconductor", "hub", "trade", "manufacturing", "corridor"
    ],

    "Environmental Impact": [
        "water", "groundwater", "desalination", "pollution", "emissions",
        "environmental compliance", "power grid", "land conversion", "lng"
    ],

    "Indigenous Rights / Displacement": [
        "aeta", "ancestral domain", "displacement", "indigenous",
        "resettlement", "land rights"
    ],

    "Civil Society / Opposition": [
        "kalikasan", "makabayan", "ibon", "protest", "opposition",
        "moratorium", "activist", "criticiz"
    ],

    "Government / Policy": [
        "bcda", "dict", "dti", "marcos", "bingcang", "aguda",
        "declaration", "summit", "policy", "regulation"
    ],

    "Geopolitics / Security": [
        "coercive dependencies", "supply chain security", "geopolit",
        "national security", "strategic"
    ],
}

# ---------------------------------------------------------
# STANCE CLASSIFICATION
# ---------------------------------------------------------
SUPPORTIVE_WORDS = [
    "boost", "growth", "opportunity", "partnership", "investment surge",
    "milestone", "welcomed", "progress", "modernization", "development",
    "job creation", "breakthrough"
]

CRITICAL_WORDS = [
    "displacement", "protest", "concern", "threat", "backlash",
    "depletion", "violation", "harm", "risk", "opposition",
    "exploitation", "controversy", "outcry"
]

# ---------------------------------------------------------
# THEME CLASSIFIER
# ---------------------------------------------------------
def classify_themes(text: str) -> str:
    text_lower = text.lower()

    matched = [
        theme
        for theme, kws in THEME_KEYWORDS.items()
        if any(k in text_lower for k in kws)
    ]

    return ", ".join(matched) if matched else "Uncategorized"


# ---------------------------------------------------------
# STANCE CLASSIFIER
# ---------------------------------------------------------
def classify_stance(text: str) -> str:
    text_lower = text.lower()

    support = sum(w in text_lower for w in SUPPORTIVE_WORDS)
    critical = sum(w in text_lower for w in CRITICAL_WORDS)

    if support > critical:
        return "Supportive"

    elif critical > support:
        return "Critical"

    return "Neutral"


# ---------------------------------------------------------
# WORLD NEWS API
# ---------------------------------------------------------
def fetch_news(topics, days_back=20, page_size=20):
    """Fetch news from World News API's search-news endpoint,
    filtered to Philippine sources.

    days_back is kept short because this runs daily and only
    needs to capture recent articles.
    """

    from_date = (
        datetime.now() - timedelta(days=days_back)
    ).strftime("%Y-%m-%d")

    all_articles = []

    for topic in topics:

        params = {
            "api-key": WORLD_NEWS_API_KEY,
            "text": topic,
            "source-country": "ph",
            "language": "en",
            "earliest-publish-date": from_date,
            "number": page_size,
        }

        try:
            resp = requests.get(
                WORLD_NEWS_URL,
                params=params,
                timeout=10
            )

        except requests.exceptions.RequestException as e:
            print(f"Request failed for '{topic}': {e}")
            continue

        if resp.status_code != 200:
            print(
                f"World News API error for '{topic}' "
                f"(status {resp.status_code}): "
                f"{resp.text[:150]}"
            )
            continue

        data = resp.json()

        for article in data.get("news", []):

            title = article.get("title") or ""

            text_snippet = (
                article.get("summary")
                or (article.get("text", "") or "")[:300]
            )

            full_text = f"{title} {text_snippet}"

            all_articles.append({
                "topic": topic,
                "title": title,
                "description": text_snippet,
                "source": article.get(
                    "source_country",
                    "Unknown"
                ),
                "url": article.get("url"),
                "published_at": article.get("publish_date"),
                "themes": classify_themes(full_text),
                "stance": classify_stance(full_text),
                "fetched_at": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            })

    return pd.DataFrame(all_articles)


# ---------------------------------------------------------
# GMA RSS
# ---------------------------------------------------------
def fetch_rss_news(days_back=20):
    """Fetch recent articles from the GMA News RSS feed.

    Only articles published within the specified lookback period
    are included.
    """

    cutoff_date = datetime.now() - timedelta(days=days_back)

    all_articles = []

    try:
        feed = feedparser.parse(GMA_RSS_URL)

        print(
            f"GMA RSS: found {len(feed.entries)} feed entries."
        )

        for entry in feed.entries:

            title = entry.get("title") or ""
            description = entry.get("summary") or ""
            url = entry.get("link")

            # -------------------------------------------------
            # Parse publication date
            # -------------------------------------------------
            published_at = entry.get("published")

            published_datetime = None

            if entry.get("published_parsed"):
                try:
                    published_datetime = datetime(
                        *entry.published_parsed[:6]
                    )
                except Exception:
                    published_datetime = None

            # Skip articles older than the lookback period
            if published_datetime and published_datetime < cutoff_date:
                continue

            full_text = f"{title} {description}"

            all_articles.append({
                "topic": "GMA RSS",
                "title": title,
                "description": description,
                "source": "GMA News",
                "url": url,
                "published_at": published_at,
                "themes": classify_themes(full_text),
                "stance": classify_stance(full_text),
                "fetched_at": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            })

    except Exception as e:
        print(f"GMA RSS request failed: {e}")

    return pd.DataFrame(all_articles)


# ---------------------------------------------------------
# APPEND NEW ARTICLES TO GOOGLE SHEETS
# ---------------------------------------------------------
def append_new_articles_to_sheet(df: pd.DataFrame):

    if df.empty:
        print("No articles fetched.")
        return

    # -----------------------------------------------------
    # Ensure exact Google Sheets column order
    # -----------------------------------------------------
    columns = [
        "topic",
        "title",
        "description",
        "source",
        "url",
        "published_at",
        "themes",
        "stance",
        "fetched_at",
    ]

    df = df[columns]

    # -----------------------------------------------------
    # Get existing URLs from Google Sheets
    # -----------------------------------------------------
    existing = sheet.get_all_records()

    existing_urls = (
        {row["url"] for row in existing if row.get("url")}
        if existing
        else set()
    )

    # -----------------------------------------------------
    # Remove duplicate URLs within the newly fetched data
    # -----------------------------------------------------
    df = df.drop_duplicates(
        subset=["url"],
        keep="first"
    )

    # -----------------------------------------------------
    # Remove articles already in Google Sheets
    # -----------------------------------------------------
    df = df[~df["url"].isin(existing_urls)]

    if df.empty:
        print("No new unique articles to append.")
        return

    # -----------------------------------------------------
    # Add header if sheet is empty
    # -----------------------------------------------------
    if not existing:
        sheet.append_row(columns)

    # -----------------------------------------------------
    # Append new articles
    # -----------------------------------------------------
    sheet.append_rows(
        df.fillna("").values.tolist()
    )

    print(
        f"Appended {len(df)} new articles to Google Sheets."
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":

    print("Fetching news from World News API...")

    api_df = fetch_news(default_topics)

    print(
        f"World News API returned {len(api_df)} articles."
    )

    print("Fetching news from GMA RSS...")

    rss_df = fetch_rss_news()

    print(
        f"GMA RSS returned {len(rss_df)} recent articles."
    )

    # -----------------------------------------------------
    # Combine API and RSS results
    # -----------------------------------------------------
    df = pd.concat(
        [api_df, rss_df],
        ignore_index=True
    )

    print(
        f"Total articles before deduplication: {len(df)}"
    )

    # -----------------------------------------------------
    # Append only new articles
    # -----------------------------------------------------
    append_new_articles_to_sheet(df)
