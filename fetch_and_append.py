import os
import json
import requests
import pandas as pd
import gspread
import feedparser

from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta


# ---------------------------------------------------------
# DEBUG
# ---------------------------------------------------------
print("FETCH SCRIPT STARTED")


# ---------------------------------------------------------
# CREDENTIALS
# (from GitHub Actions Secrets, not a local file)
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
# WORLD NEWS API
# ---------------------------------------------------------
WORLD_NEWS_URL = "https://api.worldnewsapi.com/search-news"


# ---------------------------------------------------------
# RSS SOURCES
# ---------------------------------------------------------
RSS_SOURCES = {
    "GMA News": "https://data.gmanews.tv/gno/rss/news/feed.xml",
    "Philippine Daily Inquirer": "https://www.inquirer.net/fullfeed",
    "Manila Bulletin": "https://mb.com.ph/rss/articles",
    "Philippine Star": "https://www.philstar.com/rss/headlines",
    "Rappler": "https://www.rappler.com/feed/",
    "Philippine News Agency": "https://www.pna.gov.ph/latest.rss",  # new
    "Abante": "https://www.abante.com.ph/feed",  # new
}

RSS_KEYWORDS = [
    "pax silica",
    "new clark city",
    "clark freeport",
    "tarlac ai",
    "bcda",
    "economic security zone",
    "henry aguda",
    "hyperscaler",
    "aeta ancestral domain",
    "kalikasan",
]


# ---------------------------------------------------------
# SEARCH TOPICS (World News API)
# ---------------------------------------------------------
# Trimmed to fit the free tier's 50 points/day budget
# while keeping thematic variety.

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
    "DICT Pax Silica",
    "Economic Security Zone",
    "AI data center Philippines",

    # Civil society & environment
    "Makabayan bloc Pax Silica",
    "IBON Foundation Pax Silica",
    "Aeta ancestral domain",
    "desalination data center",
    "water table depletion Luzon",
    "kain suka",
]


# ---------------------------------------------------------
# THEME CLASSIFICATION
# ---------------------------------------------------------
THEME_KEYWORDS = {

    "Economic Development": [
        "investment",
        "jobs",
        "gdp",
        "economic zone",
        "supply chain",
        "semiconductor",
        "hub",
        "trade",
        "manufacturing",
        "corridor",
        "value chain",
        "industrial corridor"
    ],

    "Environmental & Resources Impact": [
        "water",
        "water table",
        "water depletion",
        "water scarcity",
        "energy consumption",
        "energy demand",
        "power consumption",
        "electricity demand",
        "power grid",
        "groundwater",
        "desalination",
        "pollution",
        "emissions",
        "environmental compliance",
        "power grid",
        "land conversion",
        "lng"
        "aeta",
        "ancestral domain",
        "displacement",
        "indigenous",
        "resettlement",
        "land rights"
    ],

    "Technological Advancement": [
        "artificial intelligence",
        "ai infrastructure",
        "data center",
        "hyperscaler",
        "cloud computing",
        "semiconductor",
        "chip",
        "chip design",
        "wafer",
        "fabrication",
        "technology transfer",
        "digital infrastructure",
        "computing infrastructure",
        "advanced manufacturing"
],

"Human Capital & Employment": [
        "workforce",
        "human capital",
        "skills training",
        "workforce development",
        "training",
        "technical skills",
        "vocational training",
        "skilled workers",
        "engineers",
        "technical professionals",
        "talent",
        "jobs",
        "job creation",
        "employment opportunities",
        "upskilling",
        "reskilling"
],

"Supply-Chain Resilience": [
        "supply chain",
        "supply chain resilience",
        "supply chain security",
        "critical minerals",
        "mineral supply",
        "semiconductor supply chain",
        "regional partners",
        "regional integration",
        "diversification",
        "market diversification",
        "supplier diversification",
        "strategic dependencies",
        "coercive dependencies",
        "single market",
        "alternative markets",
        "trusted partners",
        "economic resilience"
],

    "Institutional Governance": [
        "bcda",
        "dict",
        "dti",
        "marcos",
        "bingcang",
        "aguda",
        "bilateral agreement",
        "multilateral agreement",
        "declaration",
        "summit",
        "policy",
        "regulation"
        "oversight",
        "legal framework"
        "civil society",
        "Kalikasan",
        "Makabayan",
        "IBON",
        "protest",
        "opposition",
        "moratorium",
        "activist",
        "walkout",
        "criticize"
    ],

    "Geopolitical Security": [
        "coercive dependencies",
        "civilian industrial zone",
        "supply chain security",
        "geopolitics",
        "national security",
        "strategic"
    ],
}


# ---------------------------------------------------------
# STANCE CLASSIFICATION
# ---------------------------------------------------------
SUPPORTIVE_WORDS = [
    "boost",
    "growth",
    "opportunity",
    "partnership",
    "investment surge",
    "milestone",
    "welcomed",
    "progress",
    "modernization",
    "development",
    "job creation",
    "breakthrough"
]

CRITICAL_WORDS = [
    "displacement",
    "protest",
    "concern",
    "threat",
    "backlash",
    "depletion",
    "violation",
    "harm",
    "risk",
    "opposition",
    "exploitation",
    "controversy",
    "outcry",
    "worse"
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

    support = sum(
        w in text_lower
        for w in SUPPORTIVE_WORDS
    )

    critical = sum(
        w in text_lower
        for w in CRITICAL_WORDS
    )

    if support > critical:
        return "Supportive"

    elif critical > support:
        return "Critical"

    return "Neutral"


# ---------------------------------------------------------
# RELEVANCE FILTER
# ---------------------------------------------------------
def is_relevant(row) -> bool:
    """Keep only articles that actually mention Pax Silica by name."""
    combined = f"{row['title']} {row['description']}".lower()
    return "pax silica" in combined


# ---------------------------------------------------------
# WORLD NEWS API
# ---------------------------------------------------------
def fetch_news(
    topics,
    days_back=20,
    page_size=20
):
    """Fetch news from World News API's search-news endpoint,
    filtered to Philippine sources.
    """

    print("Starting World News API fetch...")

    from_date = (
        datetime.now() - timedelta(days=days_back)
    ).strftime("%Y-%m-%d")

    all_articles = []

    for topic in topics:

        print(
            f"Searching World News API: {topic}"
        )

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

            print(
                f"Request failed for '{topic}': {e}"
            )

            continue

        if resp.status_code != 200:

            print(
                f"World News API error for '{topic}' "
                f"(status {resp.status_code}): "
                f"{resp.text[:150]}"
            )

            continue

        data = resp.json()

        articles = data.get("news", [])

        print(
            f"  → {len(articles)} articles returned"
        )

        for article in articles:

            title = article.get("title") or ""

            text_snippet = (
                article.get("summary")
                or (
                    article.get("text", "") or ""
                )[:300]
            )

            full_text = (
                f"{title} {text_snippet}"
            )

            all_articles.append({

                "topic": topic,

                "title": title,

                "description": text_snippet,

                "source": article.get(
                    "source_country",
                    "Unknown"
                ),

                "url": article.get("url"),

                "published_at": article.get(
                    "publish_date"
                ),

                "themes": classify_themes(
                    full_text
                ),

                "stance": classify_stance(
                    full_text
                ),

                "fetched_at": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            })

    return pd.DataFrame(all_articles)


# ---------------------------------------------------------
# RSS FEEDS (multi-source)
# ---------------------------------------------------------
def fetch_rss_feed(
    source_name: str,
    feed_url: str,
    days_back: int = 20
) -> pd.DataFrame:
    """Fetch and keyword-filter a single RSS feed for
    Pax Silica-related coverage.
    """

    print(f"Starting {source_name} RSS fetch...")

    cutoff_date = datetime.now() - timedelta(days=days_back)

    all_articles = []

    try:

        feed = feedparser.parse(feed_url)

        print(
            f"{source_name} RSS: found "
            f"{len(feed.entries)} feed entries."
        )

        for entry in feed.entries:

            title = entry.get("title") or ""
            description = entry.get("summary") or ""
            url = entry.get("link")

            full_text = f"{title} {description}".lower()

            matched_keywords = [
                keyword
                for keyword in RSS_KEYWORDS
                if keyword in full_text
            ]

            if not matched_keywords:
                continue

            published_at = entry.get("published")
            published_datetime = None

            if entry.get("published_parsed"):

                try:

                    published_datetime = datetime(
                        *entry.published_parsed[:6]
                    )

                except Exception:

                    published_datetime = None

            if (
                published_datetime
                and published_datetime < cutoff_date
            ):
                continue

            themes = classify_themes(full_text)
            stance = classify_stance(full_text)

            all_articles.append({

                "topic": ", ".join(matched_keywords),

                "title": title,

                "description": description,

                "source": source_name,

                "url": url,

                "published_at": published_at,

                "themes": themes,

                "stance": stance,

                "fetched_at": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            })

    except Exception as e:

        print(f"{source_name} RSS request failed: {e}")

    print(
        f"{source_name} RSS returned "
        f"{len(all_articles)} relevant recent articles."
    )

    return pd.DataFrame(all_articles)


def fetch_all_rss_news(days_back: int = 20) -> pd.DataFrame:
    """Fetch and combine all configured RSS sources."""

    dfs = [
        fetch_rss_feed(name, url, days_back)
        for name, url in RSS_SOURCES.items()
    ]

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------
# APPEND NEW ARTICLES TO GOOGLE SHEETS
# ---------------------------------------------------------
def append_new_articles_to_sheet(
    df: pd.DataFrame
):

    if df.empty:

        print(
            "No articles fetched."
        )

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
    # Get existing URLs
    # -----------------------------------------------------
    print(
        "Checking existing Google Sheets records..."
    )

    existing = sheet.get_all_records()

    existing_urls = (
        {
            row["url"]
            for row in existing
            if row.get("url")
        }
        if existing
        else set()
    )

    print(
        f"Existing URLs in Google Sheets: "
        f"{len(existing_urls)}"
    )

    # -----------------------------------------------------
    # Remove duplicates within current fetch
    # -----------------------------------------------------
    before_dedup = len(df)

    df = df.drop_duplicates(
        subset=["url"],
        keep="first"
    )

    print(
        f"Removed "
        f"{before_dedup - len(df)} "
        f"duplicate URLs from current fetch."
    )

    # -----------------------------------------------------
    # Remove articles already in Google Sheets
    # -----------------------------------------------------
    before_existing_filter = len(df)

    df = df[
        ~df["url"].isin(existing_urls)
    ]

    print(
        f"Removed "
        f"{before_existing_filter - len(df)} "
        f"articles already in Google Sheets."
    )

    if df.empty:

        print(
            "No new unique articles to append."
        )

        return

    # -----------------------------------------------------
    # Add header if sheet is empty
    # -----------------------------------------------------
    if not existing:

        sheet.append_row(
            columns
        )

    # -----------------------------------------------------
    # Append new articles
    # -----------------------------------------------------
    sheet.append_rows(
        df.fillna("").values.tolist()
    )

    print(
        f"Appended {len(df)} "
        f"new articles to Google Sheets."
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":

    print(
        "Fetching news from World News API..."
    )

    api_df = fetch_news(
        default_topics
    )

    print(
        f"World News API returned "
        f"{len(api_df)} articles."
    )

    print(
        "Fetching news from RSS feeds..."
    )

    rss_df = fetch_all_rss_news()

    print(
        f"RSS feeds returned "
        f"{len(rss_df)} relevant recent articles total."
    )

    # -----------------------------------------------------
    # Combine API + RSS
    # -----------------------------------------------------
    df = pd.concat(
        [
            api_df,
            rss_df
        ],
        ignore_index=True
    )

    print(
        f"Total articles before "
        f"relevance filtering: {len(df)}"
    )

    # -----------------------------------------------------
    # Keep only articles that actually mention Pax Silica
    # -----------------------------------------------------
    before_relevance_filter = len(df)

    if not df.empty:
        df = df[df.apply(is_relevant, axis=1)]

    print(
        f"Removed "
        f"{before_relevance_filter - len(df)} "
        f"articles not mentioning 'Pax Silica' by name."
    )

    print(
        f"Total articles before deduplication: {len(df)}"
    )

    # -----------------------------------------------------
    # Append only new articles
    # -----------------------------------------------------
    append_new_articles_to_sheet(
        df
    )
