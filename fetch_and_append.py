import os
import json
import requests
import pandas as pd
import gspread
import feedparser

from urllib.parse import urlparse
from google.oauth2.service_account import Credentials
from datetime import datetime, timezone


# ============================================================
# CONFIG
# ============================================================

WORLD_NEWS_URL = "https://api.worldnewsapi.com/search-news"

SHEET_TAB_NAME = "Clean_Data"

RSS_FEEDS = {
    "GMA News": "https://www.gmanetwork.com/news/rss/",
    "Inquirer Newsinfo": "https://newsinfo.inquirer.net/feed",
    "Inquirer Business": "https://business.inquirer.net/feed",
    "Inquirer Global Nation": "https://globalnation.inquirer.net/feed",
    "Manila Bulletin": "https://mb.com.ph/feed/",
    "Philstar": "https://www.philstar.com/rss/headlines",
    "Rappler": "https://www.rappler.com/feed/",
    "PNA": "https://www.pna.gov.ph/rss",
    "Abante": "https://www.abante.com.ph/feed/",
}


# ============================================================
# KEYWORDS
# ============================================================

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

DEFAULT_TOPICS = [
    "Pax Silica",
    "New Clark City",
    "Clark Freeport",
    "Tarlac AI hub",
    "BCDA",
    "DICT Pax Silica",
    "Economic Security Zone",
    "AI data center Philippines",
    "Makabayan bloc Pax Silica",
    "IBON Foundation Pax Silica",
    "Aeta ancestral domain",
    "desalination data center",
    "water table depletion Luzon",
    "kain suka",
]


# ============================================================
# THEME CLASSIFICATION
# ============================================================

THEME_KEYWORDS = {
    "Economic Development": [
        "investment", "jobs", "gdp", "economic zone", "supply chain",
        "semiconductor", "hub", "trade", "manufacturing", "corridor",
        "value chain", "industrial corridor",
    ],
    "Environmental & Resource Impact": [
        "water", "water table", "water depletion", "water scarcity",
        "energy consumption", "energy demand", "power consumption",
        "electricity demand", "power grid", "groundwater", "desalination",
        "pollution", "emissions", "environmental compliance",
        "land conversion", "lng", "aeta", "ancestral domain",
        "displacement", "indigenous", "resettlement", "land rights",
    ],
    "Technological Advancement": [
        "artificial intelligence", "ai infrastructure", "data center",
        "hyperscaler", "cloud computing", "semiconductor", "chip",
        "chip design", "wafer", "fabrication", "technology transfer",
        "digital infrastructure", "computing infrastructure",
        "advanced manufacturing",
    ],
    "Human Capital & Employment": [
        "workforce", "human capital", "skills training",
        "workforce development", "training", "technical skills",
        "vocational training", "skilled workers", "engineers",
        "technical professionals", "talent", "jobs", "job creation",
        "employment opportunities", "upskilling", "reskilling",
    ],
    "Supply-Chain Resilience": [
        "supply chain", "supply chain resilience", "supply chain security",
        "critical minerals", "mineral supply", "semiconductor supply chain",
        "regional partners", "regional integration", "diversification",
        "market diversification", "supplier diversification",
        "strategic dependencies", "coercive dependencies", "single market",
        "alternative markets", "trusted partners", "economic resilience",
    ],
    "Institutional Governance": [
        "bcda", "dict", "dti", "marcos", "bingcang", "aguda",
        "bilateral agreement", "multilateral agreement", "declaration",
        "summit", "policy", "regulation", "oversight", "legal framework",
        "civil society", "kalikasan", "makabayan", "ibon", "protest",
        "opposition", "moratorium", "activist", "walkout", "dialogue",
        "multi-stakeholder dialogue", "criticize",
    ],
    "Geopolitical Security": [
        "coercive dependencies", "civilian industrial zone",
        "supply chain security", "geopolitics", "national security",
        "strategic alignment", "strategic dependence",
        "strategic dependency", "security implications",
        "economic security", "china",
    ],
}

SUPPORTIVE_WORDS = [
    "support", "supports", "supported", "back", "backs", "backed",
    "welcome", "welcomes", "welcomed", "approve", "approved", "benefit",
    "benefits", "opportunity", "opportunities", "growth", "investment",
    "job creation", "development", "expansion",
]

CRITICAL_WORDS = [
    "oppose", "opposes", "opposed", "criticize", "criticizes",
    "criticized", "criticism", "concern", "concerns", "risk", "risks",
    "threat", "threatens", "environmental damage", "displacement",
    "pollution", "depletion", "moratorium", "protest", "protests",
    "protested", "reject", "rejected",
]


# ============================================================
# RELEVANCE FILTER
# ============================================================

ANCHOR_TERMS = ["pax silica"]

EXCLUDE_TERMS = [
    "cayetano", "cebu pacific", "condo for sale", "house and lot",
    "job vacancy", "job opening", "hiring now", "flight promo",
    "concert", "basketball", "showbiz",
]

SECONDARY_TERMS = [
    "new clark city", "clark freeport", "tarlac ai hub",
    "economic security zone", "bcda", "aeta ancestral domain",
    "semiconductor hub", "data center philippines",
]

def is_relevant(row) -> bool:
    combined = f"{row.get('title', '')} {row.get('description', '')}".lower()

    if any(term in combined for term in EXCLUDE_TERMS):
        return False

    if any(term in combined for term in ANCHOR_TERMS):
        return True

    secondary_hits = sum(term in combined for term in SECONDARY_TERMS)
    return secondary_hits >= 2


# ============================================================
# OUTLET NAME HELPER
# ============================================================

def get_outlet_name(url: str) -> str:
    """Extract a readable outlet name from the article URL's domain,
    since World News API does not return a clean publisher name field."""
    try:
        domain = urlparse(url).netloc.replace("www.", "")
        return domain.split(".")[0].replace("-", " ").title()
    except Exception:
        return "Unknown"


# ============================================================
# CLASSIFIERS
# ============================================================

def classify_themes(text):
    text = str(text or "").lower()
    matched_themes = [
        theme for theme, keywords in THEME_KEYWORDS.items()
        if any(keyword.lower() in text for keyword in keywords)
    ]
    return ", ".join(matched_themes) if matched_themes else "Uncategorized"

def classify_stance(text):
    text = str(text or "").lower()
    supportive_hits = sum(k in text for k in SUPPORTIVE_WORDS)
    critical_hits = sum(k in text for k in CRITICAL_WORDS)
    if supportive_hits > critical_hits:
        return "Supportive"
    if critical_hits > supportive_hits:
        return "Critical"
    return "Neutral"


# ============================================================
# GOOGLE SHEETS
# ============================================================

EXPECTED_COLUMNS = [
    "topic", "title", "description", "source", "url",
    "published_at", "themes", "stance", "fetched_at",
]

def get_google_sheet():
    service_account_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")

    if not service_account_json:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON environment variable is missing.")
    if not sheet_id:
        raise RuntimeError("GOOGLE_SHEET_ID environment variable is missing.")

    credentials_info = json.loads(service_account_json)
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = Credentials.from_service_account_info(credentials_info, scopes=scopes)
    gc = gspread.authorize(credentials)
    spreadsheet = gc.open_by_key(sheet_id)

    try:
        worksheet = spreadsheet.worksheet(SHEET_TAB_NAME)
    except gspread.WorksheetNotFound:
        print(f"Worksheet '{SHEET_TAB_NAME}' not found. Creating it now...")
        worksheet = spreadsheet.add_worksheet(
            title=SHEET_TAB_NAME, rows=1000, cols=len(EXPECTED_COLUMNS)
        )
        worksheet.append_row(EXPECTED_COLUMNS, value_input_option="USER_ENTERED")

    return worksheet

def ensure_headers(sheet):
    first_row = sheet.row_values(1)
    if not first_row:
        sheet.append_row(EXPECTED_COLUMNS, value_input_option="USER_ENTERED")
        return
    if first_row[:len(EXPECTED_COLUMNS)] != EXPECTED_COLUMNS:
        print("Warning: Clean_Data headers do not match expected column structure.")
        print(f"Existing headers: {first_row}")
        print(f"Expected headers: {EXPECTED_COLUMNS}")


# ============================================================
# WORLD NEWS API
# ============================================================

def fetch_world_news():
    api_key = os.environ.get("WORLD_NEWS_API_KEY")
    if not api_key:
        raise RuntimeError("WORLD_NEWS_API_KEY environment variable is missing.")

    articles = []

    for topic in DEFAULT_TOPICS:
        print(f"Fetching World News API topic: {topic}")

        params = {
            "text": topic,
            "language": "en",
            "source-country": "ph",   # restored — was missing, causing non-PH results
            "number": 50,
        }
        headers = {"x-api-key": api_key}

        try:
            response = requests.get(WORLD_NEWS_URL, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"World News API error for '{topic}': {e}")
            continue

        for article in data.get("news", []):
            title = (article.get("title") or "").strip()
            description = (article.get("summary") or article.get("text") or "").strip()
            url = (article.get("url") or "").strip()

            if not title or not url:
                continue

            row = {
                "topic": topic,
                "title": title,
                "description": description,
                "source": get_outlet_name(url),   # parsed from domain, not a nonexistent API field
                "url": url,
                "published_at": (article.get("publish_date") or article.get("published") or ""),
            }

            if not is_relevant(row):
                continue

            full_text = f"{title} {description}"
            row["themes"] = classify_themes(full_text)
            row["stance"] = classify_stance(full_text)
            row["fetched_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            articles.append(row)

    return articles


# ============================================================
# RSS
# ============================================================

def fetch_rss_articles():
    articles = []

    for source_name, feed_url in RSS_FEEDS.items():
        print(f"Fetching RSS: {source_name}")
        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            print(f"RSS error for {source_name}: {e}")
            continue

        for entry in feed.entries:
            title = (entry.get("title") or "").strip()
            description = (entry.get("summary") or entry.get("description") or "").strip()
            url = (entry.get("link") or "").strip()

            if not title or not url:
                continue

            combined = f"{title} {description}".lower()
            if not any(keyword.lower() in combined for keyword in RSS_KEYWORDS):
                continue

            row = {
                "topic": "RSS",
                "title": title,
                "description": description,
                "source": source_name,
                "url": url,
                "published_at": (entry.get("published") or entry.get("updated") or ""),
            }

            if not is_relevant(row):
                continue

            full_text = f"{title} {description}"
            row["themes"] = classify_themes(full_text)
            row["stance"] = classify_stance(full_text)
            row["fetched_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            articles.append(row)

    return articles


# ============================================================
# NORMALIZE DATA
# ============================================================

def normalize_articles(articles):
    if not articles:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    df = pd.DataFrame(articles)
    for column in EXPECTED_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df = df[EXPECTED_COLUMNS]

    df = df[df["url"].fillna("").astype(str).str.strip() != ""]
    df = df.drop_duplicates(subset=["url"], keep="first")

    return df


# ============================================================
# APPEND ONLY NEW ARTICLES
# ============================================================

def append_new_articles_to_sheet(df, sheet):
    if df.empty:
        print("No relevant articles found.")
        return

    ensure_headers(sheet)

    print(f"Reading existing URLs from {SHEET_TAB_NAME}...")
    existing_records = sheet.get_all_records()
    existing_urls = {
        str(record.get("url", "")).strip()
        for record in existing_records
        if str(record.get("url", "")).strip()
    }

    print(f"Existing articles in {SHEET_TAB_NAME}: {len(existing_urls)}")

    new_df = df[~df["url"].isin(existing_urls)].copy()

    if new_df.empty:
        print("No new articles to append.")
        return

    rows = new_df[EXPECTED_COLUMNS].fillna("").values.tolist()
    sheet.append_rows(rows, value_input_option="USER_ENTERED")

    print(f"Appended {len(rows)} new articles to '{SHEET_TAB_NAME}'.")
    print("Themes found in appended articles:")

    theme_counts = new_df["themes"].fillna("").str.split(", ").explode().value_counts()
    for theme, count in theme_counts.items():
        print(f"  {theme}: {count}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PAX SILICA PHILIPPINE NEWS FETCH")
    print("=" * 60)
    print(f"Target Google Sheets tab: {SHEET_TAB_NAME}")

    sheet = get_google_sheet()
    ensure_headers(sheet)

    world_news_articles = fetch_world_news()
    print(f"World News relevant articles: {len(world_news_articles)}")

    rss_articles = fetch_rss_articles()
    print(f"RSS relevant articles: {len(rss_articles)}")

    all_articles = world_news_articles + rss_articles
    print(f"Total relevant articles before deduplication: {len(all_articles)}")

    df = normalize_articles(all_articles)
    print(f"Unique relevant articles: {len(df)}")

    append_new_articles_to_sheet(df, sheet)

    print("=" * 60)
    print("FETCH COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
