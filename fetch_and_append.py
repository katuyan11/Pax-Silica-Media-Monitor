import os
import json
import requests
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta

# ---------------------------------------------------------
# CREDENTIALS (from GitHub Actions Secrets, not a local file)
# ---------------------------------------------------------
NEWS_API_KEY = os.environ["NEWS_API_KEY"]
GOOGLE_SERVICE_ACCOUNT_JSON = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]  # full JSON as a string
SHEET_ID = os.environ["GOOGLE_SHEET_ID"]

SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds_dict = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
gc = gspread.authorize(creds)
sheet = gc.open_by_key(SHEET_ID).sheet1

NEWS_API_URL = "https://newsapi.org/v2/top-headlines"  # changed from /v2/everything

def fetch_news(topics, page_size=20):
    all_articles = []

    for topic in topics:
        params = {
            "q": f'"{topic}"',
            "country": "ph",       # restricts to PH-tagged sources
            "language": "en",       # note: language + country together can be redundant/conflicting on some plans, safe to drop this if it errors
            "pageSize": page_size,
            "apiKey": NEWS_API_KEY,
        }

default_topics = [
    "Pax Silica", "Pax Silica Summit", "Pax Silica Declaration", "Jacob Helberg",
    "silicon supply chain", "AI supply chain", "coercive dependencies",
    "Pax Silica Philippines", "Pax Silica hub", "Economic Security Zone",
    "New Clark City", "Clark Freeport", "Luzon Economic Corridor",
    "Tarlac AI hub", "Capas Tarlac", "Central Luzon data center",
    "BCDA", "Bases Conversion and Development Authority", "DICT Pax Silica",
    "Henry Aguda", "DTI Pax Silica", "Bingcang BCDA", "Marcos Pax Silica",
    "Kalikasan", "Makabayan bloc Pax Silica", "IBON Foundation",
    "Aeta ancestral domain", "indigenous peoples displacement Tarlac",
    "Environmental Compliance Certificate", "data center water consumption",
    "water table depletion Luzon", "desalination data center",
    "groundwater extraction Philippines", "data center power grid Philippines",
    "Malampaya LNG", "Nueva Ecija power plant", "noise pollution data center",
    "agricultural land conversion Tarlac", "food security Luzon industrialization",
    "kain suka", "AI data center Philippines", "hyperscaler moratorium",
    "semiconductor jobs Philippines",
]

THEME_KEYWORDS = {
    "Economic / Investment": ["investment", "jobs", "gdp", "economic zone", "supply chain",
                               "semiconductor", "hub", "trade", "manufacturing", "corridor"],
    "Environmental Impact": ["water", "groundwater", "desalination", "pollution", "emissions",
                              "environmental compliance", "power grid", "land conversion", "lng"],
    "Indigenous Rights / Displacement": ["aeta", "ancestral domain", "displacement", "indigenous",
                                         "resettlement", "land rights"],
    "Civil Society / Opposition": ["kalikasan", "makabayan", "ibon", "protest", "opposition",
                                    "moratorium", "activist", "criticiz"],
    "Government / Policy": ["bcda", "dict", "dti", "marcos", "bingcang", "aguda",
                             "declaration", "summit", "policy", "regulation"],
    "Geopolitics / Security": ["coercive dependencies", "supply chain security", "geopolit",
                                "national security", "strategic"],
}

SUPPORTIVE_WORDS = ["boost", "growth", "opportunity", "partnership", "investment surge",
                     "milestone", "welcomed", "progress", "modernization", "development",
                     "job creation", "breakthrough"]

CRITICAL_WORDS = ["displacement", "protest", "concern", "threat", "backlash",
                   "depletion", "violation", "harm", "risk", "opposition",
                   "exploitation", "controversy", "outcry"]

def classify_themes(text: str) -> str:
    text_lower = text.lower()
    matched = [theme for theme, kws in THEME_KEYWORDS.items() if any(k in text_lower for k in kws)]
    return ", ".join(matched) if matched else "Uncategorized"  # joined to a string for clean Sheets storage

def classify_stance(text: str) -> str:
    text_lower = text.lower()
    support = sum(w in text_lower for w in SUPPORTIVE_WORDS)
    critical = sum(w in text_lower for w in CRITICAL_WORDS)
    if support > critical:
        return "Supportive"
    elif critical > support:
        return "Critical"
    return "Neutral"

def fetch_news(topics, days_back=2, page_size=20):
    """days_back kept short since this runs daily — we only need what's new since yesterday."""
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    all_articles = []

    for topic in topics:
        params = {
            "q": f'"{topic}"',
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": page_size,
            "apiKey": NEWS_API_KEY,
        }
        try:
            resp = requests.get(NEWS_API_URL, params=params, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"Request failed for '{topic}': {e}")
            continue

        if resp.status_code != 200:
            print(f"News API error for '{topic}' (status {resp.status_code}): {resp.text[:150]}")
            continue

        for article in resp.json().get("articles", []):
            title = article.get("title") or ""
            desc = article.get("description") or ""
            text = f"{title} {desc}"
            all_articles.append({
                "topic": topic,
                "title": title,
                "description": desc,
                "source": article.get("source", {}).get("name", "Unknown"),
                "url": article.get("url"),
                "published_at": article.get("publishedAt"),
                "themes": classify_themes(text),
                "stance": classify_stance(text),
                "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

    return pd.DataFrame(all_articles)

def append_new_articles_to_sheet(df: pd.DataFrame):
    if df.empty:
        print("No new articles fetched.")
        return

    existing = sheet.get_all_records()
    existing_urls = {row["url"] for row in existing} if existing else set()

    df = df[~df["url"].isin(existing_urls)]  # dedupe against what's already in the sheet
    if df.empty:
        print("No new unique articles to append.")
        return

    if not existing:
        sheet.append_row(df.columns.tolist())  # write header row once, if sheet is empty

    sheet.append_rows(df.values.tolist())
    print(f"Appended {len(df)} new articles.")

if __name__ == "__main__":
    df = fetch_news(default_topics)
    append_new_articles_to_sheet(df)
