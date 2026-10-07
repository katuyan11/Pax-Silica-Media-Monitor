import os
import re
import json
import time
import requests
import pandas as pd
import gspread
import feedparser
import urllib.parse

from urllib.parse import urlparse
from google.oauth2.service_account import Credentials
from datetime import datetime, timezone, timedelta

from googlenewsdecoder import gnewsdecoder


# ============================================================
# CONFIG
# ============================================================

WORLD_NEWS_URL = "https://api.worldnewsapi.com/search-news"

SHEET_TAB_NAME = "Clean_Data"

# No genuine Pax Silica coverage can predate this — anything older is
# necessarily about an unrelated historical BCDA/New Clark City story.
MIN_PUBLISH_DATE = datetime(2025, 12, 1, tzinfo=timezone.utc)

RSS_FEEDS = {
    "GMA News": "https://www.gmanetwork.com/news/rss/",
    "Inquirer Newsinfo": "https://newsinfo.inquirer.net/feed",
    "Inquirer Business": "https://business.inquirer.net/feed",
    "Inquirer Global Nation": "https://globalnation.inquirer.net/feed",
    "Inquirer Global Opinion": "https://opinion.inquirer.net/feed",
    "Manila Bulletin": "https://mb.com.ph/feed/",
    "Philstar": "https://www.philstar.com/rss/headlines",
    "Rappler": "https://www.rappler.com/feed/",
    "PNA": "https://www.pna.gov.ph/rss",
    "Abante": "https://www.abante.com.ph/feed/",
}

# ============================================================
# TIMESTAMP NORMALIZER
# ============================================================

SOURCE_TZ = "UTC"            # timezone of feed dates that carry no offset
TARGET_TZ = "Asia/Manila"    # timezone stored in the sheet
TS_FORMAT = "%Y-%m-%d %H:%M:%S"   # zero-padded, e.g. 2026-10-03 08:00:14


def normalize_timestamp(value) -> str:
    """Return 'YYYY-MM-DD HH:MM:SS' in TARGET_TZ, or '' if unparseable."""
    if value is None or value == "":
        return ""

    # feedparser's published_parsed / updated_parsed are always UTC
    if isinstance(value, time.struct_time):
        value = datetime(*value[:6])
        source_tz = "UTC"
    else:
        source_tz = SOURCE_TZ

    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""

    if ts.tzinfo is None:
        ts = ts.tz_localize(source_tz)

    return ts.tz_convert(TARGET_TZ).strftime(TS_FORMAT)


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
        "value chain", "industrial corridor", "real estate", "economy",
        "power demand", "business groups", "industrialization", "Makati Business Club",
        "Filipino industries", "industries", "industrialization", "industrial investments",
    ],
    "Environmental & Resource Impact": [
        "water", "water table", "water depletion", "water scarcity",
        "energy consumption", "energy demand", "power consumption",
        "electricity demand", "power grid", "groundwater", "desalination",
        "pollution", "emissions", "environmental compliance",
        "land conversion", "lng", "aeta", "ancestral domain",
        "displacement", "indigenous", "resettlement", "land rights",
        "heat",
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
        "high value work", "high-value work",
    ],
    "Supply-Chain Resilience": [
        "supply chain", "supply chain resilience", "supply chain security",
        "critical minerals", "mineral supply", "semiconductor supply chain",
        "regional partners", "regional integration", "diversification",
        "market diversification", "supplier diversification",
        "strategic dependencies", "coercive dependencies", "single market",
        "alternative markets", "trusted partners", "economic resilience",
        "chipmaker", "chipmakers", "mineral", "minerals", "PH minerals",
    ],
    "Institutional Governance": [
        "bcda", "dict", "dti", "marcos", "bingcang", "aguda",
        "bilateral agreement", "multilateral agreement", "declaration",
        "summit", "policy", "regulation", "oversight", "legal framework",
        "civil society", "kalikasan", "makabayan", "ibon", "protest",
        "opposition", "moratorium", "activist", "walkout", "dialogue",
        "multi-stakeholder dialogue", "criticize", "akbayan", "Government",
        "PBBM", "President Marcos", "President", "Imee Marcos", "Sara Duterte",
        "consultations", "consultation", "Senator", "arrested", "Rep.", "Gov't",
        "exec", "lobbies", "lobbied", "lobby", "public hearing",
    ],
    "Geopolitical Security": [
        "coercive dependencies", "civilian industrial zone",
        "supply chain security", "geopolitics", "national security",
        "strategic alignment", "strategic dependence",
        "strategic dependency", "security implications",
        "economic security", "China", "sovereignty", "US", "America",
    ],
}

SUPPORTIVE_WORDS = [
    "support", "supports", "supported", "back", "backs", "backed",
    "welcome", "welcomes", "welcomed", "approve", "approved", "benefit",
    "benefits", "opportunity", "opportunities", "growth", "investment",
    "job creation", "development", "expansion", "boost", "income",
    "jobs", "bets", "trusted", "trusted partnership", "trust", 
    "beef up PH resiliency, industrialization", "keen on",
    "new global position", "promise", "sees promise", "promising",
    "good deal", "good for", "defends", "supercharge",
]

CRITICAL_WORDS = [
    "oppose", "opposes", "opposed", "criticize", "criticizes",
    "criticized", "criticism", "concern", "concerns", "risk", "risks",
    "threat", "threatens", "environmental damage", "displacement",
    "pollution", "depletion", "moratorium", "protest", "protests",
    "protested", "reject", "rejected", "limbo", "fear", "misplaced",
    "against", "protests", "protested", "derail", "losing", "scrutiny",
    "scrutinizes", "displaced", "lost", "backlash", "clash", "walkout",
    "pushback", "worse", "opposing", "sell out", "slams", "massive sellout",
    "No to Pax Silica", "backlash", "exploitation", "scrutiny", "opposition",
    "scraps", "debunks claims", "laban", "kontra", "Scrap Pax Silica", "slammed",
    "slams", "force them off", "force off", "condemns", "condemn", "violent",
    "dispute", "protest", "protesters", "cancellation", "cancel", "flags risk",
    "costly", "plunder", "tough questions", "top concerns",
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
# GOOGLE NEWS RSS HELPERS
# ============================================================

def clean_html(raw_html: str) -> str:
    """Strip HTML tags from Google News RSS's summary field,
    which contains markup rather than real snippet text."""
    return re.sub(r"<[^>]+>", "", raw_html or "").replace("&nbsp;", " ").strip()


def resolve_google_news_url(google_url: str):
    """Decode Google News' redirect token to get the real publisher URL.
    Returns (resolved_url, success) instead of silently falling back —
    a fallback to the raw, unresolved google_url was causing duplicate
    appends, since an undecoded link doesn't match the real article's
    URL or outlet name from a prior successful run."""
    try:
        result = gnewsdecoder(google_url, interval=1)
        if result.get("success") and result.get("decoded_url"):
            return result["decoded_url"], True
        return google_url, False
    except Exception as e:
        print(f"DECODE EXCEPTION for {google_url[:80]}...: {type(e).__name__}: {e}")
        return google_url, False


def strip_source_from_title(title: str, source_name: str) -> str:
    """Google News RSS titles are formatted 'Article Title - Source Name'.
    Remove the trailing source name so it doesn't pollute word-frequency
    analysis (e.g. 'Philstar.com' or 'GMA News' showing up as a top term)."""
    if not title:
        return title
    if source_name:
        title = re.sub(
            rf"\s*[-–]\s*{re.escape(source_name)}\s*$",
            "",
            title,
            flags=re.IGNORECASE,
        )
    # Fallback: also strip a generic trailing " - Something" segment
    # in case the source name didn't match exactly
    title = re.sub(r"\s*[-–]\s*[A-Za-z0-9.\s]{2,30}$", "", title) if source_name else title
    return title.strip()


# ============================================================
# DEDUPLICATION HELPERS
# ============================================================

def make_dedup_key(title: str, source: str) -> str:
    """Fallback dedup signal alongside URL matching. Google News can
    reissue a different (or unresolved) URL token for the same real
    article depending on which search query surfaced it, so URL-only
    dedup can miss true duplicates. This normalizes title + source as
    a backstop check."""
    normalized_title = re.sub(r"\s+", " ", (title or "").strip().lower())
    normalized_source = (source or "").strip().lower()
    return f"{normalized_source}::{normalized_title}"


def normalize_title_only(title: str) -> str:
    """Fallback dedup signal independent of source name. Source labels can
    vary between fetches even for correctly-resolved articles (Google's
    entry.source.title vs. our domain-derived fallback name may phrase
    the same outlet differently), so this catches matches that the
    source-inclusive dedup_key would miss."""
    return re.sub(r"\s+", " ", (title or "").strip().lower())


# ============================================================
# CLASSIFIERS  (replaces the existing "CLASSIFIERS" section in
# fetch_and_append.py; keep THEME_KEYWORDS, SUPPORTIVE_WORDS and
# CRITICAL_WORDS exactly where they are, above this block)
# ============================================================

# Terms that must match with their exact capitalization. "US" (the country)
# would otherwise match the pronoun "us" in every other headline.
CASE_SENSITIVE_TERMS = {"US"}


def compile_term(term: str, allow_plural: bool = False):
    """Compile a keyword into a whole-word regex.

    (?<!\\w) and (?!\\w) require that the keyword is not embedded inside a
    longer word, so "dict" no longer matches "predict" and "dti" no longer
    matches inside other words. Using lookarounds instead of \\b also lets
    terms that end in punctuation, such as "Rep." and "Gov't", match.

    allow_plural lets the keyword also match a trailing "s" or "es"
    ("data center" -> "data centers"). It is used for themes only: theme
    matching is true/false per theme, so extra matches cannot inflate a score.
    Stance lists are scored by counting matches, and they already list their
    variants explicitly ("protest", "protests", "protested"), so they use
    exact whole-word matching to avoid counting one word twice.
    """
    suffix = r"(?:e?s)?" if allow_plural else ""
    flags = 0 if term in CASE_SENSITIVE_TERMS else re.IGNORECASE
    return re.compile(rf"(?<!\w){re.escape(term)}{suffix}(?!\w)", flags)


# Compiled once at import. dict.fromkeys() removes repeated entries (for
# example "protest", "slams" and "backlash" each appear more than once in
# CRITICAL_WORDS) so that a duplicated entry cannot count twice.
THEME_PATTERNS = {
    theme: [compile_term(k, allow_plural=True) for k in dict.fromkeys(keywords)]
    for theme, keywords in THEME_KEYWORDS.items()
}
SUPPORTIVE_PATTERNS = [compile_term(k) for k in dict.fromkeys(SUPPORTIVE_WORDS)]
CRITICAL_PATTERNS = [compile_term(k) for k in dict.fromkeys(CRITICAL_WORDS)]


def _normalize_text(text) -> str:
    """Collapse runs of whitespace so multi-word keywords match reliably.
    Case is handled by the patterns themselves, so the text is NOT lowercased."""
    return re.sub(r"\s+", " ", str(text or ""))


def classify_themes(text):
    text = _normalize_text(text)
    matched_themes = [
        theme for theme, patterns in THEME_PATTERNS.items()
        if any(p.search(text) for p in patterns)
    ]
    return ", ".join(matched_themes) if matched_themes else "Uncategorized"


def classify_stance(text):
    text = _normalize_text(text)
    supportive_hits = sum(1 for p in SUPPORTIVE_PATTERNS if p.search(text))
    critical_hits = sum(1 for p in CRITICAL_PATTERNS if p.search(text))
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
            "source-country": "ph",
            "number": 20,
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
                "source": get_outlet_name(url),
                "url": url,
                "published_at": normalize_timestamp(
                    article.get("publish_date") or article.get("published") or ""
                ),
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
                "published_at": normalize_timestamp(
                    entry.get("published_parsed")
                    or entry.get("updated_parsed")
                    or entry.get("published")
                    or entry.get("updated")
                    or ""
                ),
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
# GOOGLE NEWS RSS (free, no quota — Philippine edition search)
# ============================================================

_decode_cache = {}  # module-level cache, cleared each run since this is a script

def fetch_google_news_rss():
    """Search Google News RSS, filtered to Philippine edition, for each topic.
    Free and unlimited — no quota, unlike World News API.

    Google's keyword search is looser than World News API's, so generic
    BCDA/New Clark City stories unrelated to Pax Silica can otherwise slip
    through. To compensate, this source requires the article to literally
    name "Pax Silica" — the shared is_relevant() secondary-term rule is not
    applied here, since it's too permissive for this particular source.
    """

    articles = []
    skipped_decode_failures = 0

    for topic in DEFAULT_TOPICS:
        print(f"Fetching Google News RSS: {topic}")

        encoded_query = urllib.parse.quote(topic)
        feed_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-PH&gl=PH&ceid=PH:en"

        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            print(f"Google News RSS error for '{topic}': {e}")
            continue

        print(f"  {len(feed.entries)} entries returned")

        for entry in feed.entries:
            raw_title = (entry.get("title") or "").strip()
            raw_summary = entry.get("summary") or ""
            google_url = (entry.get("link") or "").strip()

            if not raw_title or not google_url:
                continue

            # --- Cheap relevance pre-check on raw text, BEFORE decoding ---
            # Decoding is the slow, network-bound step (one+ requests per
            # article, with a deliberate sleep between them to avoid being
            # rate-limited). Most entries from broad queries like "BCDA" or
            # "Aeta ancestral domain" won't mention Pax Silica at all, so
            # filtering here first avoids decoding articles we'd throw away
            # anyway.
            raw_combined = f"{raw_title} {clean_html(raw_summary)}".lower()
            if not any(term in raw_combined for term in ANCHOR_TERMS):
                continue
            if any(term in raw_combined for term in EXCLUDE_TERMS):
                continue

            # --- Date filter: skip anything published before Pax Silica existed ---
            published_parsed = entry.get("published_parsed")
            if published_parsed:
                published_dt = datetime(*published_parsed[:6], tzinfo=timezone.utc)
                if published_dt < MIN_PUBLISH_DATE:
                    continue

            # --- Decode, using an in-run cache ---
            # The same real article can surface under several different
            # DEFAULT_TOPICS queries in one run, each time with a fresh
            # redirect token, so cache on the raw token to avoid re-decoding
            # (and re-hitting the network) for stories we've already resolved
            # — or already failed to resolve, in the same run.
            if google_url in _decode_cache:
                resolved_url, decode_success = _decode_cache[google_url]
            else:
                resolved_url, decode_success = resolve_google_news_url(google_url)
                _decode_cache[google_url] = (resolved_url, decode_success)

            # --- Skip entries that failed to decode ---
            # Keeping a failed decode's raw, unresolved news.google.com URL
            # was causing duplicate appends: it doesn't match the real
            # article's URL or outlet name from a prior successful run, so
            # dedup treats it as a new article. Skipping means it'll simply
            # get picked up cleanly on a later run instead.
            if not decode_success:
                skipped_decode_failures += 1
                continue

            source_name = ""
            if entry.get("source"):
                source_name = entry.get("source", {}).get("title", "")
            if not source_name:
                source_name = get_outlet_name(resolved_url)

            title = strip_source_from_title(raw_title, source_name)
            description = clean_html(raw_summary)

            row = {
                "topic": topic,
                "title": title,
                "description": description,
                "source": source_name or "Google News",
                "url": resolved_url,
                "published_at": normalize_timestamp(
                    entry.get("published_parsed") or entry.get("published") or ""
                ),
            }

            full_text = f"{title} {description}"
            row["themes"] = classify_themes(full_text)
            row["stance"] = classify_stance(full_text)
            row["fetched_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            articles.append(row)

    print(f"Skipped due to decode failure: {skipped_decode_failures}")
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
    existing_dedup_keys = {
        make_dedup_key(record.get("title", ""), record.get("source", ""))
        for record in existing_records
    }
    existing_titles_only = {
        normalize_title_only(record.get("title", ""))
        for record in existing_records
        if str(record.get("title", "")).strip()
    }

    print(f"Existing articles in {SHEET_TAB_NAME}: {len(existing_urls)}")

    new_df = df.copy()
    new_df["dedup_key"] = new_df.apply(
        lambda r: make_dedup_key(r["title"], r["source"]), axis=1
    )
    new_df["title_only_key"] = new_df["title"].apply(normalize_title_only)

    new_df = new_df[
        ~new_df["url"].isin(existing_urls)
        & ~new_df["dedup_key"].isin(existing_dedup_keys)
        & ~new_df["title_only_key"].isin(existing_titles_only)
    ].copy()

    new_df = new_df.drop(columns=["dedup_key", "title_only_key"])

    if new_df.empty:
        print("No new articles to append.")
        return

    rows = new_df[EXPECTED_COLUMNS].fillna("").values.tolist()
    sheet.append_rows(rows, value_input_option="RAW")

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

    # Fail-safe: World News API is OFF unless explicitly enabled
    run_world_news = os.environ.get("RUN_WORLD_NEWS", "false").lower() == "true"

    sheet = get_google_sheet()
    ensure_headers(sheet)

    if run_world_news:
        print("Running World News API...")
        world_news_articles = fetch_world_news()
        print(f"World News relevant articles: {len(world_news_articles)}")
    else:
        world_news_articles = []
        print("Skipping World News API this run (RSS-only schedule).")

    rss_articles = fetch_rss_articles()
    print(f"RSS relevant articles: {len(rss_articles)}")

    google_news_articles = fetch_google_news_rss()
    print(f"Google News RSS relevant articles: {len(google_news_articles)}")

    all_articles = world_news_articles + rss_articles + google_news_articles
    print(f"Total relevant articles before deduplication: {len(all_articles)}")

    df = normalize_articles(all_articles)
    print(f"Unique relevant articles: {len(df)}")

    append_new_articles_to_sheet(df, sheet)

    print("=" * 60)
    print("FETCH COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
