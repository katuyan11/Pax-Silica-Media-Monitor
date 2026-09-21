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
    "Philippine News Agency": "https://www.pna.gov.ph/latest.rss",
    "Abante": "https://www.abante.com.ph/feed",
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
default_topics = [

    # Core identity
    "Pax Silica",
    "Pax Silica Philippines",
    "Pax Silica Summit",
