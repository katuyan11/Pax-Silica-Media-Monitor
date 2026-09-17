print("FETCH SCRIPT STARTED")
# ---------------------------------------------------------
# GMA RSS
# ---------------------------------------------------------
def fetch_rss_news(days_back=20):
    """Fetch recent GMA News articles matching Pax Silica-related keywords.

    The GMA RSS feed is a general news feed, so keyword filtering is
    applied to identify potentially relevant articles before they are
    added to the dataset.
    """

    # -----------------------------------------------------
    # Keywords used to identify potentially relevant articles
    # -----------------------------------------------------
    rss_keywords = [
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
            # Combine title + description for keyword matching
            # -------------------------------------------------
            full_text = f"{title} {description}".lower()

            matched_keywords = [
                keyword
                for keyword in rss_keywords
                if keyword in full_text
            ]

            # -------------------------------------------------
            # Skip unrelated GMA articles
            # -------------------------------------------------
            if not matched_keywords:
                continue

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

            # -------------------------------------------------
            # Skip articles older than the lookback period
            # -------------------------------------------------
            if published_datetime and published_datetime < cutoff_date:
                continue

            # -------------------------------------------------
            # Classify themes and stance
            # -------------------------------------------------
            themes = classify_themes(full_text)
            stance = classify_stance(full_text)

            # -------------------------------------------------
            # Add article
            # -------------------------------------------------
            all_articles.append({
                "topic": ", ".join(matched_keywords),
                "title": title,
                "description": description,
                "source": "GMA News",
                "url": url,
                "published_at": published_at,
                "themes": themes,
                "stance": stance,
                "fetched_at": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            })

    except Exception as e:
        print(f"GMA RSS request failed: {e}")

    return pd.DataFrame(all_articles)
