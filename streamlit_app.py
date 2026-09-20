import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('punkt')
nltk.download('punkt_tab')  # newer NLTK versions require this separately
nltk.download('stopwords')

st.set_page_config(page_title="Pax Silica NLP News Monitor", layout="wide")
st.title("Pax Silica NLP News Monitor")
st.markdown("""
Natural Language Processing (NLP) monitoring of dominant themes and stances in Pax Silica coverage, 
an evolving Philippine news topic tracked daily since September 17, 2026 — built with Python and Streamlit.
""")

SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

@st.cache_data(ttl=3600)
def load_data():
    creds_dict = st.secrets["google_service_account"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(st.secrets["GOOGLE_SHEET_ID"]).worksheet("Clean_Data")
    records = sheet.get_all_records()
    df = pd.DataFrame(records)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    return df

# Phrases to treat as a single term instead of splitting into separate words
MULTI_WORD_TERMS = [
    "artificial intelligence",
    "new clark city",
    "clark freeport",
    "economic security zone",
    "ancestral domain",
    "data center",
]

def get_top_terms(texts: list[str], n: int = 15) -> list[tuple[str, int]]:
    """Tokenize, remove stopwords, and return the n most common terms.
    Known multi-word phrases are preserved as single entries."""

    stop_words = set(stopwords.words('english'))
    stop_words.update({'pax', 'silica'})

    all_words = []

    for text in texts:
        text_lower = text.lower()

        for phrase in MULTI_WORD_TERMS:
            joined = phrase.replace(" ", "_")
            text_lower = re.sub(r'\b' + re.escape(phrase) + r'\b', joined, text_lower)

        tokens = word_tokenize(text_lower)
        words = [w for w in tokens if (w.isalpha() or "_" in w) and w not in stop_words and len(w) > 2]
        all_words.extend(words)

    counted = Counter(all_words).most_common(n)
    return [(term.replace("_", " "), count) for term, count in counted]

df = load_data()

if df.empty:
    st.info("No data yet. Check back after the next daily fetch runs.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Most Frequent Categories")
        theme_series = df["themes"].str.split(", ").explode()
        theme_counts = theme_series.value_counts().reset_index()
        theme_counts.columns = ["theme", "count"]
        fig_themes = px.bar(theme_counts, x="theme", y="count")
        fig_themes.update_yaxes(dtick=1)
        st.plotly_chart(fig_themes, use_container_width=True)

    with col2:
        st.subheader("Article Stance")
        stance_counts = df["stance"].value_counts().reset_index()
        stance_counts.columns = ["stance", "count"]
        st.plotly_chart(px.pie(stance_counts, names="stance", values="count"), use_container_width=True)

    st.subheader("Theme × Stance Heatmap")

    theme_stance_df = df.copy()
    theme_stance_df["themes"] = theme_stance_df["themes"].str.split(", ")
    theme_stance_df = theme_stance_df.explode("themes")

    heatmap_data = (
        theme_stance_df.groupby(["themes", "stance"])
        .size()
        .reset_index(name="count")
        .pivot(index="themes", columns="stance", values="count")
        .fillna(0)
    )

    fig_heatmap = px.imshow(
        heatmap_data,
        labels=dict(x="Stance", y="Theme", color="Article Count"),
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig_heatmap.update_xaxes(side="top")
    fig_heatmap.update_layout(coloraxis_colorbar=dict(dtick=1))
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.caption(
        "Note: an article can span multiple themes, so it is counted once per theme here — "
        "cell totals will add up to more than the total number of articles."
    )

    st.subheader("Frequently Mentioned Words in Coverage")
    combined_texts = (df["title"] + " " + df["description"]).tolist()
    top_terms = get_top_terms(combined_texts)

    terms_df = pd.DataFrame(top_terms, columns=["term", "count"])
    fig_terms = px.bar(terms_df, x="term", y="count")
    fig_terms.update_yaxes(dtick=1)
    st.plotly_chart(fig_terms, use_container_width=True)

    st.subheader("Articles")
    st.dataframe(
        df[["published_at", "source", "title", "themes", "stance", "url"]].sort_values(
            "published_at", ascending=False
        ),
        use_container_width=True
    )
