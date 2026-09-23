import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px
import re
import nltk
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Pax Silica NLP News Monitor",
    layout="wide"
)

# =========================================================
# NLTK SETUP
# =========================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# =========================================================
# TITLE
# =========================================================

st.title("Pax Silica NLP News Monitor")

st.markdown(
    """
    <div style="
        text-align: justify;
        color: black;
        font-size: 16px;
        margin-bottom: 18px;
    ">
    This prototype monitors Philippine news coverage related to Pax Silica using automated
    news ingestion and rule-based Natural Language Processing (NLP), including keyword-based
    theme classification, stance detection, text preprocessing, and word-frequency analysis
    to identify dominant themes and stances. Coverage has been tracked daily since September 17, 2026,
    using Python and Streamlit.
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# REFRESH BUTTON
# =========================================================

if st.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# =========================================================
# GOOGLE SHEETS CONNECTION
# =========================================================

@st.cache_data(ttl=3600)
def load_data():

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    credentials = Credentials.from_service_account_info(
        st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"],
        scopes=scopes
    )

    gc = gspread.authorize(credentials)

    sheet = gc.open_by_key(
        st.secrets["GOOGLE_SHEET_ID"]
    ).worksheet("Clean_Data")

    records = sheet.get_all_records()

    return pd.DataFrame(records)


df = load_data()

# =========================================================
# CHECK DATA
# =========================================================

if df.empty:
    st.warning("No article data available yet.")
    st.stop()

# =========================================================
# CLEAN DATA
# =========================================================

df["published_at"] = pd.to_datetime(
    df["published_at"],
    errors="coerce"
)

df["themes"] = df["themes"].fillna("")
df["stance"] = df["stance"].fillna("Neutral")

# Normalize stance
df["stance"] = (
    df["stance"]
    .astype(str)
    .str.strip()
    .str.title()
)

valid_stances = [
    "Supportive",
    "Neutral",
    "Critical"
]

df.loc[
    ~df["stance"].isin(valid_stances),
    "stance"
] = "Neutral"

# =========================================================
# RESEARCH QUESTIONS
# =========================================================

st.header("RQ1: What themes are represented in Philippine media coverage of Pax Silica?")

st.markdown(
    """
    <div style="
        text-align: justify;
        color: black;
        font-size: 14px;
        margin-bottom: 20px;
    ">
    <strong>Note:</strong> The categories were defined based on recurring topics and issues
    identified in the corpus and subsequently operationalized through keyword-based classification.
    The thematic categories were developed inductively from patterns observed in the collected
    news coverage.
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# THEME DESCRIPTIONS
# =========================================================

THEME_ORDER = [
    "Economic Development",
    "Technological Advancement",
    "Human Capital & Employment",
    "Supply-Chain Resilience",
    "Environmental & Resource Impact",
    "Institutional Governance",
    "Geopolitical Security"
]

THEME_DESCRIPTIONS = {

    "Economic Development":
        "Pax Silica as an economic opportunity through investment, industrial growth, "
        "high-value manufacturing, the Luzon Economic Corridor, and concerns over "
        "economic dependency on foreign partners.",

    "Human Capital & Employment":
        "Job creation, workforce readiness, technical training, labor standards, and "
        "whether employment benefits local communities or depends on imported skilled labor.",

    "Environmental & Resource Impact":
        "Impacts on energy, water, land, mining, food security, displacement of indigenous "
        "peoples, ecological conditions, power and water demand, critical-mineral extraction, "
        "and contamination.",

    "Geopolitical Security":
        "Pax Silica as a strategic response to China, Philippine-US alliance dynamics, "
        "sovereignty and territorial concerns, and security vulnerabilities involving "
        "critical infrastructure and data centers.",

    "Technological Advancement":
        "Semiconductors, artificial intelligence, data centers, advanced manufacturing, "
        "technology transfer, and digital infrastructure.",

    "Supply-Chain Resilience":
        "Efforts to diversify and secure supply chains, reduce dependence on China for "
        "semiconductors, rare earths, and advanced manufacturing inputs, and strengthen "
        "the Philippines' role in regional supply chains.",

    "Institutional Governance":
        "Regulation, transparency, legislative scrutiny, community and Indigenous opposition, "
        "civil society, land and resource rights, policy critiques, and competing "
        "pro-development and critical interpretations."
}

left_themes = THEME_ORDER[:4]
right_themes = THEME_ORDER[4:]

# Increased spacing between the two columns
col1, spacer, col2 = st.columns([1, 0.20, 1])

with col1:

    for theme in left_themes:

        st.markdown(
            f"""
            <div style="
                text-align: justify;
                color: black;
                margin-bottom: 18px;
            ">
            <strong>{theme}</strong><br>
            {THEME_DESCRIPTIONS[theme]}
            </div>
            """,
            unsafe_allow_html=True
        )

with col2:

    for theme in right_themes:

        st.markdown(
            f"""
            <div style="
                text-align: justify;
                color: black;
                margin-bottom: 18px;
            ">
            <strong>{theme}</strong><br>
            {THEME_DESCRIPTIONS[theme]}
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================================================
# RQ2
# =========================================================

st.header(
    "RQ2: How do the themes and stances represented in media coverage "
    "change over time as new developments emerge?"
)

st.markdown(
    """
    <div style="
        text-align: justify;
        color: black;
        font-size: 14px;
        margin-bottom: 10px;
    ">
    Note: Each bubble shows how many articles tackled a given theme and stance on a specific
    date — bigger bubbles mean more articles, and the color shows whether the coverage leaned
    positive, negative, or neutral. The number next to each theme's name is its total article
    count across the whole monitoring period. Since one article can touch on multiple themes,
    these totals will add up to more than the overall article count.
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# PREPARE BUBBLE DATA
# =========================================================

bubble_df = df.copy()

bubble_df["date"] = bubble_df["published_at"].dt.date

bubble_df = bubble_df.dropna(
    subset=["date"]
)

bubble_df["theme_list"] = bubble_df["themes"].apply(
    lambda x: [
        t.strip()
        for t in str(x).split(",")
        if t.strip() in THEME_ORDER
    ]
)

bubble_df = bubble_df.explode(
    "theme_list"
)

bubble_df = bubble_df[
    bubble_df["theme_list"].notna()
]

theme_totals = (
    bubble_df
    .groupby("theme_list", observed=False)
    .size()
    .reindex(THEME_ORDER, fill_value=0)
)

theme_labels = {
    theme: f"{theme} ({theme_totals[theme]})"
    for theme in THEME_ORDER
}

bubble_df["theme_label"] = bubble_df["theme_list"].map(
    theme_labels
)

bubble_data = (
    bubble_df
    .groupby(
        ["date", "theme_label", "stance"],
        observed=False
    )
    .size()
    .reset_index(name="article_count")
)

# =========================================================
# BUBBLE CHART
# =========================================================

fig_bubble = px.scatter(
    bubble_data,
    x="date",
    y="theme_label",
    size="article_count",
    color="stance",
    hover_data={
        "date": True,
        "theme_label": True,
        "stance": True,
        "article_count": True
    },
    category_orders={
        "theme_label": [
            theme_labels[theme]
            for theme in THEME_ORDER
        ],
        "stance": [
            "Supportive",
            "Neutral",
            "Critical"
        ]
    },
    size_max=45
)

fig_bubble.update_layout(
    height=550,
    xaxis_title="Date",
    yaxis_title="Theme",
    legend_title="Stance",
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20
    )
)

fig_bubble.update_yaxes(
    categoryorder="array",
    categoryarray=[
        theme_labels[theme]
        for theme in THEME_ORDER
    ]
)

st.plotly_chart(
    fig_bubble,
    use_container_width=True
)

# =========================================================
# STANCE DISTRIBUTION
# =========================================================

st.header("Detected Article-Level Stance")

stance_counts = (
    df["stance"]
    .value_counts()
    .reindex(
        ["Supportive", "Neutral", "Critical"],
        fill_value=0
    )
    .reset_index()
)

stance_counts.columns = [
    "Stance",
    "Articles"
]

fig_stance = px.pie(
    stance_counts,
    names="Stance",
    values="Articles",
    hole=0.35
)

fig_stance.update_layout(
    height=450,
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20
    )
)

st.plotly_chart(
    fig_stance,
    use_container_width=True
)

st.markdown(
    """
    <div style="
        text-align: justify;
        color: black;
        font-size: 14px;
        margin-top: 5px;
        margin-bottom: 55px;
    ">
    The stance classification identifies whether individual articles are predominantly
    supportive, critical, or neutral toward Pax Silica-related developments based on
    predefined keyword patterns. This is a rule-based classification and should be
    interpreted as an indication of article-level framing rather than a measure of
    author or public opinion.
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# WORD FREQUENCY
# =========================================================

st.header("Frequently Mentioned Words in Coverage")

st.markdown(
    """
    <div style="
        text-align: justify;
        color: black;
        font-size: 14px;
        margin-bottom: 10px;
    ">
    The word cloud presents the most frequently mentioned words and terms across the entire
    news corpus collected by the monitor. The results are cumulative, covering the period
    from the start of monitoring on September 17, 2026, to the present.
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# TEXT PREPROCESSING
# =========================================================

text_columns = []

if "title" in df.columns:
    text_columns.extend(
        df["title"].fillna("").astype(str).tolist()
    )

if "description" in df.columns:
    text_columns.extend(
        df["description"].fillna("").astype(str).tolist()
    )

full_text = " ".join(text_columns)

# Lowercase
full_text = full_text.lower()

# Preserve selected multi-word terms
multi_word_terms = [
    "artificial intelligence",
    "new clark city",
    "clark freeport",
    "economic security zone",
    "ancestral domain",
    "data center"
]

placeholder_map = {}

for i, term in enumerate(multi_word_terms):

    placeholder = f"multiwordterm{i}"

    placeholder_map[placeholder] = term

    full_text = full_text.replace(
        term,
        placeholder
    )

# Tokenize
tokens = word_tokenize(
    full_text
)

# Stopwords
stop_words = set(
    stopwords.words("english")
)

custom_stopwords = {
    "pax",
    "silica",
    "said",
    "philippines",
    "philippine",
    "will",
    "also",
    "one",
    "new",
    "would",
    "could",
    "may",
    "mr",
    "ms",
    "according",
    "including"
}

stop_words.update(
    custom_stopwords
)

# Keep words only
filtered_tokens = []

for token in tokens:

    token = re.sub(
        r"[^a-z0-9]",
        "",
        token
    )

    if not token:
        continue

    if token in stop_words:
        continue

    filtered_tokens.append(
        token
    )

# =========================================================
# RESTORE MULTI-WORD TERMS
# =========================================================

restored_tokens = []

for token in filtered_tokens:

    if token in placeholder_map:

        restored_tokens.append(
            placeholder_map[token]
        )

    else:

        restored_tokens.append(
            token
        )

# =========================================================
# WORD FREQUENCY
# =========================================================

word_counts = Counter(
    restored_tokens
)

# =========================================================
# WORD CLOUD
# =========================================================

if word_counts:

    wordcloud = WordCloud(
        width=1400,
        height=700,
        background_color="white",
        max_words=100,
        collocations=False
    ).generate_from_frequencies(
        word_counts
    )

    fig_wc, ax = plt.subplots(
        figsize=(16, 8)
    )

    ax.imshow(
        wordcloud,
        interpolation="bilinear"
    )

    ax.axis("off")

    st.pyplot(
        fig_wc,
        use_container_width=True
    )

else:

    st.info(
        "Not enough text available to generate a word cloud."
    )

# =========================================================
# ARTICLE COLLECTION
# =========================================================

st.header("Article Collection")

st.markdown(
    """
    <div style="
        text-align: justify;
        color: black;
        font-size: 14px;
        margin-bottom: 15px;
    ">
    New articles are automatically collected through RSS feeds six times daily while
    World News API is queried at 7:00 AM and 7:00 PM.
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# ARTICLE TABLE
# =========================================================

display_columns = [
    "published_at",
    "title",
    "source",
    "themes",
    "stance",
    "url"
]

available_columns = [
    col
    for col in display_columns
    if col in df.columns
]

article_display = df[
    available_columns
].copy()

if "published_at" in article_display.columns:

    article_display["published_at"] = (
        article_display["published_at"]
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

st.dataframe(
    article_display,
    use_container_width=True,
    hide_index=True
)
