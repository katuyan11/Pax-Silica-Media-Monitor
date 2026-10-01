import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px
import plotly.graph_objects as go

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

from wordcloud import WordCloud
import matplotlib.pyplot as plt


# ============================================================
# NLTK SETUP
# ============================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

# [BUG FIX] st.set_page_config must be the FIRST Streamlit command.
# It was previously called after the CSS and banner st.markdown calls.
st.set_page_config(
    page_title="Pax Silica News Monitor",
    layout="wide"
)

# ------------------------------------------------------------
# [DESIGN FIX] Shared CSS for the header + metric cards.
# Keeps the metric row visually grouped (bordered cards instead
# of three bare numbers floating on white) and gives the intro
# copy a touch of breathing room instead of running paragraph
# straight into paragraph.
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ------------------------------------------------------
       [FONT SIZE CONTROL] Bump this to scale most body text
       in the app (paragraphs, markdown, dataframe cell text).
       Scoped narrowly on purpose — NOT a wildcard like
       [class*="st-"] and NOT .stCaption — because those had
       been catching Streamlit's own heading components (the
       banner h1, "Articles Collected by the Monitor" subheader)
       and the footer caption, shrinking/growing things that
       were never meant to move.
    ------------------------------------------------------ */
    html, body, .stMarkdown p, .stDataFrame, .stSelectbox label {
        font-size: 18px;
    }
    /* Most body/explanatory paragraphs below are hardcoded inline
       (font-size: 14px / 15px) rather than using a class, so the
       rule above can't reach them — inline styles win by default.
       This overrides those inline sizes app-wide. Delete this block
       if you'd rather size each paragraph individually. */
    .stMarkdown div[style*="font-size: 14px"] {
        font-size: 18px !important;
    }
    .stMarkdown div[style*="font-size: 15px"] {
        font-size: 19px !important;
    }
    .metric-card {
        background: #F7F9FB;
        border: 1px solid #E3E8EC;
        border-radius: 9px;
        padding: 0.8rem 1.0rem;
        height: 35%;
    }
    .metric-card .metric-label {
        font-size: 0.78rem;
        font-weight: 550;
        letter-spacing: 0.2px;
        text-transform: uppercase;
        color: #6B7785;
        margin: 0 0 0.25rem 0;
    }
    .metric-card .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #17203A;
        margin: 0;
        line-height: 1.1;
    }
    .metric-card .metric-note {
        font-size: 0.75rem;
        color: #6B7785;
        margin: 0.2rem 0 0 0;
    }
    .intro-copy p {
        text-align: left;
        color: #2A2F36;
        font-size: 19px;
        line-height: 1.6;
        margin-bottom: 0.9rem;
    }

    /* [DESIGN FIX] Refresh button: palette-matched, no default look */
    .stButton > button {
        background: #FFFFFF;
        color: #21295C;
        border: 1px solid #C9D2DC;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #F1F5F9;
        border-color: #244B6E;
        color: #244B6E;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# [DESIGN FIX] Softer, near-flat gradient; divider line removed;
# eyebrow letter-spacing reduced; invalid `propercase` value removed.
st.markdown("""
<div style="
    background: linear-gradient(90deg, #1E2A50 0%, #244B6E 100%);
    padding: 1.6rem 2.2rem;
    border-radius: 0;
    margin-bottom: 1.3rem;
">
  <p style="color:#E8A33D; font-weight:750; letter-spacing:0.8px; font-size:0.95rem; margin:0 0 0.5rem 0; line-height:1.3;">
    An Eskwelabs Data Analytics Capstone Project
  </p>
  <h1 style="color:white; font-size:1.75rem; font-weight:800; margin:0 0 0.8rem 0; padding:0; line-height:1.25;">
    Monitoring the Conversation: A Media Monitoring Prototype to Track Coverage of the Pax Silica Initiative in the Philippines
  </h1>
  <p style="color:#F2D9A8; font-size:1.05rem; font-weight:600; margin:0; line-height:1.3;">
    Using rule-based keyword matching (NLP) to track themes and stances
  </p>
</div>
""", unsafe_allow_html=True)

# [DESIGN FIX] Intro trimmed to two paragraphs; methodology moved
# into a collapsed expander.
st.markdown(
    """
    <div class="intro-copy" style="margin-bottom: 8px;">
    <p>Pax Silica covers technology, AI infrastructure, and critical-mineral supply chains, and the Philippines is positioned to play a role in this global alliance. Media coverage is growing, touching on investment and jobs, economic growth, environmental and indigenous concerns, governance, sovereignty, and the country’s place in global supply chains.</p>
    <p>Following that conversation manually is difficult: new developments unfold, new voices enter, and different outlets focus on different aspects. The corpus includes news outlets, explainer sites, and organizational statements that circulate as part of the public conversation. This dashboard looks at what is being discussed, how the conversation shifts over time, and how local and international media outlets differ in their coverage.</p>
    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("How this prototype works"):
    st.write(
        "This prototype combines automated article collection with rule-based NLP. "
        "News articles, columns, editorials, press releases, and explainer articles are processed and classified by theme and stance using text preprocessing and rule-based "
        "keyword matching. These approaches provide a structured "
        "way to explore the themes and stances appearing across the monitored coverage."
    )

# ============================================================
# STANCE DISPLAY COLORS (kept consistent across all stance charts)
# ============================================================

STANCE_COLORS = {
    "Supportive": "#1f77b4",
    "Neutral": "#a9c6e8",
    "Critical": "#d62728"
}

STANCE_ORDER = [
    "Supportive",
    "Neutral",
    "Critical"
]


# ============================================================
# GOOGLE SHEETS CONNECTION
# ============================================================

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]


@st.cache_data(ttl=3600)
def load_data():

    creds_dict = st.secrets["google_service_account"]

    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=SCOPES
    )

    gc = gspread.authorize(creds)

    sheet = (
        gc
        .open_by_key(st.secrets["GOOGLE_SHEET_ID"])
        .worksheet("Clean_Data")
    )

    records = sheet.get_all_records()

    df = pd.DataFrame(records)

    if not df.empty and "published_at" in df.columns:

        df["published_at"] = pd.to_datetime(
            df["published_at"],
            errors="coerce"
        )

    return df


# ============================================================
# THEME ORDER (fixed reference order — used for stance heatmap
# and anywhere themes need a stable, non-data-dependent order)
# ============================================================

THEME_ORDER = [
    "Economic Development",
    "Technological Advancement",
    "Human Capital & Employment",
    "Supply-Chain Resilience",
    "Environmental & Resource Impact",
    "Institutional Governance",
    "Geopolitical Security"
]


# ============================================================
# THEME DESCRIPTIONS
# ============================================================

THEME_DESCRIPTIONS = {

    "Economic Development":
        "Pax Silica as an economic opportunity through investment, "
        "industrial growth, high-value manufacturing, the Luzon Economic "
        "Corridor, and concerns over potential economic dependency on "
        "foreign partners.",

    "Human Capital & Employment":
        "Job creation, workforce readiness, technical training, labor "
        "standards, and whether Pax Silica-related employment will benefit "
        "local communities or depend on imported skilled labor.",

    "Environmental & Resource Impact":
        "Pax Silica’s impacts on energy, water, land, mining, food security, "
        "displacement of indigenous peoples, and ecological conditions, "
        "including concerns over power and water demand, critical-mineral "
        "extraction, and environmental contamination.",

    "Geopolitical Security":
        "Coverage of Pax Silica as a strategic response to China, including "
        "Philippine-US alliance dynamics, sovereignty and territorial "
        "concerns, and the security vulnerabilities of critical "
        "infrastructure and data centers.",

    "Technological Advancement":
        "Pax Silica’s development of semiconductors, AI, data centers, "
        "advanced manufacturing, and technology transfer from international "
        "partners.",

    "Supply-Chain Resilience":
        "Efforts to diversify and secure critical supply chains by reducing "
        "dependence on China for semiconductors, rare earths, and advanced "
        "manufacturing inputs while strengthening the Philippines’ role "
        "in the coalition.",

    "Institutional Governance":
        "Coverage of government regulation, transparency, legislative "
        "scrutiny, community and Indigenous opposition, civil society "
        "mobilization, land and resource rights, policy critiques, and "
        "competing pro- and anti-Pax Silica interpretations."
}


# ============================================================
# SIGNIFICANT EVENTS / DEVELOPMENTS
# ============================================================

SIGNIFICANT_EVENTS = [

    {
        "date": "2026-04-17",
        "label": "PH joins Pax Silica",
        "description": (
            "The Philippines formally joined the US-led Pax Silica initiative "
            "and plans for a 4,000-acre industrial hub in the Luzon Economic "
            "Corridor were announced."
        )
    },

    {
        "date": "2026-05-04",
        "label": "PH-Israel cooperation",
        "description": (
            "Philippines-Israel discussions on critical minerals processing "
            "and AI technology were linked to Pax Silica."
        )
    },

    {
        "date": "2026-05-04",
        "label": "PH-UAE AI infrastructure talks",
        "description": (
            "Philippine officials discussed energy and digital infrastructure "
            "partnerships with UAE companies in connection with Pax Silica."
        )
    },

    {
        "date": "2026-07-27",
        "label": "Pax Silica highlighted in SONA",
        "description": (
            "The proposed Pax Silica Industrial Hub was highlighted during "
            "the President's 2026 State of the Nation Address as part of "
            "the Luzon Economic Corridor and the country's advanced "
            "manufacturing strategy."
        )
    },

    {
        "date": "2026-08-07",
        "label": "Development timeline announced",
        "description": (
            "BCDA and DTI officials provided further details on the proposed "
            "Pax Silica hub, including its development timeline and planned "
            "initial site development."
        )
    },

    {
        "date": "2026-09-10",
        "label": "Luzon Economic Corridor forum",
        "description": (
            "The Luzon Economic Corridor Investment Forum generated further "
            "discussion of the proposed Pax Silica development and its "
            "investment implications."
        )
    }
]


# ============================================================
# EVENT CLASSIFICATION
# ============================================================

INTERNATIONAL_EVENT_LABELS = {
    "PH-Israel cooperation",
    "PH-UAE AI infrastructure talks"
}


def prepare_significant_events(events):

    """Convert configured event dates into a clean DataFrame."""

    empty_columns = [
        "date",
        "label",
        "description",
        "event_type"
    ]

    if not events:
        return pd.DataFrame(columns=empty_columns)

    events_df = pd.DataFrame(events)

    if (
        "date" not in events_df.columns
        or "label" not in events_df.columns
    ):
        return pd.DataFrame(columns=empty_columns)

    if "description" not in events_df.columns:
        events_df["description"] = ""

    events_df["date"] = pd.to_datetime(
        events_df["date"],
        errors="coerce"
    )

    events_df = events_df[
        events_df["date"].notna()
    ].copy()

    events_df["event_type"] = events_df["label"].apply(
        lambda label:
            "International"
            if label in INTERNATIONAL_EVENT_LABELS
            else "Domestic"
    )

    return (
        events_df[
            [
                "date",
                "label",
                "description",
                "event_type"
            ]
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )


# ============================================================
# MULTI-WORD TERMS
# ============================================================

MULTI_WORD_TERMS = [
    "artificial intelligence",
    "new clark city",
    "clark freeport",
    "economic security zone",
    "ancestral domain",
    "data center",
]


# ============================================================
# FREQUENT TERM EXTRACTION
# ============================================================

def get_top_terms(
    texts,
    n=100
):

    """
    Tokenize text, remove stopwords, and return frequent terms.
    Known multi-word phrases are preserved as single entries.
    """

    stop_words = set(
        stopwords.words("english")
    )

    stop_words.update({
        "pax",
        "silica",
        "said",
        "philippines"
    })

    all_words = []

    for text in texts:

        if pd.isna(text):
            continue

        text_lower = str(text).lower()

        for phrase in MULTI_WORD_TERMS:

            joined = phrase.replace(
                " ",
                "_"
            )

            text_lower = re.sub(
                r"\b" + re.escape(phrase) + r"\b",
                joined,
                text_lower
            )

        tokens = word_tokenize(
            text_lower
        )

        words = [
            word
            for word in tokens
            if (
                word.isalpha()
                or "_" in word
            )
            and word not in stop_words
            and len(word) > 2
        ]

        all_words.extend(words)

    counted = Counter(
        all_words
    ).most_common(n)

    return [
        (
            term.replace("_", " "),
            count
        )
        for term, count in counted
    ]


# ============================================================
# LOAD DATA
# ============================================================

# [DESIGN FIX] Refresh button: no emoji, palette-matched (see CSS),
# with a tooltip that explains what it does.
refresh_col_label, refresh_col_button = st.columns([5, 1])

with refresh_col_label:
    st.markdown(
        "<div style='color:#6B7785; font-size:0.85rem; margin-top:0.4rem;'>"
        "Data updates automatically from the daily fetch."
        "</div>",
        unsafe_allow_html=True
    )

with refresh_col_button:
    if st.button(
        "Refresh data",
        use_container_width=True,
        help="Reload the latest articles from the daily fetch."
    ):

        st.cache_data.clear()
        st.rerun()


df = load_data()


# ============================================================
# MAIN APP
# ============================================================

if df.empty:

    st.info(
        "No data yet. Check back after the next daily fetch runs."
    )

else:

    # ========================================================
    # BASIC COLUMN SAFETY
    # ========================================================

    required_columns = [
        "published_at",
        "stance",
        "themes"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        st.error(
            "The Google Sheet is missing required columns: "
            + ", ".join(missing_columns)
        )

        st.stop()


    # ========================================================
    # [FIX 5] CORPUS OVERVIEW — surfaced immediately, so a
    # reader can gauge the size/credibility of the dataset
    # before reading any chart.
    # ========================================================

    if "url" in df.columns:

        overview_article_count = df["url"].nunique()

    else:

        overview_article_count = len(df)

    valid_dates = pd.to_datetime(
        df["published_at"],
        errors="coerce"
    ).dropna()

    if not valid_dates.empty:

        date_range_text = (
            f"{valid_dates.min().strftime('%b %Y')} – "
            f"{valid_dates.max().strftime('%b %Y')}"
        )

    else:

        date_range_text = "N/A"

    if not valid_dates.empty:

        latest_article_text = valid_dates.max().strftime("%b %d, %Y")

    else:

        latest_article_text = "N/A"

    overview_cols = st.columns(3)

    # ------------------------------------------------------------
    # [DESIGN FIX] Metrics rendered as bordered "cards" (via the
    # .metric-card CSS above) instead of bare st.metric() text, so
    # the three stats read as one grouped KPI strip.
    # ------------------------------------------------------------

    with overview_cols[0]:

        st.markdown(
            f"""
            <div class="metric-card">
                <p class="metric-label">Articles tracked</p>
                <p class="metric-value">{overview_article_count:,}</p>
                <p class="metric-note">Unique articles</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with overview_cols[1]:

        st.markdown(
            f"""
            <div class="metric-card">
                <p class="metric-label">Coverage period</p>
                <p class="metric-value">{date_range_text}</p>
                <p class="metric-note">Latest article: {latest_article_text}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with overview_cols[2]:

        # [FIX] Now counts unique values from "source" — the same
        # column the Articles Collected table displays — rather than
        # "outlet", which isn't a column in this sheet and was
        # silently falling back to "N/A".
        if "source" in df.columns:

            outlets_value = f"{df['source'].nunique():,}"

        else:

            outlets_value = "N/A"

        st.markdown(
            f"""
            <div class="metric-card">
                <p class="metric-label">Outlets monitored</p>
                <p class="metric-value">{outlets_value}</p>
                <p class="metric-note">Unique sources</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)

    st.markdown(
        "<hr style='border: none; border-top: 1px solid #ddd; margin: 8px 0 20px 0;'>",
        unsafe_allow_html=True
    )


    # ========================================================
    # [FIX 4] BUILD THEME COUNTS EARLY — moved up from later in
    # the file so Section 1's theme grid can be ordered by
    # actual coverage volume instead of an arbitrary fixed list.
    # ========================================================

    themes_df = df.copy()

    themes_df["themes"] = (
        themes_df["themes"]
        .fillna("")
        .astype(str)
        .str.split(", ")
    )

    themes_df = themes_df.explode(
        "themes"
    )

    themes_df = themes_df[
        themes_df["themes"].isin(THEME_ORDER)
    ].copy()

    theme_counts = (
        themes_df["themes"]
        .value_counts()
        .reindex(
            THEME_ORDER,
            fill_value=0
        )
    )

    # Display order: highest-volume theme first, ties broken by
    # THEME_ORDER's original sequence for stability.
    THEME_DISPLAY_ORDER = (
        theme_counts
        .sort_values(ascending=False)
        .index
        .tolist()
    )



    # ========================================================
    # RESEARCH QUESTION 1
    # ========================================================

    st.markdown(
        """
        <div style="
            font-size: 28px;
            font-weight: 800;
            margin-top: 10px;
            margin-bottom: 4px;
        ">
        1. What is the media talking about?
        </div>

        <div style="
            font-size: 15px;
            font-style: italic;
            font-weight: 400;
            color: black;
            margin-bottom: 10px;
        ">
        A breakdown of the themes — from jobs to geopolitics —
        that dominate Pax Silica coverage, and where the gaps 
        are for deeper investigation.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            color: black;
            text-align: left;
            font-size: 14px;
            margin-bottom: 20px;
        ">
        <strong>Note:</strong> The categories were defined based on recurring
        topics and issues identified in the corpus and subsequently
        operationalized through keyword-based classification. The thematic
        categories were developed inductively from patterns observed in the
        collected news coverage. Themes below are ordered by article volume,
        most-covered first.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # THEME DESCRIPTIONS — SINGLE COLUMN, ORDERED BY VOLUME
    # AND LABELED WITH ARTICLE COUNTS
    # ========================================================

    for theme in THEME_DISPLAY_ORDER:

        st.markdown(
            f"**{theme}** &nbsp;·&nbsp; "
            f"<span style='color:#555; font-size:13px;'>"
            f"{theme_counts[theme]:,} article"
            f"{'s' if theme_counts[theme] != 1 else ''}</span>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="
                color: black;
                text-align: left;
                margin-bottom: 18px;
            ">
                {THEME_DESCRIPTIONS[theme]}
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        "<hr style='border: none; border-top: 1px solid #ddd; margin: 8px 0 20px 0;'>",
        unsafe_allow_html=True
    )

    # ========================================================
    # RESEARCH QUESTION 2
    # ========================================================

    st.markdown(
        """
        <div style="
            font-size: 28px;
            font-weight: 800;
            margin-top: 20px;
            margin-bottom: 4px;
        ">
        2. Is the coverage positive, critical, or neutral —
        and on what topics?
        </div>

        <div style="
            font-size: 15px;
            font-style: italic;
            font-weight: 400;
            color: black;
            margin-bottom: 15px;
        ">
        Where media attention concentrates, and whether the stance
        leans supportive, critical, or balanced — a guide to which 
        themes are worth scrutinizing more closely.
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # DETECTED ARTICLE-LEVEL STANCE + HEAT MAP
    # ========================================================

    col1, spacer, col2 = st.columns(
        [1, 0.15, 1]
    )


    # ========================================================
    # DETECTED ARTICLE-LEVEL STANCE
    # ========================================================

    with col1:

        # ----------------------------------------------------
        # CALCULATE STANCE DISTRIBUTION
        # ----------------------------------------------------

        stance_counts = (
            df["stance"]
            .fillna("Neutral")
            .astype(str)
            .str.strip()
            .str.title()
            .replace({
                "Positive": "Supportive",
                "Negative": "Critical"
            })
            .value_counts()
            .reindex(
                STANCE_ORDER,
                fill_value=0
            )
            .reset_index()
        )

        stance_counts.columns = [
            "stance",
            "count"
        ]

        stance_total = stance_counts["count"].sum()

        stance_counts["percent"] = (
            stance_counts["count"] / stance_total * 100
            if stance_total > 0
            else 0
        )

        # ----------------------------------------------------
        # IDENTIFY HIGHEST + SECOND-HIGHEST STANCE
        # ----------------------------------------------------
        
        if stance_total > 0:
        
            ranked_stances = (
                stance_counts
                .sort_values(
                    by="count",
                    ascending=False
                )
                .reset_index(drop=True)
            )
        
            highest_stance = ranked_stances.loc[0, "stance"]
            highest_stance_share = ranked_stances.loc[0, "percent"]
        
            if len(ranked_stances) > 1:
        
                second_highest_stance = (
                    ranked_stances.loc[1, "stance"]
                )
        
                second_highest_share = (
                    ranked_stances.loc[1, "percent"]
                )
        
                stance_takeaway = (
                    f"{highest_stance} coverage accounts for "
                    f"{round(highest_stance_share):.0f}% of articles, "
                    f"followed by {second_highest_stance} coverage "
                    f"at {round(second_highest_share):.0f}%."
                )
        
            else:
        
                stance_takeaway = (
                    f"{highest_stance} coverage accounts for "
                    f"{round(highest_stance_share):.0f}% of articles."
                )
        
        else:
        
            stance_takeaway = (
                "No article-level stance could be detected"
            )
        
        # ----------------------------------------------------
        # TAKEAWAY TITLE
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div style="
                font-size: 22px;
                font-weight: 700;
                color: black;
                text-align: center;
                margin-bottom: 10px;
            ">
                {stance_takeaway}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # STANCE CHART LABELS
        # ----------------------------------------------------

        stance_counts["segment_label"] = stance_counts.apply(
            lambda row: (
                f"{row['stance']}<br>{int(row['count'])} "
                f"({row['percent']:.1f}%)"
            ),
            axis=1
        )

        stance_counts["row"] = "All coverage"


        # ----------------------------------------------------
        # CREATE STACKED BAR
        # ----------------------------------------------------

        fig_stance = px.bar(
            stance_counts,
            x="count",
            y="row",
            color="stance",
            orientation="h",
            text="segment_label",
            category_orders={
                "stance": STANCE_ORDER
            },
            color_discrete_map=STANCE_COLORS
        )

        fig_stance.update_traces(
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(
                size=12,
                color="white"
            ),
            marker_line_width=0
        )

        fig_stance.update_layout(
            barmode="stack",
            height=220,
            showlegend=True,
            legend_title_text="Stance",
            xaxis_title="Number of articles",
            yaxis_title="",
            yaxis=dict(
                showticklabels=False
            ),
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=40
            )
        )

        st.plotly_chart(
            fig_stance,
            use_container_width=True,
            config={
                "responsive": True
            }
        )


        # ----------------------------------------------------
        # CHART TITLE — BOTTOM, CENTERED, ITALIC
        # ----------------------------------------------------

        st.markdown(
            """
            <div style="
                font-size: 14px;
                font-weight: 400;
                font-style: italic;
                color: black;
                text-align: center;
                margin-top: -8px;
                margin-bottom: 10px;
            ">
                Detected Article-Level Stance
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # EXPLANATORY TEXT
        # ----------------------------------------------------

        st.markdown(
            """
            <div style="
                text-align: left;
                color: black;
                font-size: 14px;
                margin-bottom: 10px;
            ">
            Stance is estimated using predefined words and phrases associated
            with supportive or critical language in the available article text.
            The classifier counts these indicators and assigns the stance based
            on the stronger signal. Articles without a clear predominance of
            either signal are classified as Neutral. This is a rule-based
            classification and should be interpreted as a detected linguistic
            signal rather than a definitive statement of the article's or
            author's position.
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # STANCE DEFINITIONS — BELOW THE STANCE CHART
        # ----------------------------------------------------

        st.markdown(
            """
            <div style="
                font-size: 13px;
                color: black;
                margin-top: 10px;
                margin-bottom: 10px;
                text-align: left;
            ">
            <b>Definitions</b><br><br>
            <b>Supportive:</b> Article text contains more supportive language than critical language, such as references to benefits, opportunities, growth, investment, jobs, or backing and welcoming of Pax Silica-related developments.<br><br>
            <b>Neutral:</b> Article text shows no clear predominance of either signal. This includes straightforward reporting, explainers, and articles where supportive and critical language appear in roughly equal measure or not at all.<br><br>
            <b>Critical:</b> Article text contains more critical language than supportive language, such as references to concerns, risks, opposition, protests, displacement, scrutiny, or backlash toward Pax Silica-related developments.
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # SPACER
    # ========================================================

    with spacer:

        st.markdown(
            "<div style='height: 1px;'></div>",
            unsafe_allow_html=True
        )


    # ========================================================
    # THEME × STANCE HEAT MAP
    # ========================================================

    with col2:

        heatmap_df = df.copy()

        # ----------------------------------------------------
        # NORMALIZE STANCE
        # ----------------------------------------------------

        heatmap_df["stance"] = (
            heatmap_df["stance"]
            .fillna("Neutral")
            .astype(str)
            .str.strip()
            .str.title()
            .replace({
                "Positive": "Supportive",
                "Negative": "Critical"
            })
        )

        heatmap_df = heatmap_df[
            heatmap_df["stance"].isin(
                STANCE_ORDER
            )
        ].copy()


        # ----------------------------------------------------
        # EXPLODE THEMES
        # ----------------------------------------------------

        heatmap_df["themes"] = (
            heatmap_df["themes"]
            .fillna("")
            .astype(str)
            .str.split(", ")
        )

        heatmap_df = heatmap_df.explode(
            "themes"
        )

        heatmap_df = heatmap_df[
            heatmap_df["themes"].isin(
                THEME_ORDER
            )
        ].copy()

        heatmap_df = heatmap_df.reset_index(
            drop=True
        )


        # ----------------------------------------------------
        # CALCULATE COUNTS
        # ----------------------------------------------------

        heatmap_counts = (
            heatmap_df
            .groupby(
                [
                    "themes",
                    "stance"
                ],
                observed=False
            )
            .size()
            .unstack(
                fill_value=0
            )
        )

        heatmap_counts = heatmap_counts.reindex(
            index=THEME_ORDER,
            columns=STANCE_ORDER,
            fill_value=0
        )

        heatmap_counts["Total"] = (
            heatmap_counts[STANCE_ORDER]
            .sum(axis=1)
        )


        # ----------------------------------------------------
        # ROW-NORMALIZED PERCENTAGES
        # ----------------------------------------------------

        row_totals = (
            heatmap_counts["Total"]
            .replace(0, np.nan)
        )

        heatmap_pct = (
            heatmap_counts[STANCE_ORDER]
            .div(
                row_totals,
                axis=0
            )
            .fillna(0)
            * 100
        )

        # ----------------------------------------------------
        # IDENTIFY HIGHEST CRITICAL AND SUPPORTIVE SHARES
        # ----------------------------------------------------
        
        critical_shares = heatmap_pct["Critical"]
        
        valid_critical_shares = (
            critical_shares[
                heatmap_counts["Total"] > 0
            ]
        )
        
        supportive_shares = heatmap_pct["Supportive"]
        
        valid_supportive_shares = (
            supportive_shares[
                heatmap_counts["Total"] > 0
            ]
        )
        
        if (
            not valid_critical_shares.empty
            and not valid_supportive_shares.empty
        ):
        
            highest_critical_theme = (
                valid_critical_shares.idxmax()
            )
        
            highest_critical_share = (
                valid_critical_shares.max()
            )
        
            highest_supportive_theme = (
                valid_supportive_shares.idxmax()
            )
        
            highest_supportive_share = (
                valid_supportive_shares.max()
            )
        
            heatmap_takeaway = (
                f"{highest_critical_theme} has the highest share "
                f"of Critical coverage at "
                f"{round(highest_critical_share):.0f}%, while "
                f"{highest_supportive_theme} has the highest share "
                f"of Supportive coverage at "
                f"{round(highest_supportive_share):.0f}%."
            )
                
        elif not valid_critical_shares.empty:
        
            highest_critical_theme = (
                valid_critical_shares.idxmax()
            )
        
            highest_critical_share = (
                valid_critical_shares.max()
            )
        
            heatmap_takeaway = (
                f"{highest_critical_theme} has the highest share "
                f"of Critical coverage at "
                f"{round(highest_critical_share):.0f}%."
            )
        
        elif not valid_supportive_shares.empty:
        
            highest_supportive_theme = (
                valid_supportive_shares.idxmax()
            )
        
            highest_supportive_share = (
                valid_supportive_shares.max()
            )
        
            heatmap_takeaway = (
                f"{highest_supportive_theme} has the highest share "
                f"of Supportive coverage at "
                f"{round(highest_supportive_share):.0f}%."
            )
        
        else:
        
            heatmap_takeaway = (
                "No theme-level stance pattern could be detected"
            )


        # ----------------------------------------------------
        # TAKEAWAY TITLE — TOP OF HEAT MAP
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div style="
                font-size: 22px;
                font-weight: 700;
                color: black;
                text-align: center;
                margin-bottom: 8px;
            ">
                {heatmap_takeaway}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # ORDER THEMES BY TOTAL VOLUME
        # ----------------------------------------------------

        heatmap_theme_order = (
            heatmap_counts["Total"]
            .sort_values(
                ascending=False
            )
            .index
            .tolist()
        )

        heatmap_counts = (
            heatmap_counts
            .reindex(
                heatmap_theme_order
            )
        )

        heatmap_pct = (
            heatmap_pct
            .reindex(
                heatmap_theme_order
            )
        )


        # ----------------------------------------------------
        # BUILD DISPLAY MATRICES
        # ----------------------------------------------------

        display_columns = (
            STANCE_ORDER
            + ["Total"]
        )

        z_matrix = heatmap_pct.copy()

        z_matrix["Total"] = np.nan

        text_matrix = (
            heatmap_counts[
                display_columns
            ]
        )


        # ----------------------------------------------------
        # CREATE HEAT MAP
        # ----------------------------------------------------

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=z_matrix[
                    display_columns
                ].values,

                x=display_columns,

                y=heatmap_theme_order,

                text=text_matrix.values,

                texttemplate="%{text}",

                textfont=dict(
                    size=12
                ),

                colorscale="Blues",

                zmin=0,

                zmax=100,

                colorbar=dict(
                    title="% of theme's<br>articles",
                    ticksuffix="%"
                ),

                xgap=3,

                ygap=3,

                hovertemplate=(
                    "Theme: %{y}<br>"
                    "Column: %{x}<br>"
                    "Articles: %{text}"
                    "<extra></extra>"
                )
            )
        )


        fig_heatmap.update_layout(
            height=500,

            xaxis_title="",

            yaxis_title="",

            margin=dict(
                l=10,
                r=20,
                t=20,
                b=20
            )
        )

        fig_heatmap.update_yaxes(
            categoryorder="array",

            categoryarray=heatmap_theme_order,

            autorange="reversed",

            automargin=True
        )


        st.plotly_chart(
            fig_heatmap,
            use_container_width=True,
            config={
                "responsive": True
            }
        )


        # ----------------------------------------------------
        # CHART TITLE — BOTTOM, CENTERED, ITALIC
        # ----------------------------------------------------

        st.markdown(
            """
            <div style="
                font-size: 14px;
                font-weight: 400;
                font-style: italic;
                color: black;
                text-align: center;
                margin-top: -18px;
                margin-bottom: 10px;
            ">
                Theme × Stance Heat Map
            </div>
            """,
            unsafe_allow_html=True
        )

    # ----------------------------------------------------
        # EXPLANATORY TEXT
        # ----------------------------------------------------

        st.markdown(
            """
            <div style="
                text-align: left;
                color: black;
                font-size: 14px;
                margin-bottom: 10px;
            ">
            Cell color shows each theme's stance <em>mix</em> — the share of
            that theme's articles falling into each stance — so themes of
            different sizes can be compared fairly. Numbers show the actual
            article count. The Total column (uncolored) shows the cumulative
            number of articles for that theme.
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("---")

    # ========================================================
    # RESEARCH QUESTION 3
    # ========================================================

    st.markdown(
        """
        <div style="
            font-size: 28px;
            font-weight: 800;
            margin-top: 10px;
            margin-bottom: 4px;
        ">
        3. How has the story changed over time?
        </div>

        <div style="
            font-size: 15px;
            font-style: italic;
            font-weight: 400;
            color: black;
            margin-bottom: 10px;
        ">
        Tracking shifts in themes and stances over time as real-world events unfold, 
        showing how coverage responds to developments on the ground.
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # MONTHLY SUMMARY
    # ========================================================

    monthly_df = df.copy()

    monthly_df["published_date"] = pd.to_datetime(
        monthly_df["published_at"],
        errors="coerce"
    )

    monthly_df = monthly_df[
        monthly_df["published_date"].notna()
    ].copy()

    if monthly_df.empty:

        st.warning(
            "No valid publication dates are available for monthly analysis."
        )

    else:

        monthly_df["month"] = (
            monthly_df["published_date"]
            .dt.to_period("M")
        )


        # ----------------------------------------------------
        # CREATE UNIQUE ARTICLE ID
        # ----------------------------------------------------

        if "url" in monthly_df.columns:

            monthly_df["article_id"] = (
                monthly_df["url"]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        else:

            monthly_df["article_id"] = ""

        missing_id = monthly_df["article_id"].eq("")

        monthly_df.loc[
            missing_id,
            "article_id"
        ] = (
            "row_"
            + monthly_df.index.astype(str)
        )

        monthly_df = monthly_df.drop_duplicates(
            subset="article_id"
        )


        # ----------------------------------------------------
        # AVAILABLE MONTHS
        # ----------------------------------------------------

        available_months = sorted(
            monthly_df["month"]
            .dropna()
            .unique(),
            reverse=True
        )

        month_labels = {
            month.strftime("%Y-%m"): month.strftime("%B %Y")
            for month in available_months
        }

        selected_month_key = st.selectbox(
            "Select month:",
            options=list(month_labels.keys()),
            format_func=lambda key: month_labels[key],
            key="rq3_selected_month"
        )

        selected_month = pd.Period(
            selected_month_key,
            freq="M"
        )

        selected_month_df = monthly_df[
            monthly_df["month"] == selected_month
        ].copy()

        selected_month_name = (
            selected_month.strftime("%B %Y")
        )


        # ----------------------------------------------------
        # MONTHLY ARTICLE COUNT
        # ----------------------------------------------------

        monthly_article_count = (
            selected_month_df["article_id"]
            .nunique()
        )


        # ----------------------------------------------------
        # MONTHLY THEME COUNTS
        # ----------------------------------------------------

        monthly_theme_df = selected_month_df.copy()

        monthly_theme_df["themes"] = (
            monthly_theme_df["themes"]
            .fillna("")
            .astype(str)
            .str.split(", ")
        )

        monthly_theme_df = (
            monthly_theme_df
            .explode("themes")
        )

        monthly_theme_df = monthly_theme_df[
            monthly_theme_df["themes"].isin(
                THEME_ORDER
            )
        ].copy()

        monthly_theme_df = (
            monthly_theme_df
            .drop_duplicates(
                subset=[
                    "article_id",
                    "themes"
                ]
            )
        )

        monthly_theme_counts = (
            monthly_theme_df
            .groupby("themes")["article_id"]
            .nunique()
            .reindex(
                THEME_ORDER,
                fill_value=0
            )
        )

        ranked_monthly_themes = sorted(
            monthly_theme_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        leading_monthly_themes = [
            (theme, count)
            for theme, count in ranked_monthly_themes
            if count > 0
        ]


        # ----------------------------------------------------
        # MONTHLY STANCE
        # ----------------------------------------------------

        monthly_stance_counts = (
            selected_month_df["stance"]
            .fillna("Neutral")
            .astype(str)
            .str.strip()
            .str.title()
            .replace({
                "Positive": "Supportive",
                "Negative": "Critical"
            })
            .value_counts()
            .reindex(
                STANCE_ORDER,
                fill_value=0
            )
        )

        monthly_stance_total = (
            monthly_stance_counts.sum()
        )

        if monthly_stance_total > 0:

            monthly_dominant_stance = (
                monthly_stance_counts.idxmax()
            )

            monthly_dominant_stance_count = (
                monthly_stance_counts.max()
            )

            monthly_dominant_stance_share = (
                monthly_dominant_stance_count
                / monthly_stance_total
                * 100
            )

        else:

            monthly_dominant_stance = "Neutral"
            monthly_dominant_stance_share = 0


        # ----------------------------------------------------
        # SIGNIFICANT EVENTS
        # ----------------------------------------------------

        monthly_events_df = (
            prepare_significant_events(
                SIGNIFICANT_EVENTS
            )
        )

        if not monthly_events_df.empty:

            monthly_events_df["event_month"] = (
                monthly_events_df["date"]
                .dt.to_period("M")
            )

            monthly_selected_events = (
                monthly_events_df[
                    monthly_events_df["event_month"]
                    == selected_month
                ].copy()
            )

        else:

            monthly_selected_events = pd.DataFrame()


        # ----------------------------------------------------
        # IDENTIFY EMERGING THEME
        # ----------------------------------------------------

        if not leading_monthly_themes:

            monthly_theme_text = (
                "No clearly dominant theme was detected in the "
                "available coverage."
            )

        elif len(leading_monthly_themes) == 1:

            theme, count = leading_monthly_themes[0]

            monthly_theme_text = (
                f"{theme} was the most prominent theme, appearing in "
                f"{count} article"
                f"{'s' if count != 1 else ''}."
            )

        else:

            theme_1, count_1 = leading_monthly_themes[0]
            theme_2, count_2 = leading_monthly_themes[1]

            monthly_theme_text = (
                f"{theme_1} led the coverage with "
                f"{count_1} article"
                f"{'s' if count_1 != 1 else ''}, followed by "
                f"{theme_2} with "
                f"{count_2} article"
                f"{'s' if count_2 != 1 else ''}."
            )


        # ----------------------------------------------------
        # EVENT CONTEXT
        # ----------------------------------------------------

        if not monthly_selected_events.empty:

            event_labels = (
                monthly_selected_events["label"]
                .dropna()
                .astype(str)
                .tolist()
            )

            if len(event_labels) == 1:

                monthly_event_text = (
                    f"This coverage coincided with "
                    f"{event_labels[0]}."
                )

            else:

                monthly_event_text = (
                    "This coverage coincided with "
                    + ", ".join(event_labels[:-1])
                    + " and "
                    + event_labels[-1]
                    + "."
                )

        else:

            monthly_event_text = (
                "No significant developments in the event tracker "
                "were recorded for this month."
            )


        # ----------------------------------------------------
        # BUILD MONTHLY SUMMARY
        # ----------------------------------------------------

        monthly_summary = (
            f"In {selected_month_name}, the corpus contained "
            f"{monthly_article_count} unique article"
            f"{'s' if monthly_article_count != 1 else ''}. "
            f"{monthly_theme_text} "
            f"The detected stance of the coverage was predominantly "
            f"{monthly_dominant_stance.lower()}, accounting for "
            f"{monthly_dominant_stance_share:.1f}% of articles. "
            f"{monthly_event_text}"
        )


        # ----------------------------------------------------
        # DISPLAY MONTHLY SUMMARY
        # ----------------------------------------------------

        st.markdown(
            f"""
            <h5 style="margin-bottom: 0.2rem;">
                {selected_month_name} — What is the dominant theme and stance?
            </h5>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            monthly_summary
        )

        st.caption(
            "Theme counts are not mutually exclusive because a single "
            "article may be classified under more than one theme."
        )


    # ========================================================
    # THEME × STANCE BUBBLE MATRIX
    # ========================================================

    bubble_df = df.copy()


    # --------------------------------------------------------
    # CLEAN STANCE LABELS
    # --------------------------------------------------------

    bubble_df["stance"] = (
        bubble_df["stance"]
        .fillna("Neutral")
        .astype(str)
        .str.strip()
        .str.title()
        .replace({
            "Positive": "Supportive",
            "Negative": "Critical"
        })
    )

    bubble_df = bubble_df[
        bubble_df["stance"].isin(STANCE_ORDER)
    ].copy()


    # --------------------------------------------------------
    # CONVERT PUBLICATION TIMESTAMP TO DATETIME
    # --------------------------------------------------------

    bubble_df["date"] = pd.to_datetime(
        bubble_df["published_at"],
        errors="coerce"
    ).dt.normalize()  # truncate to date, drop time-of-day


    # --------------------------------------------------------
    # PREPARE ARTICLE TITLE
    # --------------------------------------------------------

    if "title" in bubble_df.columns:

        bubble_df["article_title"] = (
            bubble_df["title"]
            .fillna("Untitled article")
            .astype(str)
        )

    else:

        bubble_df["article_title"] = (
            "Untitled article"
        )


    # --------------------------------------------------------
    # PREPARE PUBLICATION DATE FOR HOVER
    # --------------------------------------------------------

    bubble_df["published_date_display"] = (
        bubble_df["date"]
        .dt.strftime("%d %b %Y")
        .fillna("Unknown date")
    )


    # --------------------------------------------------------
    # HOVER TEXT
    # --------------------------------------------------------

    bubble_df["article_hover"] = (
        "Article Title: "
        + bubble_df["article_title"]
        + "<br>Date Published: "
        + bubble_df["published_date_display"]
    )


    # --------------------------------------------------------
    # SPLIT MULTI-LABEL THEMES
    # --------------------------------------------------------

    bubble_df["themes"] = (
        bubble_df["themes"]
        .fillna("")
        .astype(str)
        .str.split(", ")
    )

    bubble_df = bubble_df.explode(
        "themes"
    )


    # --------------------------------------------------------
    # KEEP ONLY DEFINED THEMES
    # --------------------------------------------------------

    bubble_df = bubble_df[
        bubble_df["themes"].isin(
            THEME_ORDER
        )
    ].copy()


    # --------------------------------------------------------
    # REMOVE ROWS WITHOUT USABLE DATES
    # --------------------------------------------------------

    bubble_df = bubble_df[
        bubble_df["date"].notna()
    ].copy()


    # --------------------------------------------------------
    # FORCE THEME COLUMN INTO FIXED CATEGORY ORDER
    # --------------------------------------------------------

    bubble_df["themes"] = pd.Categorical(
        bubble_df["themes"],
        categories=THEME_ORDER,
        ordered=True
    )

    all_theme_labels = THEME_ORDER.copy()

    theme_labels = {
        theme: theme
        for theme in THEME_ORDER
    }


    # --------------------------------------------------------
    # AGGREGATE ARTICLES BY DATE + THEME + STANCE
    # --------------------------------------------------------

    bubble_data = (
        bubble_df
        .groupby(
            [
                "date",
                "themes",
                "stance"
            ],
            observed=False
        )
        .size()
        .reset_index(
            name="article_count"
        )
    )


    # --------------------------------------------------------
    # PREPARE HOVER INFORMATION
    # --------------------------------------------------------

    hover_data = (
        bubble_df
        .groupby(
            [
                "date",
                "themes",
                "stance"
            ],
            observed=False
        )["article_hover"]
        .apply(
            lambda values:
                "<br><br>".join(
                    values.astype(str)
                )
        )
        .reset_index(
            name="article_hover"
        )
    )

    bubble_data = bubble_data.merge(
        hover_data,
        on=[
            "date",
            "themes",
            "stance"
        ],
        how="left"
    )


    # --------------------------------------------------------
    # CREATE THEME LABELS
    # --------------------------------------------------------

    bubble_data["theme_label"] = (
        bubble_data["themes"]
        .map(theme_labels)
    )

    bubble_data["theme_label"] = pd.Categorical(
        bubble_data["theme_label"],
        categories=all_theme_labels,
        ordered=True
    )


    # --------------------------------------------------------
    # SORT DATA
    # --------------------------------------------------------

    bubble_data = bubble_data.sort_values(
        [
            "themes",
            "date"
        ]
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # PREPARE SIGNIFICANT EVENTS
    # --------------------------------------------------------

    events_df = prepare_significant_events(
        SIGNIFICANT_EVENTS
    )


    # ========================================================
    # GENERATE BUBBLE MATRIX
    # ========================================================

    if not bubble_data.empty:

        fig_bubble = px.scatter(
            bubble_data,

            x="date",

            y="theme_label",

            size="article_count",

            color="stance",

            size_max=45,

            custom_data=[
                "article_hover"
            ],

            category_orders={
                "theme_label": all_theme_labels,

                "stance": STANCE_ORDER
            },

            labels={
                "date": "Publication Date",
                "theme_label": "Theme",
                "stance": "Stance",
                "article_count": "Articles"
            }
        )


        # ----------------------------------------------------
        # HOVER INFORMATION
        # ----------------------------------------------------

        fig_bubble.update_traces(
            hovertemplate=(
                "%{customdata[0]}"
                "<extra></extra>"
            )
        )


        # ----------------------------------------------------
        # FORCE ALL SEVEN THEMES ON Y-AXIS
        # ----------------------------------------------------

        fig_bubble.update_yaxes(
            categoryorder="array",
            categoryarray=all_theme_labels,
            autorange="reversed",
            automargin=True
        )


        # ----------------------------------------------------
        # BASE CHART LAYOUT
        # ----------------------------------------------------

        fig_bubble.update_layout(
            height=600,
            autosize=True,
            xaxis_title="Publication Date",
            yaxis_title="Theme",
            legend_title="Stance",
            hovermode="closest",
            margin=dict(
                l=10,
                r=20,
                t=80,
                b=70
            )
        )

        fig_bubble.update_xaxes(
            tickangle=45,
            automargin=True
        )


        # ----------------------------------------------------
        # ADD SIGNIFICANT-EVENT MARKERS
        # ----------------------------------------------------

        if not events_df.empty:

            chart_min_date = (
                bubble_data["date"].min()
            )

            chart_max_date = (
                bubble_data["date"].max()
            )

            visible_events = events_df[
                (
                    events_df["date"]
                    >= chart_min_date
                )
                & (
                    events_df["date"]
                    <= chart_max_date
                )
            ].copy()


            # =================================================
            # EVENT TYPE
            # =================================================

            domestic_events = (
                visible_events[
                    visible_events["event_type"]
                    == "Domestic"
                ].copy()
            )

            international_events = (
                visible_events[
                    visible_events["event_type"]
                    == "International"
                ].copy()
            )


            # -------------------------------------------------
            # LABEL POSITIONING
            # -------------------------------------------------

            domestic_label_rows = [
                1.015,
                1.065,
                1.115,
                1.165
            ]

            international_label_rows = [
                -0.115,
                -0.175,
                -0.235,
                -0.295
            ]


            # -------------------------------------------------
            # ESTIMATE LABEL WIDTH
            # -------------------------------------------------

            def estimate_label_width(label):

                return max(
                    4,
                    len(str(label)) * 0.42
                )


            # -------------------------------------------------
            # ASSIGN EVENTS TO LABEL ROWS
            # -------------------------------------------------

            def assign_event_rows(
                event_subset,
                label_rows
            ):

                row_last_date = [
                    None
                    for _ in label_rows
                ]

                row_last_width = [
                    0
                    for _ in label_rows
                ]

                assignments = []

                for _, event in event_subset.iterrows():

                    event_date = event["date"]

                    label = str(
                        event["label"]
                    )

                    current_width = (
                        estimate_label_width(
                            label
                        )
                    )

                    selected_row = None


                    # -----------------------------------------
                    # TRY EACH ROW
                    # -----------------------------------------

                    for row_index in range(
                        len(label_rows)
                    ):

                        if (
                            row_last_date[row_index]
                            is None
                        ):

                            selected_row = row_index
                            break

                        previous_date = (
                            row_last_date[row_index]
                        )

                        previous_width = (
                            row_last_width[row_index]
                        )

                        required_gap = (
                            (
                                previous_width
                                + current_width
                            ) / 2
                            + 2.0
                        )

                        actual_gap = abs(
                            (
                                event_date
                                - previous_date
                            ).total_seconds()
                            / 86400
                        )

                        if actual_gap >= required_gap:

                            selected_row = row_index
                            break


                    # -----------------------------------------
                    # IF NO ROW IS FREE, USE GREATEST GAP
                    # -----------------------------------------

                    if selected_row is None:

                        available_gaps = []

                        for row_index in range(
                            len(label_rows)
                        ):

                            previous_date = (
                                row_last_date[row_index]
                            )

                            previous_width = (
                                row_last_width[row_index]
                            )

                            required_gap = (
                                (
                                    previous_width
                                    + current_width
                                ) / 2
                                + 2.0
                            )

                            actual_gap = abs(
                                (
                                    event_date
                                    - previous_date
                                ).total_seconds()
                                / 86400
                            )

                            available_gaps.append(
                                actual_gap
                                - required_gap
                            )

                        selected_row = (
                            available_gaps.index(
                                max(available_gaps)
                            )
                        )


                    # -----------------------------------------
                    # STORE ASSIGNMENT
                    # -----------------------------------------

                    assignments.append(
                        (
                            event,
                            selected_row
                        )
                    )

                    row_last_date[
                        selected_row
                    ] = event_date

                    row_last_width[
                        selected_row
                    ] = current_width

                return assignments


            domestic_assignments = (
                assign_event_rows(
                    domestic_events,
                    domestic_label_rows
                )
            )

            international_assignments = (
                assign_event_rows(
                    international_events,
                    international_label_rows
                )
            )


            # =================================================
            # COMBINE EVENT ASSIGNMENTS
            # =================================================

            all_assignments = []

            for event, row_index in domestic_assignments:

                all_assignments.append(
                    (
                        event,
                        domestic_label_rows[row_index],
                        "top"
                    )
                )

            for event, row_index in international_assignments:

                all_assignments.append(
                    (
                        event,
                        international_label_rows[row_index],
                        "bottom"
                    )
                )


            # =================================================
            # ADD EVENT MARKERS AND LABELS
            # =================================================

            for (
                event,
                label_y,
                label_position
            ) in all_assignments:

                event_date = event["date"]

                label = str(
                    event["label"]
                )


                # ---------------------------------------------
                # VERTICAL EVENT MARKER
                # ---------------------------------------------

                fig_bubble.add_shape(
                    type="line",
                    x0=event_date,
                    x1=event_date,
                    y0=0.04,
                    y1=0.94,
                    xref="x",
                    yref="paper",
                    line=dict(
                        width=0.8,
                        dash="dot",
                        color="gray"
                    )
                )


                # ---------------------------------------------
                # EVENT LABEL
                # ---------------------------------------------

                annotation_kwargs = dict(
                    x=event_date,
                    y=label_y,
                    xref="x",
                    yref="paper",
                    text=label,
                    showarrow=False,
                    font=dict(
                        size=10,
                        color="black"
                    ),
                    xanchor="center",
                    yanchor=(
                        "bottom"
                        if label_position == "top"
                        else "top"
                    ),
                    align="center",
                    bgcolor="rgba(255,255,255,0.85)",
                    borderpad=1
                )

                description = str(
                    event["description"]
                )

                fig_bubble.add_annotation(
                    **annotation_kwargs,
                    hovertext=description,
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=11
                    )
                )

        # ========================================================
        # BUBBLE MATRIX TAKEAWAY
        # ========================================================

        if not events_df.empty:

            # Use a window around each event rather than exact-date
            # matching — coverage responding to an announcement, forum,
            # or SONA mention typically builds in the days afterward,
            # not just on the event's own calendar day. Exact-date
            # matching was pinning the takeaway to a near-fixed handful
            # of articles that barely changes as the corpus grows.
            EVENT_WINDOW_DAYS = 7

            event_dates = events_df["date"].dropna()

            event_period_mask = pd.Series(False, index=bubble_data.index)
            for event_date in event_dates:
                window_start = event_date - pd.Timedelta(days=1)
                window_end = event_date + pd.Timedelta(days=EVENT_WINDOW_DAYS)
                event_period_mask |= bubble_data["date"].between(window_start, window_end)

            event_period_data = bubble_data[event_period_mask].copy()

            if not event_period_data.empty:

                # Find the (theme, stance) combo with the most articles —
                # not theme and stance maxed independently, which can produce
                # a headline that doesn't match any actual cluster in the chart.
                combo_counts = (
                    event_period_data
                    .groupby(["themes", "stance"], observed=False)["article_count"]
                    .sum()
                    .sort_values(ascending=False)
                )

                top_theme, top_stance = combo_counts.index[0]

                # Separately check whether Critical coverage is concentrated
                # in a specific theme — often the more meaningful signal
                # even when it isn't the single largest combo overall.
                critical_by_theme = (
                    event_period_data[event_period_data["stance"] == "Critical"]
                    .groupby("themes", observed=False)["article_count"]
                    .sum()
                    .sort_values(ascending=False)
                )

                if not critical_by_theme.empty and critical_by_theme.iloc[0] > 0:
                    critical_theme = critical_by_theme.index[0]
                else:
                    critical_theme = None

                if critical_theme and critical_theme != top_theme:
                    bubble_takeaway = (
                        f"Around significant events, coverage was most concentrated on "
                        f"{top_theme} with a predominantly {top_stance} stance, while "
                        f"Critical coverage clustered specifically around {critical_theme}."
                    )
                else:
                    bubble_takeaway = (
                        f"Coverage around significant events is most strongly "
                        f"concentrated on {top_theme}, with {top_stance} "
                        f"emerging as the dominant stance during these periods."
                    )

            else:
                bubble_takeaway = (
                    "Coverage around significant events shows shifts in "
                    "dominant themes and stances across the coverage period."
                )

        else:
            bubble_takeaway = (
                "Coverage around significant events shows shifts in "
                "dominant themes and stances across the coverage period."
            )

        # --------------------------------------------------------
        # CRITICAL COVERAGE TREND SINCE SONA
        # --------------------------------------------------------
        # Checks whether Critical coverage's share of total coverage
        # rose meaningfully after the SONA mention specifically — this
        # is the shift you noticed visually in the chart, so it's worth
        # calling out explicitly rather than leaving the takeaway static
        # on "what theme/stance dominates" alone.

        sona_events = events_df[
            events_df["label"].astype(str).str.contains("SONA", case=False, na=False)
        ]

        if not sona_events.empty:

            sona_date = sona_events["date"].iloc[0]

            before_sona = bubble_data[bubble_data["date"] < sona_date]
            after_sona = bubble_data[bubble_data["date"] >= sona_date]

            def critical_share(data):
                total = data["article_count"].sum()
                if total == 0:
                    return None
                critical_count = data.loc[
                    data["stance"] == "Critical", "article_count"
                ].sum()
                return critical_count / total

            share_before = critical_share(before_sona)
            share_after = critical_share(after_sona)

            # Only surface this if the shift is large enough to be a real
            # pattern rather than noise from a small before-period sample.
            if (
                share_before is not None
                and share_after is not None
                and (share_after - share_before) >= 0.10
            ):
                bubble_takeaway += (
                    f" Critical coverage has climbed since the SONA mention — "
                    f"from {share_before:.0%} of coverage beforehand to "
                    f"{share_after:.0%} afterward."
                )

        # ----------------------------------------------------
        # DISPLAY TAKEAWAY ABOVE THE BUBBLE CHART
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 22px;
                font-weight: 700;
                color: black;
                margin-top: 10px;
                margin-bottom: 10px;
            ">
            {bubble_takeaway}
            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # [MERGE FIX] DISPLAY THE BUBBLE MATRIX
        # This call was missing from the source you sent — the
        # figure was fully built (scatter points, event lines,
        # event labels) but never rendered to the page, which is
        # why the chart appeared to have vanished.
        # ----------------------------------------------------

        st.plotly_chart(
            fig_bubble,
            use_container_width=True,
            config={
                "responsive": True
            }
        )


        # ----------------------------------------------------
        # DISPLAY CHART TITLE BELOW THE BUBBLE CHART
        # ----------------------------------------------------

        st.markdown(
            """
            <div style="
                text-align: center;
                font-size: 15px;
                font-style: italic;
                font-weight: 400;
                color: black;
                margin-top: 4px;
                margin-bottom: 18px;
            ">
            News Coverage Over Time
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.info(
            "No dated, theme-tagged articles are available yet to "
            "plot on the coverage-over-time chart."
        )


    # ========================================================
    # SIGNIFICANT EVENTS TABLE
    # ========================================================

    if not bubble_data.empty and not events_df.empty:

        if not events_df.empty:

            chart_min_date = (
                bubble_data["date"].min()
            )

            chart_max_date = (
                bubble_data["date"].max()
            )

            visible_events = events_df[
                (
                    events_df["date"]
                    >= chart_min_date
                )
                & (
                    events_df["date"]
                    <= chart_max_date
                )
            ].copy()

            if not visible_events.empty:

                st.markdown(
                    "**Significant events during the monitoring period**"
                )

                event_display = visible_events.copy()

                event_display["date"] = (
                    event_display["date"]
                    .dt.strftime("%d %b %Y")
                )

                event_display = event_display.rename(
                    columns={
                        "date": "Date",
                        "label": "Event",
                        "description": "Description"
                    }
                )

                st.dataframe(
                    event_display[
                        [
                            "Date",
                            "Event",
                            "Description"
                        ]
                    ],
                    hide_index=True,
                    use_container_width=True
                )

            else:

                # [MERGE FIX] Original message here was copy-pasted from
                # the bubble-matrix empty-state ("...to generate the
                # bubble matrix"), which doesn't describe this table.
                # Replaced with wording that matches what's actually
                # being checked.
                st.info(
                    "No significant events fall within the current "
                    "monitoring period."
                )


    st.markdown(
        "<hr style='border: none; border-top: 1px solid #ddd; margin: 8px 0 20px 0;'>",
        unsafe_allow_html=True
    )
    
    # ========================================================
    # RESEARCH QUESTION 4
    # ========================================================

    st.markdown(
        """
        <div style="
            font-size: 28px;
            font-weight: 800;
            margin-top: 10px;
            margin-bottom: 4px;
        ">
        4. How do local state-owned, local independent media and international outlets differ in their coverage of Pax Silica?
        </div>

        <div style="
            font-size: 15px;
            font-style: italic;
            font-weight: 400;
            color: black;
            margin-bottom: 10px;
        ">
        If I only monitor from a certain type of outlet, what perspective might I be missing?
        </div>
        """,
        unsafe_allow_html=True
    ) 
        # ========================================================
    # OUTLET TYPE MAPPING
    # ========================================================
    # A few sources are excluded rather than mapped — see EXCLUDED_SOURCES
    # below. facebook.com posts don't identify which page posted them, and
    # "ph" is a URL-parsing artifact, not a real outlet name.

    OUTLET_TYPE_MAP = {

        # --- State-Owned Local ---
        "Philippine Information Agency": "State-Owned Local",
        "Philippine News Agency": "State-Owned Local",
        "ptvnews.ph": "State-Owned Local",
        "pco.gov.ph": "State-Owned Local",

        # --- Independent Local ---
        "The Tribune": "Independent Local",
        "Daily Tribune": "Independent Local",
        "Tribune.net.ph": "Independent Local",
        "DZRH": "Independent Local",
        "politiko.com.ph": "Independent Local",
        "Inquirer.net": "Independent Local",
        "newsinfo.inquirer.net": "Independent Local",
        "Philippine Daily Inquirer": "Independent Local",
        "Philippine Daily Inquirer - Business": "Independent Local",
        "Philippine Daily Inquirer - Newsinfo": "Independent Local",
        "Inquirer Technology": "Independent Local",
        "Manila Standard": "Independent Local",
        "Punto! Central Luzon": "Independent Local",
        "Philstar.com": "Independent Local",
        "Philippine Star": "Independent Local",
        "Philstar Life": "Independent Local",
        "SunStar Publishing Inc.": "Independent Local",
        "Cebu Daily News": "Independent Local",
        "Bulatlat": "Independent Local",
        "Rappler": "Independent Local",
        "Newsbytes.PH": "Independent Local",
        "Explained PH": "Independent Local",
        "Manila Bulletin": "Independent Local",
        "ABS-CBN": "Independent Local",
        "ABS-CBN News": "Independent Local",
        "Abs Cbn": "Independent Local",
        "abs-cbn.com": "Independent Local",
        "Interaksyon": "Independent Local",
        "GMA Network": "Independent Local",
        "GMA News": "Independent Local",
        "The Voice Newsweekly": "Independent Local",
        "pageone.ph": "Independent Local",
        "Kami.com.ph": "Independent Local",
        "Sugbo.ph": "Independent Local",
        "The Summit Express": "Independent Local",
        "Pampanga News Now": "Independent Local",
        "BusinessWorld Online": "Independent Local",
        "Bworldonline": "Independent Local",
        "BusinessMirror": "Independent Local",
        "Malaya Business Insight": "Independent Local",
        "PEP.ph": "Independent Local",
        "The Manila Times": "Independent Local",
        "Manilatimes": "Independent Local",
        "negrosnowdaily.com": "Independent Local",
        "Davao Today": "Independent Local",
        "The GUIDON": "Independent Local",
        "startupfortune.com": "Independent Local",
        "thephilbiznews": "Independent Local",
        "InsiderPH": "Independent Local",
        "Bilyonaryo Business": "Independent Local",
        "Technobaboy": "Independent Local",
        "Karapatan": "Independent Local",  # advocacy org, not a press outlet — flagged in write-up
        "Raissa Robles": "Independent Local",  # PH-based; also an SCMP correspondent — ambiguous, see note
        "Sam Beltran": "Independent Local",  # unconfirmed outlet affiliation
        "Mindanao Gold Star Daily": "Independent Local",

        # --- International ---
        "Gulf News": "International",
        "ANI News": "International",
        "South China Morning Post": "International",
        "scmp.com": "International",
        "visiontimes.com": "International",
        "W.Media": "International",
        "Supply Chain Digital Magazine": "International",
        "Crypto Briefing": "International",
        "Taiwan News": "International",
        "International Business Times UK": "International",
        "Eco-Business": "International",
        "fDi Intelligence": "International",
        "Global Times": "International",
        "Radio Free Asia": "International",
        "Jason Gutierrez For Rfa": "International",
        "Arab News": "International",
        "AFP": "International",
        "Agence France-Presse": "International",
        "Decode39": "International",
        "Jay Hilotin": "International",  # Gulf News correspondent
        "Ellie Aben": "International",  # Gulf News correspondent
        "The Foundation for American Innovation": "International",
        "Peace Brigades International-Canada": "International",  # advocacy org, not a press outlet
        "ChemAnalyst": "International",
        "The Diplomat – Asia-Pacific Current Affairs Magazine": "International",
        "canadianinquirer.net": "International",
    }


    # --------------------------------------------------------
    # SOURCES TO EXCLUDE FROM THIS COMPARISON
    # --------------------------------------------------------
    # facebook.com doesn't identify which page posted the article, and
    # "ph" is a URL-parsing artifact rather than a real outlet name —
    # neither can be reliably classified by outlet type.

    EXCLUDED_SOURCES = [
        "facebook.com",
        "ph",
    ]


    # ========================================================
    # OUTLET TYPE × STANCE CHART
    # ========================================================

    outlet_df = df[
        ~df["source"].isin(EXCLUDED_SOURCES)
    ].copy()

    outlet_df["outlet_type"] = (
        outlet_df["source"]
        .map(OUTLET_TYPE_MAP)
        .fillna("Independent Local")
    )

    outlet_df["stance"] = (
        outlet_df["stance"]
        .fillna("Neutral")
        .astype(str)
        .str.strip()
        .str.title()
        .replace({
            "Positive": "Supportive",
            "Negative": "Critical"
        })
    )

    outlet_df = outlet_df[
        outlet_df["stance"].isin(STANCE_ORDER)
    ].copy()


    # --------------------------------------------------------
    # AGGREGATE COUNTS BY OUTLET TYPE + STANCE
    # --------------------------------------------------------

    outlet_stance = (
        outlet_df
        .groupby(
            [
                "outlet_type",
                "stance"
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )


    # --------------------------------------------------------
    # CONVERT TO PERCENT OF EACH OUTLET TYPE'S TOTAL
    # --------------------------------------------------------

    outlet_stance["percent"] = (
        outlet_stance
        .groupby("outlet_type")["count"]
        .transform(
            lambda x: x / x.sum() * 100
        )
    )

        # --------------------------------------------------------
    # OUTLET TYPE ORDER
    # --------------------------------------------------------

    OUTLET_TYPE_ORDER = [
        "State-Owned Local",
        "Independent Local",
        "International"
    ]


    # --------------------------------------------------------
    # OUTLET TYPE TAKEAWAY
    # --------------------------------------------------------
    # Always names both the most Supportive and most Critical outlet
    # type, so the reader gets the full contrast rather than only one
    # side of it.

    if not outlet_stance.empty:

        supportive_shares = (
            outlet_stance[
                outlet_stance["stance"] == "Supportive"
            ]
            .set_index("outlet_type")["percent"]
        )

        critical_shares = (
            outlet_stance[
                outlet_stance["stance"] == "Critical"
            ]
            .set_index("outlet_type")["percent"]
        )

        # Outlet types with zero articles in a given stance won't appear
        # in the groupby result above, so fill in 0% for any missing type.
        for outlet_type in outlet_stance["outlet_type"].unique():
            if outlet_type not in supportive_shares.index:
                supportive_shares[outlet_type] = 0.0
            if outlet_type not in critical_shares.index:
                critical_shares[outlet_type] = 0.0

        highest_supportive_type = supportive_shares.idxmax()
        highest_supportive_value = supportive_shares.max()

        highest_critical_type = critical_shares.idxmax()
        highest_critical_value = critical_shares.max()

        outlet_takeaway = (
            f"{highest_supportive_type} coverage leans most Supportive "
            f"({highest_supportive_value:.0f}% of its coverage), while "
            f"{highest_critical_type} coverage leans most Critical "
            f"({highest_critical_value:.0f}%)."
        )

    else:
        outlet_takeaway = (
            "No outlet-tagged, stance-classified articles are available "
            "yet to compare coverage by outlet type."
        )


        # --------------------------------------------------------
    # BUILD CHART
    # --------------------------------------------------------

    if not outlet_stance.empty:

        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 20px;
                font-weight: 700;
                color: black;
                margin-top: 10px;
                margin-bottom: 10px;
            ">
            {outlet_takeaway}
            </div>
            """,
            unsafe_allow_html=True
        )

        STANCE_COLOR_MAP = {
            "Supportive": "#1f77b4",
            "Neutral": "#a9c6e8",
            "Critical": "#d62728",
        }

        fig_outlet = px.bar(
            outlet_stance,

            x="percent",

            y="outlet_type",

            color="stance",

            orientation="h",

            category_orders={
                "outlet_type": OUTLET_TYPE_ORDER,
                "stance": STANCE_ORDER
            },

            color_discrete_map=STANCE_COLOR_MAP,

            labels={
                "percent": "% of Coverage",
                "outlet_type": "Outlet Type",
                "stance": "Stance"
            }
        )

        fig_outlet.update_yaxes(
            categoryorder="array",
            categoryarray=OUTLET_TYPE_ORDER,
            autorange="reversed"
        )

        fig_outlet.update_layout(
            barmode="stack",
            height=350,
            autosize=True,
            xaxis_title="% of Coverage",
            yaxis_title="Outlet Type",
            legend_title="Stance",
            margin=dict(
                l=10,
                r=20,
                t=40,
                b=40
            )
        )

        _, chart_col, _ = st.columns([1, 2, 1])
        with chart_col:
            st.plotly_chart(
                fig_outlet,
                use_container_width=True,
                config={
                    "responsive": True
                }
            )

        # ----------------------------------------------------
        # CHART TITLE: BOTTOM, CENTERED, ITALIC
        # ----------------------------------------------------

        st.markdown(
            """
            <div style="
                font-size: 14px;
                font-weight: 400;
                font-style: italic;
                color: black;
                text-align: center;
                margin-top: -8px;
                margin-bottom: 10px;
            ">
                Stance by Outlet Type
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.info(
            "No outlet-tagged, stance-classified articles are available "
            "yet to compare coverage by outlet type."
        )
	    

    # ========================================================
    # OUTLET TYPE × THEME CHART  [NEW]
    # ========================================================
    # Shows which themes each outlet type covers. Values are the share
    # of that outlet type's unique articles that touch each theme, so
    # outlet types with very different article volumes can be compared
    # fairly. Because an article can carry more than one theme, each
    # column can add up to more than 100%.

    st.markdown(
        "<hr style='border: none; border-top: 1px solid #ddd; margin: 8px 0 20px 0;'>",
        unsafe_allow_html=True
    )

    theme_outlet_df = outlet_df.copy()


    # --------------------------------------------------------
    # UNIQUE ARTICLE ID (same approach used in the monthly summary)
    # --------------------------------------------------------

    if "url" in theme_outlet_df.columns:

        theme_outlet_df["article_id"] = (
            theme_outlet_df["url"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    else:

        theme_outlet_df["article_id"] = ""

    missing_theme_outlet_id = theme_outlet_df["article_id"].eq("")

    theme_outlet_df.loc[
        missing_theme_outlet_id,
        "article_id"
    ] = (
        "row_"
        + theme_outlet_df.index.astype(str)
    )

    theme_outlet_df = theme_outlet_df.drop_duplicates(
        subset="article_id"
    )


    # --------------------------------------------------------
    # UNIQUE ARTICLES PER OUTLET TYPE (denominator for percentages)
    # --------------------------------------------------------

    outlet_article_totals = (
        theme_outlet_df
        .groupby("outlet_type")["article_id"]
        .nunique()
        .reindex(
            OUTLET_TYPE_ORDER,
            fill_value=0
        )
    )


    # --------------------------------------------------------
    # SPLIT MULTI-LABEL THEMES
    # --------------------------------------------------------

    theme_outlet_df["themes"] = (
        theme_outlet_df["themes"]
        .fillna("")
        .astype(str)
        .str.split(", ")
    )

    theme_outlet_df = theme_outlet_df.explode(
        "themes"
    )

    theme_outlet_df = theme_outlet_df[
        theme_outlet_df["themes"].isin(
            THEME_ORDER
        )
    ].copy()

    theme_outlet_df = theme_outlet_df.drop_duplicates(
        subset=[
            "article_id",
            "themes"
        ]
    )


    # --------------------------------------------------------
    # COUNTS AND PERCENTAGES (THEME × OUTLET TYPE)
    # --------------------------------------------------------

    outlet_theme_counts = (
        theme_outlet_df
        .groupby(
            [
                "themes",
                "outlet_type"
            ]
        )["article_id"]
        .nunique()
        .unstack(
            fill_value=0
        )
        .reindex(
            index=THEME_ORDER,
            columns=OUTLET_TYPE_ORDER,
            fill_value=0
        )
    )

    outlet_theme_pct = (
        outlet_theme_counts
        .div(
            outlet_article_totals.replace(0, np.nan),
            axis=1
        )
        .fillna(0)
        * 100
    )


    if outlet_theme_counts.values.sum() > 0:

        # ----------------------------------------------------
        # TAKEAWAY — MOST-COVERED THEME FOR EACH OUTLET TYPE
        # ----------------------------------------------------
        # Names the theme that takes the largest share of each outlet
        # type's coverage, so the reader sees what each type of outlet
        # mostly talks about. Only outlet types that actually have
        # articles are included, so an empty column can't produce a
        # misleading "top theme".

        valid_outlet_types = [
            outlet_type
            for outlet_type in OUTLET_TYPE_ORDER
            if outlet_article_totals[outlet_type] > 0
        ]

        top_theme_parts = []

        for outlet_type in valid_outlet_types:

            outlet_top_theme = (
                outlet_theme_pct[outlet_type]
                .idxmax()
            )

            outlet_top_share = (
                outlet_theme_pct.loc[outlet_top_theme, outlet_type]
            )

            top_theme_parts.append(
                f"{outlet_type} outlets mostly cover "
                f"{outlet_top_theme} ({outlet_top_share:.0f}%)"
            )

        if len(top_theme_parts) > 1:

            outlet_theme_takeaway = (
                "; ".join(top_theme_parts[:-1])
                + "; and "
                + top_theme_parts[-1]
                + "."
            )

        else:

            outlet_theme_takeaway = (
                top_theme_parts[0] + "."
            )

        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 20px;
                font-weight: 700;
                color: black;
                margin-top: 10px;
                margin-bottom: 10px;
            ">
            {outlet_theme_takeaway}
            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # BUILD HEAT MAP
        # ----------------------------------------------------

        outlet_column_labels = [
            f"{outlet_type.replace(' ', '<br>', 1)}<br>(n={int(outlet_article_totals[outlet_type])})"
            for outlet_type in OUTLET_TYPE_ORDER
        ]

        outlet_theme_text = [
            [
                f"{outlet_theme_pct.loc[theme, outlet_type]:.0f}%"
                f"<br>({int(outlet_theme_counts.loc[theme, outlet_type])})"
                for outlet_type in OUTLET_TYPE_ORDER
            ]
            for theme in THEME_ORDER
        ]

        fig_outlet_theme = go.Figure(
            data=go.Heatmap(
                z=outlet_theme_pct.values,

                x=outlet_column_labels,

                y=THEME_ORDER,

                text=outlet_theme_text,

                texttemplate="%{text}",

                textfont=dict(
                    size=12
                ),

                colorscale="Blues",

                zmin=0,

                zmax=max(
                    float(outlet_theme_pct.values.max()),
                    1.0
                ),

                colorbar=dict(
                    title="% of outlet<br>type's articles",
                    ticksuffix="%"
                ),

                xgap=3,

                ygap=3,

                hovertemplate=(
                    "Theme: %{y}<br>"
                    "Outlet type: %{x}<br>"
                    "Share / articles: %{text}"
                    "<extra></extra>"
                )
            )
        )

        fig_outlet_theme.update_layout(
            height=500,
            autosize=True,
            xaxis_title="",
            yaxis_title="",
            margin=dict(
                l=10,
                r=20,
                t=40,
                b=20
            )
        )

        fig_outlet_theme.update_xaxes(
            side="top",
            tickangle=0,
            tickfont=dict(size=13),
            automargin=True
        )

        fig_outlet_theme.update_yaxes(
            categoryorder="array",
            categoryarray=THEME_ORDER,
            autorange="reversed",
            automargin=True
        )

        _, heatmap_col, _ = st.columns([1, 2, 1])
        with heatmap_col:
            st.plotly_chart(
                fig_outlet_theme,
                use_container_width=True,
                config={
                    "responsive": True
                }
            )


        # ----------------------------------------------------
        # CHART TITLE — BOTTOM, CENTERED, ITALIC
        # ----------------------------------------------------

        st.markdown(
            """
            <div style="
                font-size: 14px;
                font-weight: 400;
                font-style: italic;
                color: black;
                text-align: center;
                margin-top: -8px;
                margin-bottom: 10px;
            ">
                Theme Coverage by Outlet Type
            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # EXPLANATORY TEXT
        # ----------------------------------------------------

        st.markdown(
            """
            <div style="
                text-align: left;
                color: black;
                font-size: 14px;
                margin-bottom: 10px;
            ">
            Cell color and the percentage show the share of each outlet type's
            unique articles that touch a given theme, so outlet types with very
            different article volumes can be compared fairly. The number in
            parentheses is the actual article count, and <em>n</em> in each
            column heading is the outlet type's total number of articles.
            Percentages can add up to more than 100% within a column because
            one article may be tagged with more than one theme. Outlet types
            with few articles should be read cautiously.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.info(
            "No outlet-tagged, theme-classified articles are available "
            "yet to compare themes by outlet type."
        )

    st.markdown("---")

    # ========================================================
    # ARTICLES COLLECTED
    # ========================================================

    st.subheader(
        "Articles Collected by the Monitor"
    )

    st.markdown(
        """
        <div style="
            text-align: left;
            color: black;
        ">
        New articles are collected through Google and media outlets'
        RSS feeds six times daily.
        </div>
        """,
        unsafe_allow_html=True
    )

    article_columns = [
        "published_at",
        "source",
        "title",
        "themes",
        "stance",
        "url"
    ]

    available_article_columns = [
        column
        for column in article_columns
        if column in df.columns
    ]

    articles_display = (
        df[available_article_columns]
        .sort_values(
            "published_at",
            ascending=False
        )
    )

    st.dataframe(
        articles_display,
        use_container_width=True
    )

    # Space between the articles section and the footer
st.markdown(
    "<div style='height: 2.5rem;'></div>",
    unsafe_allow_html=True
)

st.markdown(
    """
<hr style="border: none; border-top: 1px solid #ddd; margin: 0 0 1.5rem 0;">

<div style="
    color: #333333;
    font-style: italic;
    font-size: 1rem;
    line-height: 1.5;
    margin-bottom: 1.5rem;
">
This report and app were developed as an Eskwelabs
Data Analytics Capstone Project. As a working prototype, its
classifications may contain errors and should be read as
indicative rather than definitive. This tool monitors and classifies
media coverage on Pax Silica; the content it surfaces does not reflect the
stance or opinion of the author. More information on the project can
be found <a href="YOUR_LINK" style="color: #333333;">here</a>.
</div>
""",
    unsafe_allow_html=True,
)
	
    
