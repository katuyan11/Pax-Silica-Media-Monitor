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

st.set_page_config(
    page_title="Pax Silica News Monitor",
    layout="wide"
)

st.title(
    "Monitoring the Conversation: A News Monitoring Prototype "
    "to Track Media Coverage of the Pax Silica Initiative in the Philippines"
)

st.markdown(
    """
    <div style="
        text-align: justify;
        margin-bottom: 20px;
    ">
    This prototype monitors news coverage related to Pax Silica initiative in the 
    Philippines using automated news ingestion and rule-based Natural Language Processing
    (NLP), including keyword-based theme classification, stance detection,
    text preprocessing, and word-frequency analysis to identify dominant
    themes and stances. Coverage has been tracked since March 2026.
    </div>
    """,
    unsafe_allow_html=True
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

if st.button("Refresh Data"):

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

    overview_cols = st.columns(3)

    with overview_cols[0]:

        st.metric(
            "Articles tracked",
            f"{overview_article_count:,}"
        )

    with overview_cols[1]:

        st.metric(
            "Coverage period",
            date_range_text
        )

    with overview_cols[2]:

        # [FIX] Now counts unique values from "source" — the same
        # column the Articles Collected table displays — rather than
        # "outlet", which isn't a column in this sheet and was
        # silently falling back to "N/A".
        if "source" in df.columns:

            st.metric(
                "Outlets monitored",
                f"{df['source'].nunique():,}"
            )

        else:

            st.metric(
                "Outlets monitored",
                "N/A"
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
            text-align: justify;
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
                text-align: justify;
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
    # [FIX 3] DETECTED ARTICLE-LEVEL STANCE — replaced the pie
    # chart with a labeled 100%-width stacked bar. Precise counts
    # and percentages both show, and it reads correctly even for
    # colorblind viewers since each segment is text-labeled.
    # ========================================================

    with col1:

        st.markdown(
            """
            <div style="
                font-size: 16px;
                font-weight: 400;
                color: black;
                margin-bottom: 8px;
            ">
            Detected Article-Level Stance
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="
                text-align: justify;
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
            if stance_total > 0 else 0
        )

        stance_counts["segment_label"] = stance_counts.apply(
            lambda row: (
                f"{row['stance']}<br>{int(row['count'])} "
                f"({row['percent']:.1f}%)"
            ),
            axis=1
        )

        stance_counts["row"] = "All coverage"

        fig_stance = px.bar(
            stance_counts,
            x="count",
            y="row",
            color="stance",
            orientation="h",
            text="segment_label",
            category_orders={"stance": STANCE_ORDER},
            color_discrete_map=STANCE_COLORS
        )

        fig_stance.update_traces(
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=12, color="white"),
            marker_line_width=0
        )

        fig_stance.update_layout(
            barmode="stack",
            height=220,
            showlegend=True,
            legend_title_text="Stance",
            xaxis_title="Number of articles",
            yaxis_title="",
            yaxis=dict(showticklabels=False),
            margin=dict(l=10, r=10, t=20, b=40)
        )

        st.plotly_chart(
            fig_stance,
            use_container_width=True,
            config={
                "responsive": True
            }
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
    # [FIX 1] THEME × STANCE HEAT MAP — color now reflects each
    # theme's *share* of Supportive/Neutral/Critical coverage
    # (row-normalized %), not raw counts, so a small theme with a
    # skewed stance mix is no longer washed out by a large theme
    # with big absolute numbers. The "Total" column is shown with
    # its own count but excluded from the color scale, since
    # totals aren't a stance and were previously distorting the
    # whole map's color range. A colorbar legend is included.
    # ========================================================

    with col2:

        st.markdown(
            """
            <div style="
                font-size: 16px;
                font-weight: 400;
                color: black;
                margin-bottom: 8px;
            ">
            Theme × Stance Heat Map
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="
                text-align: justify;
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

        heatmap_df = df.copy()

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
            heatmap_df["stance"].isin(STANCE_ORDER)
        ].copy()

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

        heatmap_counts["Total"] = heatmap_counts[STANCE_ORDER].sum(axis=1)

        # Order rows by total volume, most-covered theme on top.
        heatmap_theme_order = (
            heatmap_counts["Total"]
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        heatmap_counts = heatmap_counts.reindex(heatmap_theme_order)

        # Row-normalized percentages, used for color only.
        row_totals = heatmap_counts["Total"].replace(0, np.nan)

        heatmap_pct = (
            heatmap_counts[STANCE_ORDER]
            .div(row_totals, axis=0)
            .fillna(0) * 100
        )

        # Build the z (color) matrix and text (label) matrix,
        # including an uncolored Total column at the right.
        display_columns = STANCE_ORDER + ["Total"]

        z_matrix = heatmap_pct.copy()
        z_matrix["Total"] = np.nan  # NaN cells render uncolored

        text_matrix = heatmap_counts[display_columns]

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=z_matrix[display_columns].values,
                x=display_columns,
                y=heatmap_theme_order,
                text=text_matrix.values,
                texttemplate="%{text}",
                textfont=dict(size=12),
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
                    "Articles: %{text}<extra></extra>"
                )
            )
        )

        fig_heatmap.update_layout(
            height=500,
            xaxis_title="Stance",
            yaxis_title="Theme",
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


    # ========================================================
    # TREND ANALYSIS
    # (theme_counts / themes_df now come from the early block
    # above — no longer recomputed here)
    # ========================================================

    ranked_themes = sorted(
        theme_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    leading_themes = [
        item
        for item in ranked_themes
        if item[1] > 0
    ]

    stance_counts_summary = (
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
    )

    if stance_counts_summary.sum() > 0:

        dominant_stance = stance_counts_summary.idxmax()

        dominant_stance_count = (
            stance_counts_summary[dominant_stance]
        )

        dominant_stance_share = (
            dominant_stance_count
            / stance_counts_summary.sum()
            * 100
        )

    else:

        dominant_stance = "Neutral"
        dominant_stance_share = 0

    stance_summary = (
        f"{dominant_stance.lower()} coverage accounts for "
        f"{dominant_stance_share:.1f}% of articles"
    )

    trend_blurb = stance_summary


    # ========================================================
    # COVERAGE SUMMARY
    # ========================================================

    st.markdown(
        f"""
        <div style="
            text-align: justify;
            color: black;
            font-size: 14px;
            margin-top: 10px;
            margin-bottom: 10px;
        ">
        <strong>Coverage Summary:</strong> {trend_blurb}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<hr style='border: none; border-top: 1px solid #ddd; margin: 8px 0 20px 0;'>",
        unsafe_allow_html=True
    )


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
            month: month.strftime("%B %Y")
            for month in available_months
        }

        selected_month = st.selectbox(
            "Select month:",
            options=available_months,
            format_func=lambda month:
                month_labels[month]
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
    )


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
            title="News Coverage Over Time",
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

    else:

        st.info(
            "No dated, theme-tagged articles are available yet to "
            "plot on the coverage-over-time chart."
        )


    # ========================================================
    # SIGNIFICANT EVENTS TABLE
    # ========================================================

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
        4. How do domestic and international outlets differ in their coverage of Pax Silica?
        </div>

        <div style="
            font-size: 15px;
            font-style: italic;
            font-weight: 400;
            color: black;
            margin-bottom: 10px;
        ">
        If I only monitor from certain outlets, what perspective might I be missing?
        </div>
        """,
        unsafe_allow_html=True
    )



    # ========================================================
    # ARTICLES COLLECTED
    # ========================================================

    st.subheader(
        "Articles Collected by the Monitor"
    )

    st.markdown(
        """
        <div style="
            text-align: justify;
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
