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
        margin-bottom: 30px;
    ">
    This prototype monitors Philippine news coverage related to Pax Silica
    using automated news ingestion and rule-based Natural Language Processing
    (NLP), including keyword-based theme classification, stance detection,
    text preprocessing, and word-frequency analysis to identify dominant
    themes and stances. Coverage has been tracked since March 2026.
    </div>
    """,
    unsafe_allow_html=True
)


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
# THEME ORDER
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
        that dominate Pax Silica coverage
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
        collected news coverage.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # THEME DESCRIPTIONS — TWO COLUMNS
    # ========================================================

    left_themes = THEME_ORDER[:4]
    right_themes = THEME_ORDER[4:]

    col1, spacer, col2 = st.columns(
        [1, 0.11, 1]
    )

    # --------------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------------

    with col1:

        for theme in left_themes:

            st.markdown(
                f"**{theme}**"
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

    # --------------------------------------------------------
    # SPACER
    # --------------------------------------------------------

    with spacer:

        st.markdown(
            "<div style='height: 1px;'></div>",
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------------

    with col2:

        for theme in right_themes:

            st.markdown(
                f"**{theme}**"
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
        Where media attention concentrates, and whether the tone
        leans supportive, critical, or balanced
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
                [
                    "Supportive",
                    "Neutral",
                    "Critical"
                ],
                fill_value=0
            )
            .reset_index()
        )

        stance_counts.columns = [
            "stance",
            "count"
        ]

        fig_stance = px.pie(
            stance_counts,
            names="stance",
            values="count",
            category_orders={
                "stance": [
                    "Supportive",
                    "Neutral",
                    "Critical"
                ]
            }
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
    # THEME × STANCE HEAT MAP
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
            The heat map shows the number of articles associated with each
            theme and detected stance across the entire monitoring period.
            Darker cells indicate a higher number of articles, while lighter
            cells indicate fewer articles. The total at the right shows the
            cumulative number of articles associated with each theme.
            </div>
            """,
            unsafe_allow_html=True
        )

        heatmap_df = df.copy()

        # ----------------------------------------------------
        # Normalize stance labels
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
                [
                    "Supportive",
                    "Neutral",
                    "Critical"
                ]
            )
        ].copy()

        # ----------------------------------------------------
        # Split multi-label themes
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

        # ----------------------------------------------------
        # Keep only defined themes
        # ----------------------------------------------------

        heatmap_df = heatmap_df[
            heatmap_df["themes"].isin(
                THEME_ORDER
            )
        ].copy()

        # ----------------------------------------------------
        # Reset index
        # ----------------------------------------------------

        heatmap_df = heatmap_df.reset_index(
            drop=True
        )

        # ----------------------------------------------------
        # Count articles
        # ----------------------------------------------------

        heatmap_data = (
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

        # ----------------------------------------------------
        # Force all themes and stances
        # ----------------------------------------------------

        heatmap_data = heatmap_data.reindex(
            index=THEME_ORDER,
            columns=[
                "Supportive",
                "Neutral",
                "Critical"
            ],
            fill_value=0
        )

        # ----------------------------------------------------
        # Calculate total
        # ----------------------------------------------------

        heatmap_data["Total"] = (
            heatmap_data[
                [
                    "Supportive",
                    "Neutral",
                    "Critical"
                ]
            ].sum(axis=1)
        )

        # ----------------------------------------------------
        # Order themes by total
        # ----------------------------------------------------

        heatmap_theme_order = (
            heatmap_data["Total"]
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        heatmap_data = heatmap_data.reindex(
            heatmap_theme_order
        )

        # ----------------------------------------------------
        # Generate heat map
        # ----------------------------------------------------

        fig_heatmap = px.imshow(
            heatmap_data,
            text_auto=True,
            aspect="auto",
            labels={
                "x": "Stance",
                "y": "Theme",
                "color": "Articles"
            }
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
    # BUILD THEME COUNTS
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

    # ========================================================
    # TREND ANALYSIS
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
            [
                "Supportive",
                "Neutral",
                "Critical"
            ],
            fill_value=0
        )
    )

    if stance_counts.sum() > 0:

        dominant_stance = stance_counts.idxmax()

        dominant_stance_count = (
            stance_counts[dominant_stance]
        )

        dominant_stance_share = (
            dominant_stance_count
            / stance_counts.sum()
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
        Tracking shifts in attention and tone as real-world events unfold
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # MONTHLY SUMMARY
    # ========================================================

    monthly_df = df.copy()

    # --------------------------------------------------------
    # Prepare dates
    # --------------------------------------------------------

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
        # Create unique article ID
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

        # ----------------------------------------------------
        # Keep one record per article
        # ----------------------------------------------------

        monthly_df = monthly_df.drop_duplicates(
            subset="article_id"
        )

        # ----------------------------------------------------
        # Available months
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
        # Monthly article count
        # ----------------------------------------------------

        monthly_article_count = (
            selected_month_df["article_id"]
            .nunique()
        )

        # ----------------------------------------------------
        # Monthly theme counts
        # ----------------------------------------------------

        monthly_theme_df = (
            selected_month_df.copy()
        )

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
        # Monthly stance
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
                [
                    "Supportive",
                    "Neutral",
                    "Critical"
                ],
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
        # Significant events
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
        # Identify emerging theme
        # ----------------------------------------------------

        if not leading_monthly_themes:

            monthly_theme_text = (
                "No clearly dominant theme was detected in the "
                "available coverage."
            )

        elif len(leading_monthly_themes) == 1:

            theme, count = (
                leading_monthly_themes[0]
            )

            monthly_theme_text = (
                f"{theme} was the most prominent theme, appearing "
                f"in {count} article"
                f"{'s' if count != 1 else ''}."
            )

        else:

            theme_1, count_1 = (
                leading_monthly_themes[0]
            )

            theme_2, count_2 = (
                leading_monthly_themes[1]
            )

            monthly_theme_text = (
                f"{theme_1} led the coverage with "
                f"{count_1} article"
                f"{'s' if count_1 != 1 else ''}, followed by "
                f"{theme_2} with "
                f"{count_2} article"
                f"{'s' if count_2 != 1 else ''}."
            )

        # ----------------------------------------------------
        # Event context
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
        # Build monthly summary
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
        # Display monthly summary
        # ----------------------------------------------------

        st.markdown(
            f"""
            <h3 style="margin-bottom: 0.2rem;">
                {selected_month_name} — What is the emerging theme?
            </h3>
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
    # Clean stance labels
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
        bubble_df["stance"].isin(
            [
                "Supportive",
                "Neutral",
                "Critical"
            ]
        )
    ].copy()

    # --------------------------------------------------------
    # Convert publication timestamp to datetime
    # --------------------------------------------------------

    bubble_df["date"] = pd.to_datetime(
        bubble_df["published_at"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Prepare article title
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
    # Prepare publication date for hover
    # --------------------------------------------------------

    bubble_df["published_date_display"] = (
        bubble_df["date"]
        .dt.strftime("%d %b %Y")
        .fillna("Unknown date")
    )

    # --------------------------------------------------------
    # Hover text
    # --------------------------------------------------------
    # Show ONLY article title and publication date.

    bubble_df["article_hover"] = (
        "Article Title: "
        + bubble_df["article_title"]
        + "<br>Date Published: "
        + bubble_df["published_date_display"]
    )

    # --------------------------------------------------------
    # Split multi-label themes
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
    # Keep only defined themes
    # --------------------------------------------------------

    bubble_df = bubble_df[
        bubble_df["themes"].isin(
            THEME_ORDER
        )
    ].copy()

    # --------------------------------------------------------
    # Remove rows without usable dates
    # --------------------------------------------------------

    bubble_df = bubble_df[
        bubble_df["date"].notna()
    ].copy()

    # --------------------------------------------------------
    # Force theme column into fixed category order
    # --------------------------------------------------------

    bubble_df["themes"] = pd.Categorical(
        bubble_df["themes"],
        categories=THEME_ORDER,
        ordered=True
    )

    # --------------------------------------------------------
    # Theme labels
    # --------------------------------------------------------

    all_theme_labels = THEME_ORDER.copy()

    theme_labels = {
        theme: theme
        for theme in THEME_ORDER
    }

    # --------------------------------------------------------
    # Aggregate articles by:
    # date + theme + stance
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
    # Prepare hover information
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
    # Create theme labels
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
    # Sort data
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
    # Prepare significant events
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

                "stance": [
                    "Supportive",
                    "Neutral",
                    "Critical"
                ]
            },

            labels={
                "date": "Publication Date",
                "theme_label": "Theme",
                "stance": "Stance",
                "article_count": "Articles"
            }
        )

        # ----------------------------------------------------
        # Hover information
        # ----------------------------------------------------
        # Show only article title and date published.

        fig_bubble.update_traces(
            hovertemplate=(
                "%{customdata[0]}"
                "<extra></extra>"
            )
        )

        # ----------------------------------------------------
        # Explicitly force all seven themes onto Y-axis
        # ----------------------------------------------------

        fig_bubble.update_yaxes(
            categoryorder="array",
            categoryarray=all_theme_labels,
            autorange="reversed",
            automargin=True
        )

        # ----------------------------------------------------
        # Base chart layout
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
        # Add significant-event markers
        # ----------------------------------------------------
        #
        # Domestic events are labelled above the chart.
        # International events are labelled below the chart.

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
            # EVENT LABEL PLACEMENT
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
            # Label positioning settings
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
            # Estimate label width
            # -------------------------------------------------

            def estimate_label_width(label):

                return max(
                    4,
                    len(str(label)) * 0.42
                )

            # -------------------------------------------------
            # Assign labels to rows
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
                    # Try each row
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
                    # If no row is free, choose row with
                    # greatest available spacing.
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
                    # Store assignment
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
            # ADD EVENT MARKERS AND LABELS
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

            # -------------------------------------------------
            # Add each event to chart
            # -------------------------------------------------

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
                # Vertical event marker
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
                # Event label
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
                ).strip()

                if description:

                    annotation_kwargs[
                        "hovertext"
                    ] = description

                    annotation_kwargs[
                        "hoverlabel"
                    ] = dict(
                        bgcolor="white"
                    )

                fig_bubble.add_annotation(
                    **annotation_kwargs
                )

            # -------------------------------------------------
            # Give event labels enough space
            # -------------------------------------------------

            fig_bubble.update_layout(
                margin=dict(
                    l=10,
                    r=20,
                    t=175,
                    b=155
                )
            )

        # ====================================================
        # DISPLAY BUBBLE MATRIX
        # ====================================================

        st.plotly_chart(
            fig_bubble,
            use_container_width=True,
            config={
                "responsive": True
            }
        )

        # ====================================================
        # SIGNIFICANT EVENTS TABLE
        # ====================================================

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

                event_display = (
                    visible_events.copy()
                )

                event_display["date"] = (
                    event_display["date"]
                    .dt.strftime("%d %b %Y")
                )

                event_display = (
                    event_display.rename(
                        columns={
                            "date": "Date",
                            "label": "Event",
                            "description": "Description"
                        }
                    )
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

        st.info(
            "Not enough dated theme data available to generate "
            "the bubble matrix."
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
