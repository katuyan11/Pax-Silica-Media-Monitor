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

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pax Silica News Monitor",
    layout="wide"
)

st.title("Monitoring the Conversation: A News Monitoring Prototype to Track Media Coverage of the Pax Silica Initiative in the Philippines")

st.markdown("""
<div style="
    text-align: justify;
    margin-bottom: 30px;
">
This prototype monitors Philippine news coverage related to Pax Silica using automated news ingestion and rule-based Natural Language Processing (NLP), including keyword-based theme classification, stance detection, text preprocessing, and word-frequency analysis to identify dominant themes and stances. Coverage has been tracked since March 2026.
</div>
""", unsafe_allow_html=True)


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

    sheet = gc.open_by_key(
        st.secrets["GOOGLE_SHEET_ID"]
    ).worksheet("Clean_Data")

    records = sheet.get_all_records()

    df = pd.DataFrame(records)

    if not df.empty:

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
# Add important real-world developments here to provide context
# for changes in the bubble matrix over time.
#
# Format:
# {
#     "date": "YYYY-MM-DD",
#     "label": "Short event label",
#     "description": "Optional longer description shown on hover"
# }
#
# IMPORTANT: Replace the example entries below with verified
# dates/events relevant to your monitoring period. If no events
# are entered, the bubble matrix will display normally without
# event markers.

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
        "date": "2026-04-23",
        "label": "DND discusses Pax Silica",
        "description": (
            "Defense Secretary Gilberto Teodoro discussed Pax Silica in relation "
            "to Philippine resilience and industrialization."
        )
    },

    {
        "date": "2026-05-04",
        "label": "PH-Israel cooperation",
        "description": (
            "Philippines-Israel discussions on critical minerals processing and AI "
            "technology were linked to Pax Silica."
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
        "date": "2026-07-20",
        "label": "Safeguards concerns addressed",
        "description": (
            "The government addressed concerns involving national interests, "
            "environmental protection, water resources, and possible community "
            "displacement."
        )
    },

    {
        "date": "2026-07-27",
        "label": "Pax Silica highlighted in SONA",
        "description": (
            "The proposed Pax Silica Industrial Hub was highlighted during the "
            "President's 2026 State of the Nation Address as part of the Luzon "
            "Economic Corridor and the country's advanced manufacturing strategy."
        )
    },

    {
        "date": "2026-08-07",
        "label": "Development timeline announced",
        "description": (
            "BCDA and DTI officials provided further details on the proposed Pax "
            "Silica hub, including its development timeline and planned initial "
            "site development."
        )
    },

    {
        "date": "2026-08-10",
        "label": "BCDA clarifies project concerns",
        "description": (
            "BCDA publicly addressed misconceptions concerning the Pax Silica "
            "project's scale, data-center characterization, environmental impacts, "
            "and possible displacement."
        )
    },

    {
        "date": "2026-09-10",
        "label": "Luzon Economic Corridor forum",
        "description": (
            "The Luzon Economic Corridor Investment Forum generated further "
            "discussion of the proposed Pax Silica development and its investment "
            "implications."
        )
    }
]


# ============================================================
# EVENT CLASSIFICATION
# ============================================================
# International events are placed below the chart.
# All other configured events are treated as domestic events
# and are placed above the chart.

INTERNATIONAL_EVENT_LABELS = {
    "PH-Israel cooperation",
    "PH-UAE AI infrastructure talks"
}


def prepare_significant_events(events):
    """Convert configured event dates into a clean DataFrame."""

    if not events:

        return pd.DataFrame(
            columns=["date", "label", "description"]
        )

    events_df = pd.DataFrame(events)

    if "date" not in events_df.columns or "label" not in events_df.columns:

        return pd.DataFrame(
            columns=["date", "label", "description"]
        )

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

    return events_df[
        ["date", "label", "description", "event_type"]
    ].sort_values("date")


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
    texts: list[str],
    n: int = 100
) -> list[tuple[str, int]]:

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

        text_lower = text.lower()

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
            w
            for w in tokens
            if (
                w.isalpha()
                or "_" in w
            )
            and w not in stop_words
            and len(w) > 2
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
    # RESEARCH QUESTION 1
    # ========================================================

    st.markdown(
        """
        <div style="
            font-size: 22px;
            font-weight: 500;
            font-style: italic;
            margin-top: 10px;
            margin-bottom: 10px;
        ">
        Research Question 1: What themes are represented in Philippine media coverage of Pax Silica?
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
        <strong>Note:</strong> The categories were defined based on recurring topics and issues identified in the corpus and subsequently operationalized through keyword-based classification. The thematic categories were developed inductively from patterns observed in the collected news coverage.
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # THEME DESCRIPTIONS — TWO COLUMNS
    # ========================================================

    left_themes = THEME_ORDER[:4]
    right_themes = THEME_ORDER[4:]

    # Wider gap between the two theme-description columns
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
    # SPACE BETWEEN SECTIONS
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # ========================================================
    # RESEARCH QUESTION 2
    # ========================================================

    st.markdown(
        """
        <div style="
            font-size: 22px;
            font-weight: 500;
            font-style: italic;
            margin-top: 10px;
            margin-bottom: 10px;
        ">
        Research Question 2: How do the themes and stances represented in media coverage change over time as new developments emerge?
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # COMBINED BUBBLE MATRIX NOTE
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="
            color: black;
            text-align: justify;
            font-size: 14px;
            margin-bottom: 20px;
        ">
        <strong>Note:</strong> Each bubble shows how many articles tackled a given theme and stance on a specific date — bigger bubbles mean more articles, and the color shows whether the coverage leaned positive, negative, or neutral. Vertical dotted markers indicate significant developments that may help contextualize changes in media attention and stance over time.
        </div>
        """,
        unsafe_allow_html=True
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
    )

    bubble_df = bubble_df[
        bubble_df["stance"].isin(
            [
                "Supportive",
                "Neutral",
                "Critical"
            ]
        )
    ]


    # --------------------------------------------------------
    # Convert publication timestamp to date
    # --------------------------------------------------------

    bubble_df["date"] = pd.to_datetime(
        bubble_df["published_at"],
        errors="coerce"
    ).dt.date


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
    ]


    # --------------------------------------------------------
    # Remove rows without usable dates
    # --------------------------------------------------------

    bubble_df = bubble_df[
        bubble_df["date"].notna()
    ]


    # --------------------------------------------------------
    # FORCE THEME COLUMN INTO FIXED CATEGORY ORDER
    # --------------------------------------------------------

    bubble_df["themes"] = pd.Categorical(
        bubble_df["themes"],
        categories=THEME_ORDER,
        ordered=True
    )


    # --------------------------------------------------------
    # Use theme names only for Y-axis labels
    # --------------------------------------------------------

    theme_labels = {
        theme: theme
        for theme in THEME_ORDER
    }


    # --------------------------------------------------------
    # Complete list of theme labels
    # --------------------------------------------------------

    all_theme_labels = THEME_ORDER


    # --------------------------------------------------------
    # Prepare article information for hover
    # --------------------------------------------------------

    bubble_df["article_title"] = (
        bubble_df["title"]
        .fillna("Untitled article")
        .astype(str)
    )

    bubble_df["published_date_display"] = (
        pd.to_datetime(
            bubble_df["published_at"],
            errors="coerce"
        )
        .dt.strftime("%d %b %Y")
    )

    bubble_df["hover_detail"] = (
        "<strong>Article Title:</strong> "
        + bubble_df["article_title"]
        + "<br><strong>Date Published:</strong> "
        + bubble_df["published_date_display"]
    )


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
        .agg(
            article_count=("date", "size"),
            hover_detail=(
                "hover_detail",
                lambda x: "<br><br>".join(
                    x.astype(str)
                )
            )
        )
        .reset_index()
    )


    # --------------------------------------------------------
    # Create theme labels
    # --------------------------------------------------------

    bubble_data["theme_label"] = (
        bubble_data["themes"]
        .map(theme_labels)
    )


    # --------------------------------------------------------
    # Force theme labels into fixed categorical order
    # --------------------------------------------------------

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
    )


    # --------------------------------------------------------
    # Prepare significant events for the timeline
    # --------------------------------------------------------

    events_df = prepare_significant_events(
        SIGNIFICANT_EVENTS
    )


    # --------------------------------------------------------
    # Generate bubble matrix
    # --------------------------------------------------------

    if not bubble_data.empty:

        fig_bubble = px.scatter(
            bubble_data,

            x="date",

            y="theme_label",

            size="article_count",

            color="stance",

            size_max=45,

            custom_data=[
                "hover_detail"
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
        # Do not show theme, stance, article count, or other
        # default Plotly hover fields.

        fig_bubble.update_traces(
            hovertemplate="%{customdata[0]}<extra></extra>"
        )


        # ----------------------------------------------------
        # Explicitly force all seven themes onto Y-axis
        # ----------------------------------------------------

        fig_bubble.update_yaxes(
            categoryorder="array",
            categoryarray=all_theme_labels
        )


        # ----------------------------------------------------
        # Layout
        # ----------------------------------------------------

        fig_bubble.update_layout(
            height=650,
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


        # ----------------------------------------------------
        # Add significant-event markers
        # ----------------------------------------------------
        #
        # Domestic events are labelled above the chart.
        # International events are labelled below the chart
