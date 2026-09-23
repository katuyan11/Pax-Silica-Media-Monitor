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
    page_title="Pax Silica NLP News Monitor",
    layout="wide"
)

st.title("Pax Silica NLP News Monitor")

st.markdown("""
<div style="
    text-align: justify;
">
This prototype monitors Philippine news coverage related to Pax Silica using automated news ingestion and rule-based Natural Language Processing (NLP), including keyword-based theme classification, stance detection, text preprocessing, and word-frequency analysis to identify dominant themes and stances. Coverage has been tracked daily since September 17, 2026, using Python and Streamlit.
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
        [1, 0.15, 1]
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
        <strong>Note:</strong> Each bubble shows how many articles tackled a given theme and stance on a specific date — bigger bubbles mean more articles, and the color shows whether the coverage leaned positive, negative, or neutral. The number next to each theme's name is its total article count across the whole monitoring period. Since one article can touch on multiple themes, these totals will add up to more than the overall article count.
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
    # Calculate total articles per theme
    # across the entire monitoring period
    # --------------------------------------------------------

    theme_totals = (
        bubble_df
        .groupby(
            "themes",
            observed=False
        )
        .size()
        .reindex(
            THEME_ORDER,
            fill_value=0
        )
    )


    # --------------------------------------------------------
    # Create Y-axis labels with theme totals
    # --------------------------------------------------------

    theme_labels = {
        theme: f"{theme} ({theme_totals[theme]})"
        for theme in THEME_ORDER
    }


    # --------------------------------------------------------
    # Complete list of theme labels
    # --------------------------------------------------------

    all_theme_labels = [
        theme_labels[theme]
        for theme in THEME_ORDER
    ]


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
    # Add total article count for each theme
    # --------------------------------------------------------

    bubble_data["theme_total"] = (
        bubble_data["themes"]
        .map(theme_totals)
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

            hover_name="theme_label",

            hover_data={
                "date": True,
                "theme_label": False,
                "stance": True,
                "article_count": True,
                "theme_total": True
            },

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
                "article_count": "Articles",
                "theme_total": "Theme Total"
            }
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
            height=600,
            xaxis_title="Publication Date",
            yaxis_title="Theme",
            legend_title="Stance",
            hovermode="closest"
        )


        # ----------------------------------------------------
        # Display chart
        # ----------------------------------------------------

        st.plotly_chart(
            fig_bubble,
            use_container_width=True
        )


    else:

        st.info(
            "Not enough dated theme data available to generate "
            "the bubble matrix."
        )


    # ========================================================
    # SPACE BETWEEN SECTIONS
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # ========================================================
    # DETECTED ARTICLE-LEVEL STANCE + WORD CLOUD
    # ========================================================

    col1, col2 = st.columns(2)


    # ========================================================
    # DETECTED ARTICLE-LEVEL STANCE
    # ========================================================

    with col1:

        st.subheader(
            "Detected Article-Level Stance"
        )

        st.markdown(
            """
            <div style="
                text-align: justify;
                color: black;
                font-size: 14px;
                margin-bottom: 10px;
            ">
            Stance is estimated using predefined words and phrases associated with supportive or critical language in the available article text. The classifier counts these indicators and assigns the stance based on the stronger signal. Articles without a clear predominance of either signal are classified as Neutral. This is a rule-based classification and should be interpreted as a detected linguistic signal rather than a definitive statement of the article's or author's position.
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
            .value_counts()
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
            use_container_width=True
        )


    # ========================================================
    # WORD CLOUD
    # ========================================================

    with col2:

        st.subheader(
            "Frequently Mentioned Words in Coverage"
        )

        st.markdown(
            """
            <div style="
                text-align: justify;
                color: black;
                font-size: 14px;
                margin-bottom: 10px;
            ">
            The word cloud presents the most frequently mentioned words and terms across the entire news corpus collected by the monitor. The results are cumulative, covering the period from the start of monitoring on September 17, 2026, to the present.
            </div>
            """,
            unsafe_allow_html=True
        )

        combined_texts = (
            df["title"].fillna("")
            + " "
            + df["description"].fillna("")
        ).tolist()

        top_terms = get_top_terms(
            combined_texts,
            n=100
        )

        word_frequencies = dict(
            top_terms
        )

        if word_frequencies:

            wordcloud = WordCloud(
                width=900,
                height=500,
                background_color="white",
                max_words=60,
                min_font_size=10,
                max_font_size=70,
                collocations=False
            ).generate_from_frequencies(
                word_frequencies
            )

            fig_wordcloud, ax = plt.subplots(
                figsize=(10, 5)
            )

            ax.imshow(
                wordcloud,
                interpolation="bilinear"
            )

            ax.axis("off")

            st.pyplot(
                fig_wordcloud,
                use_container_width=True
            )

            plt.close(
                fig_wordcloud
            )

        else:

            st.info(
                "Not enough text available to generate "
                "a word cloud."
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
        New articles are automatically collected through RSS feeds six times daily at 7:00 AM, 10:00 AM, 1:00 PM, 4:00 PM, 7:00 PM, and 10:00 PM Philippine time. World News API is additionally queried at 7:00 AM and 7:00 PM.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(
        df[
            [
                "published_at",
                "source",
                "title",
                "themes",
                "stance",
                "url"
            ]
        ].sort_values(
            "published_at",
            ascending=False
        ),
        use_container_width=True
    )
