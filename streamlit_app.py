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
This prototype monitors Philippine news coverage related to Pax Silica using automated news ingestion and rule-based Natural Language Processing (NLP), including keyword-based theme classification, stance detection, text preprocessing, and word-frequency analysis to identify dominant themes and stances. Coverage has been tracked daily since September 17, 2026, using Python and Streamlit.
""")


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
    "Human-Capital Upgrading",
    "Supply-Chain Resilience",
    "Environmental Sustainability",
    "Institutional Governance",
    "Geopolitical Security"
]


# ============================================================
# THEME DESCRIPTIONS
# ============================================================

THEME_DESCRIPTIONS = {
    "Economic Development":
        "Investment, industrial modernization, and high-value manufacturing.",

    "Technological Advancement":
        "Semiconductors, AI infrastructure, chip design, and technology transfer.",

    "Human-Capital Upgrading":
        "Workforce preparation, specialized skills, and technical employment.",

    "Supply-Chain Resilience":
        "Diversifying markets and strengthening regional critical-mineral and chip supply chains.",

    "Environmental Sustainability":
        "Energy and water use, land conversion, mining, and ecological impacts.",

    "Institutional Governance":
        "Regulation, transparency, accountability, and civil society or opposition responses.",

    "Geopolitical Security":
        "Strategic alignment, sovereignty, and links between industrial and military frameworks."
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
    # THEME × STANCE BUBBLE MATRIX + DESCRIPTIONS
    # ========================================================

    chart_col, description_col = st.columns(
        [2.3, 1]
    )


    # ========================================================
    # BUBBLE MATRIX
    # ========================================================

    with chart_col:

        st.subheader(
            "Theme × Stance Over Time"
        )

        st.caption(
            "Bubble size represents the number of articles. "
            "Bubble color represents stance. Themes are fixed as "
            "categorical rows; vertical position does not indicate "
            "rank or importance."
        )

        bubble_df = df.copy()

        # ----------------------------------------------------
        # Convert publication timestamp to date
        # ----------------------------------------------------

        bubble_df["date"] = pd.to_datetime(
            bubble_df["published_at"],
            errors="coerce"
        ).dt.date

        # ----------------------------------------------------
        # Split multi-label themes
        # ----------------------------------------------------

        bubble_df["themes"] = (
            bubble_df["themes"]
            .fillna("")
            .astype(str)
            .str.split(", ")
        )

        bubble_df = bubble_df.explode(
            "themes"
        )

        # ----------------------------------------------------
        # Keep only defined themes
        # ----------------------------------------------------

        bubble_df = bubble_df[
            bubble_df["themes"].isin(
                THEME_ORDER
            )
        ]

        # ----------------------------------------------------
        # Remove rows without usable dates
        # ----------------------------------------------------

        bubble_df = bubble_df[
            bubble_df["date"].notna()
        ]

        # ----------------------------------------------------
        # Aggregate articles by:
        # date + theme + stance
        # ----------------------------------------------------

        bubble_data = (
            bubble_df
            .groupby(
                [
                    "date",
                    "themes",
                    "stance"
                ]
            )
            .size()
            .reset_index(
                name="article_count"
            )
        )

        # ----------------------------------------------------
        # Force themes into fixed categorical order
        # ----------------------------------------------------

        bubble_data["themes"] = pd.Categorical(
            bubble_data["themes"],
            categories=THEME_ORDER,
            ordered=True
        )

        bubble_data = bubble_data.sort_values(
            [
                "themes",
                "date"
            ]
        )

        # ----------------------------------------------------
        # Generate bubble matrix
        # ----------------------------------------------------

        if not bubble_data.empty:

            fig_bubble = px.scatter(
                bubble_data,
                x="date",
                y="themes",
                size="article_count",
                color="stance",
                size_max=45,

                hover_name="themes",

                hover_data={
                    "date": True,
                    "themes": False,
                    "stance": True,
                    "article_count": True
                },

                category_orders={
                    "themes": THEME_ORDER,
                    "stance": [
                        "Supportive",
                        "Neutral",
                        "Critical"
                    ]
                },

                labels={
                    "date": "Publication Date",
                    "themes": "Theme",
                    "stance": "Stance",
                    "article_count": "Articles"
                }
            )

            # ------------------------------------------------
            # Improve hover labels
            # ------------------------------------------------

            fig_bubble.update_traces(
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Date: %{x|%b %d, %Y}<br>"
                    "Stance: %{marker.color}<br>"
                    "Articles: %{marker.size}"
                    "<extra></extra>"
                )
            )

            # ------------------------------------------------
            # Keep theme rows fixed
            # ------------------------------------------------

            fig_bubble.update_yaxes(
                categoryorder="array",
                categoryarray=THEME_ORDER
            )

            # ------------------------------------------------
            # Layout
            # ------------------------------------------------

            fig_bubble.update_layout(
                height=600,
                xaxis_title="Publication Date",
                yaxis_title="Theme",
                legend_title="Stance",
                hovermode="closest"
            )

            st.plotly_chart(
                fig_bubble,
                use_container_width=True
            )

            st.caption(
                "Each bubble represents the number of articles "
                "associated with a theme and stance on a given date. "
                "An article may contribute to more than one theme."
            )

        else:

            st.info(
                "Not enough dated theme data available to generate "
                "the bubble matrix."
            )


    # ========================================================
    # THEME DESCRIPTIONS
    # ========================================================

    with description_col:

        st.subheader(
            "Theme Descriptions"
        )

        for theme in THEME_ORDER:

            st.markdown(
                f"**{theme}**"
            )

            st.markdown(
                THEME_DESCRIPTIONS[theme]
            )


    # ========================================================
    # ARTICLE STANCE + WORD CLOUD
    # ========================================================

    col1, col2 = st.columns(2)


    # ========================================================
    # ARTICLE STANCE
    # ========================================================

    with col1:

        st.subheader(
            "Detected Article-Level Stance"
        )
        st.caption(
            "Stance reflects the predominant stance detected by the rule-based classifier in the available article text.
            It should not be interpreted as a definitive assessment of the article's overall position."
        )

        bubble_df = df.copy()

        stance_counts = (
            df["stance"]
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
            values="count"
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
    # ARTICLES
    # ========================================================

    st.subheader(
        "Articles"
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
