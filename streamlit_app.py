import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px

st.set_page_config(page_title="Pax Silica NLP News Monitor", layout="wide")
st.title("Pax Silica NLP News Monitor")
st.markdown("""
Track dominant themes and narratives in news coverage of Pax Silica using 
NLP techniques, Python, and Streamlit — updated daily.
""")

SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

@st.cache_data(ttl=3600)
def load_data():
    creds_dict = st.secrets["google_service_account"]  # set via Streamlit Cloud secrets, see below
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(st.secrets["GOOGLE_SHEET_ID"]).sheet1
    records = sheet.get_all_records()
    df = pd.DataFrame(records)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    return df

df = load_data()

if df.empty:
    st.info("No data yet. Check back after the next daily fetch runs.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Theme Frequency")
        theme_series = df["themes"].str.split(", ").explode()
        theme_counts = theme_series.value_counts().reset_index()
        theme_counts.columns = ["theme", "count"]
        st.plotly_chart(px.bar(theme_counts, x="theme", y="count"), use_container_width=True)

    with col2:
        st.subheader("Stance Distribution")
        stance_counts = df["stance"].value_counts().reset_index()
        stance_counts.columns = ["stance", "count"]
        st.plotly_chart(px.pie(stance_counts, names="stance", values="count"), use_container_width=True)

    st.subheader("Articles")
    st.dataframe(
        df[["published_at", "source", "title", "themes", "stance", "url"]].sort_values(
            "published_at", ascending=False
        ),
        use_container_width=True
    )
