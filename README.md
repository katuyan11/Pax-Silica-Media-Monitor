# Pax Silica Media Monitor NLP
This tool is designed for journalists and researchers who have no access to commercial media intelligence platforms,
It's a transparent, topic-specific monitor built to answer defined research questions about one policy debate, with a taxonomy designed for Philippine policy coverage and a method that can be audited end to end.
**Detailed methodology**
For each article's headline and summary, I filter for relevance, clean and lowercase the text, decode Google News links, and remove duplicates. Then I match the text against seven theme dictionaries, and I score stance by counting supportive versus critical terms. If they tie, it's Neutral.
Labels come from the keyword classifier on each article's headline and summary. The classification rules were reviewed iteratively and manually corrected misclassifications, and refined the keyword lists as errors became less frequent.

For tools, I used Python for the pipeline, pandas for data wrangling, GitHub Actions to run it on a schedule, Google Sheets as the data store, and Streamlit for the live dashboard.

For themes, each article is checked against keyword dictionaries for seven themes, from Economic Development to Geopolitical Security. An article can match more than one. The lists include official terms like "bilateral agreement" and civil-society terms like "moratorium”, among others.

For stance, I use custom supportive and critical word lists. The stronger signal wins, and a tie or no clear signal is Neutral.
