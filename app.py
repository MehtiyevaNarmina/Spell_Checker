import streamlit as st
from utils import spell_check, DICTIONARY

st.title("Spell Checker")

metrics = st.sidebar.multiselect(
    label="Metric",
    options=["levenshtein", "damerau_levenshtein", "jaro_similarity", "jaro_winkler"],
    default=["levenshtein"]
)

max_suggestions = st.sidebar.number_input(
    label="Maximum number of suggestions",
    min_value=1,
    value=3,
    placeholder="Type a number..."
)

word = st.text_input(
    label="Word to check",
    placeholder="Input your word here..."
)

submit_button = st.button(
    label="Submit"
)

if submit_button:

    if word in DICTIONARY:
        st.markdown(f"'{word}' is correct!")

    for metric in metrics:

        suggestions = spell_check(word.lower(), DICTIONARY, method=metric, max_suggestions=max_suggestions)
        metric_name = metric.replace("_", " ").title()

        if suggestions:
            st.markdown(f"Metric: {metric.upper()} → Suggestions: {', '.join(suggestions)}")
        else:
            st.markdown("No suggestions found.")
