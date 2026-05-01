import streamlit as st

# ---------------- THEME ----------------
def apply_theme():
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

    toggle = st.sidebar.toggle("🌗 Dark Mode", value=True)

    st.session_state.theme = "dark" if toggle else "light"

    if st.session_state.theme == "dark":
        bg = "#0E1117"
        card = "#161B22"
        text = "white"
    else:
        bg = "#FFFFFF"
        card = "#F5F5F5"
        text = "#000000"

    st.markdown(f"""
    <style>
    body {{
        background-color: {bg};
        color: {text};
    }}
    .metric-card {{
        background-color: {card};
        padding: 15px;
        border-radius: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)


# ---------------- TOOLTIP ----------------
def tooltip(label, tip):
    return f"{label}  \n\nℹ️ {tip}"

def metric_with_tooltip(label, value, help_text):
    return st.metric(label, value, help=help_text)