import streamlit as st

st.set_page_config(page_title="Deepfake Image Detector", layout="centered")

from src.ui.theme import inject_css  # custom styling

inject_css()  # load css styles

# setup pages navigation
navigation = st.navigation(
    [
        st.Page("pages/analyze_image.py", title="Analyze Image", default=True),
        st.Page("pages/history.py", title="History"),
    ],
    position="top",
)
navigation.run()
