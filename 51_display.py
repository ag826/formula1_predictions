import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import base64
import os
import json
from io import BytesIO

# ---------------------
# Page Config
# ---------------------
st.set_page_config(page_title="🏁 Formula 1 Predictions", layout="wide")

# ---------------------
# Custom F1-Themed Styling
# ---------------------
F1_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');

html, body, [class*="css"] {
    font-family: 'Bebas Neue', sans-serif !important;
    background-color: #0f0f0f;
    color: #ffffff;
}

h1, h2, h3, h4 {
    color: #ff1801;
}

.stButton>button {
    color: #0f0f0f !important;
    background-color: #ff1801 !important;
    border-radius: 8px;
}

.sidebar .sidebar-content {
    background-color: #1c1c1c;
}
</style>
"""
st.markdown(F1_STYLE, unsafe_allow_html=True)

# ---------------------
# Header
# ---------------------
st.title("🏎️ Formula 1 Race Predictions")
st.markdown("""
Welcome to the **Formula 1 Predictions Dashboard** – your pit wall for race forecasting!  
🔮 Powered by machine learning. Styled like the paddock.  
[View GitHub Repo](https://github.com/ag826/formula1_predictions)
""")

# ---------------------
# Image Display Utilities
# ---------------------
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Load Gemini-generated captions
caption_dict = {}
if os.path.exists("analytics_captions.json"):
    with open("analytics_captions.json", "r") as f:
        caption_dict = json.load(f)

# ---------------------
# Display ANALYTICS Images (Compact Grid Layout)
# ---------------------
st.markdown("## 📸 Analytics Visuals")

analytics_dir = Path("ANALYTICS")

if analytics_dir.exists():
    image_files = sorted([f for f in analytics_dir.glob("*.png") if f.is_file()])
    if image_files:
        num_cols = 3  # Increase number of columns to reduce whitespace
        cols = st.columns(num_cols)

        for idx, image_file in enumerate(image_files):
            image = Image.open(image_file)
            caption = caption_dict.get(image_file.name, f"📊 {image_file.stem.replace('_', ' ').title()}")
            with cols[idx % num_cols]:
                st.image(image, use_column_width=True, caption=caption)
    else:
        st.info("No PNG images found in the ANALYTICS/ folder.")
else:
    st.info("ANALYTICS/ directory not found.")

# ---------------------
# Footer
# ---------------------
st.markdown("---")
st.caption("Made with ❤️ by [@ag826](https://github.com/ag826) | Data via Ergast API | Styled like the F1 grid.")
