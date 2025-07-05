import streamlit as st
import base64
from pathlib import Path
from PIL import Image
import json
from io import BytesIO
import streamlit.components.v1 as components

# -------------------
# Page Configuration
# -------------------
st.set_page_config(page_title="🏁 F1 Predictions", layout="wide")

# -------------------
# Custom CSS Styling
# -------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');

    html, body, [class*="css"] {
        font-family: 'Bebas Neue', sans-serif;
        background-color: #0f0f0f;
        color: white;
    }

    .hero-title {
        font-size: 60px;
        font-weight: 900;
        text-align: center;
    }

    .hero-title span {
        color: #ff1801;
    }

    .subtitle {
        font-size: 20px;
        text-align: center;
        margin-top: -15px;
        color: #d3d3d3;
    }

    .kpi-container {
        display: flex;
        justify-content: center;
        gap: 80px;
        margin-top: 40px;
        margin-bottom: 40px;
    }

    .kpi-box {
        text-align: center;
    }

    .kpi-value {
        font-size: 32px;
        font-weight: bold;
        margin: 5px 0;
    }

    .kpi-label {
        color: #bbbbbb;
        font-size: 14px;
    }

    .section-title {
        font-size: 38px;
        text-align: center;
        margin-top: 60px;
    }

    .section-title span {
        color: #ff1801;
    }

    .section-subtitle {
        text-align: center;
        color: #bbb;
        margin-bottom: 30px;
    }

    .card-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 30px;
        padding: 0 10%;
    }

    .insight-card {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        color: black;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
    }

    .insight-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(255, 24, 1, 0.3);
    }

    .insight-card img {
        width: 100%;
        border-radius: 10px;
    }

    .caption-truncated {
        color: #333;
        font-size: 16px;
        margin-top: 10px;
    }

    .footer {
        margin-top: 60px;
        text-align: center;
        color: #777;
        font-size: 13px;
    }

    .stButton > button {
        background-color: #ff1801;
        color: white;
        border-radius: 30px;
        padding: 10px 30px;
        font-weight: bold;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------
# Hero Section
# -------------------
st.markdown("""
<div class="hero-title">F1 <span>PREDICTIONS</span></div>
<div class="subtitle">Harness the power of data to predict Formula 1 race outcomes</div>
""", unsafe_allow_html=True)

# -------------------
# KPI Metrics
# -------------------
st.markdown("""
<div class="kpi-container">
    <div class="kpi-box">
        <div class="kpi-icon">🏆</div>
        <div class="kpi-value">23</div>
        <div class="kpi-label">Grand Prix Races</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-icon">⚡</div>
        <div class="kpi-value">89%</div>
        <div class="kpi-label">Prediction Accuracy</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-icon">🏎️</div>
        <div class="kpi-value">20</div>
        <div class="kpi-label">Active Drivers</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------
# Predict Button
# -------------------
if st.button("🔮 Predict the Winner", use_container_width=True):
    st.success("Prediction engine would run here...")

# -------------------
# Section: Race Insights
# -------------------
st.markdown('<div class="section-title">Race <span>Insights</span></div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Real-time analytics and predictive intelligence</div>', unsafe_allow_html=True)

# -------------------
# Utilities
# -------------------
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# -------------------
# Load Images + Captions
# -------------------
analytics_dir = Path("ANALYTICS")
caption_dict = {}
if Path("analytics_captions.json").exists():
    with open("analytics_captions.json", "r") as f:
        caption_dict = json.load(f)

# -------------------
# Render Cards via HTML
# -------------------
if analytics_dir.exists():
    image_files = sorted([f for f in analytics_dir.glob("*.png")])
    icons = ["🏁", "📈", "🏎️", "🌞"]

    card_html = '<div class="card-grid">'

    for i, image_file in enumerate(image_files):
        image = Image.open(image_file)
        base64_img = image_to_base64(image)
        full_caption = caption_dict.get(image_file.name, image_file.stem.replace("_", " ").title())
        short_caption = full_caption.split(".")[0].strip() + "." if "." in full_caption else full_caption

        card_html += f"""
            <div class="insight-card" title="{full_caption}">
                <div class="insight-icon">{icons[i % len(icons)]}</div>
                <img src="data:image/png;base64,{base64_img}" />
                <p class="caption-truncated">{short_caption}</p>
            </div>
        """

    card_html += "</div>"

    components.html(card_html, height=1000)
else:
    st.warning("ANALYTICS/ folder not found or empty.")

# -------------------
# Footer
# -------------------
st.markdown('<div class="footer">Made with ❤️ by <a href="https://github.com/ag826" style="color:#ff1801;">@ag826</a> | Data from Ergast API</div>', unsafe_allow_html=True)
