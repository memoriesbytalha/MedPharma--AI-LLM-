import streamlit as st
import base64
from pathlib import Path


BACKGROUND = Path(r"D:\FYP Try\Med-Pharma AI\images\1.jpg")


def set_page_background(png_file):
    @st.cache_data()
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    
    try:
        bin_str = get_base64_of_bin_file(png_file)
    except FileNotFoundError:
        bin_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z/C/HgAGgwJ/lK3Q6wAAAABJRU5ErkJggg=="
    
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 20, 0.7)), url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}
    .stTextInput input, .stTextArea textarea {{
        color: #333333 !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
    }}
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {{
        color: #666666 !important;
        opacity: 1 !important;
    }}
    .stSelectbox select {{
        color: #333333 !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
    }}
    body, h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {{
        color: white !important;
    }}
    [data-testid="stHeader"] {{
        background-color: rgba(0,0,0,0.5) !important;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


