from PIL import Image
import streamlit as st

def set_page_config():
    page_icon = Image.open("assets/logo.png")
    st.set_page_config(layout="centered", page_title="Click Analyst", page_icon=page_icon)
