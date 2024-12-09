# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time
from PIL import Image
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
import os

from utils import new_line
from config import set_page_config
from session_state import initial_state
from data_loading import load_data
import eda_module
from missing_values_handler import handle_missing_values
from CTGD import handle_categorical_data
from scaling_functions import display_scaling_options
from transformation_functions import display_transformation_options
from feature_engineering import extract_feature, transform_feature, select_feature, show_dataframe
from data_splitting import split_data
from model_building import *


# Set configuration and initialize state
set_page_config()
initial_state()

# Progress Bar
def progress_bar():
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.0002)
        my_bar.progress(percent_complete + 1)


# Logo 
col1, col2, col3 = st.columns([0.25,1,0.25])
col2.image("./assets/logo.png", use_column_width=True)
new_line(2)

# Description
st.markdown("""Welcome to Auto Analyst Feature! 🚀
Made for Health Data Analysis of Lokahi Care's Services
Uncover the Insights from your data! No need to wait for Data Analysts.""", unsafe_allow_html=True)
st.divider()


# Dataframe selection
st.markdown("<h2 align='center'> <b> Getting Started", unsafe_allow_html=True)
new_line(1)
st.write("The first step is to upload your data. You can upload your data either by : **Upload File**, or **Write URL**. In all ways the data should be a csv file or Excel and should not exceed 200 MB.")
new_line(1)



# Uploading Way
uploading_way = st.session_state.uploading_way
col1, col2, col3 = st.columns(3,gap='large')

# Upload
def upload_click(): st.session_state.uploading_way = "upload"
col1.markdown("<h5 align='center'> Upload File", unsafe_allow_html=True)
col1.button("Upload File", key="upload_file", use_container_width=True, on_click=upload_click)
        
# URL
def url_click(): st.session_state.uploading_way = "url"
col3.markdown("<h5 align='center'> Write URL", unsafe_allow_html=True)
col3.button("Write URL", key="write_url", use_container_width=True, on_click=url_click)



# No Data
if st.session_state.df is None:

    # Upload
    if uploading_way == "upload":
        uploaded_file = st.file_uploader("Upload the Dataset", type=["csv", "xlsx", "xls"])
        if uploaded_file:
            try:
                df = load_data(uploaded_file)
                st.session_state.df = df
            except Exception as e:
                st.error(f"Error loading the file: {e}")

    # URL
    elif uploading_way == "url":
        url = st.text_input("Enter URL")
        if url:
            df = load_data(url)
            st.session_state.df = df
    
    
# Dataframe
if st.session_state.df is not None:

    # Re-initialize the variables from the state
    df = st.session_state.df
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    X_val = st.session_state.X_val
    y_val = st.session_state.y_val
    trained_model = st.session_state.trained_model
    is_train = st.session_state.is_train
    is_test = st.session_state.is_test
    is_val = st.session_state.is_val
    model = st.session_state.model
    show_eval = st.session_state.show_eval
    y_pred_train = st.session_state.y_pred_train
    y_pred_test = st.session_state.y_pred_test
    y_pred_val = st.session_state.y_pred_val
    metrics_df = st.session_state.metrics_df

    st.divider()
    new_line()

    # Call to the EDA module function
    eda_module.show_eda(df)


    # Missing Values
    handle_missing_values(df)


    # Encoding
    handle_categorical_data(st, df)


    # Scaling
    new_line()
    st.markdown("### ⚖️ Scaling", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Scaling"):
        new_line()

        # Scaling Methods
        display_scaling_options(st, df)


    # Data Transformation
    display_transformation_options(st, df)


    # Call to the EDA module function
    eda_module.show_eda(df)
    
                     
    st.divider()          
    col1, col2, col3= st.columns(3, gap='small')        

    if col1.button("🎬 Show df", use_container_width=True):
        new_line()
        st.subheader(" 🎬 Show The Dataframe")
        st.write("The dataframe is the dataframe that is used on this application to build the Machine Learning model. You can see the dataframe below 👇")
        new_line()
        st.dataframe(df, use_container_width=True)

    st.session_state.df.to_csv("df.csv", index=False)
    df_file = open("df.csv", "rb")
    df_bytes = df_file.read()
    if col2.download_button("📌 Download df", df_bytes, "df.csv", key='save_df', use_container_width=True):
        st.success("Downloaded Successfully!")


    if col3.button("⛔ Reset", use_container_width=True):
        new_line()
        st.subheader("⛔ Reset")
        st.write("Click the button below to reset the app and start over again")
        new_line()
        st.session_state.reset_1 = True

    if st.session_state.reset_1:
        col1, col2, col3 = st.columns(3)
        if col2.button("⛔ Reset", use_container_width=True, key='reset'):
            st.session_state.df = None
            st.session_state.clear()
            st.experimental_rerun()

