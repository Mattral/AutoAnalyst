# feature_engineering.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time
from PIL import Image
from wordcloud import WordCloud
from utils import new_line
from config import set_page_config
from session_state import initial_state

def extract_feature(df):
    st.markdown("#### Feature Extraction", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:  
        feat1 = st.selectbox("First Feature/s", ["Select"] + df.select_dtypes(include=np.number).columns.tolist(), key="feat_ex1", help="Select the first feature/s you want to extract.")
    with col2:
        op = st.selectbox("Mathematical Operation", ["Select", "Addition +", "Subtraction -", "Multiplication *", "Division /"], key="feat_ex_op", help="Select the mathematical operation you want to apply.")
    with col3:
        feat2 = st.selectbox("Second Feature/s",["Select"] + df.select_dtypes(include=np.number).columns.tolist(), key="feat_ex2", help="Select the second feature/s you want to extract.")
    
    if feat1 and op != "Select" and feat2:
        col1, col2, col3 = st.columns(3)
        with col2:
            feat_name = st.text_input("Feature Name", key="feat_name", help="Enter the name of the new feature.")
        
        col1, col2, col3 = st.columns([1, 0.6, 1])
        if col2.button("Extract Feature"):
            if feat_name == "":
                feat_name = f"({feat1} {op} {feat2})"
            
            if op == "Addition +":
                df[feat_name] = df[feat1] + df[feat2]
            elif op == "Subtraction -":
                df[feat_name] = df[feat1] - df[feat2]
            elif op == "Multiplication *":
                df[feat_name] = df[feat1] * df[feat2]
            elif op == "Division /":
                df[feat_name] = df[feat1] / df[feat2]
            
            st.session_state['df'] = df
            st.success(f"Feature '**_{feat_name}_**' has been extracted using {op}.")
    
    return df

def transform_feature(df):
    st.markdown("#### Feature Transformation", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:    
        feat_trans = st.multiselect("Select Feature/s", df.select_dtypes(include=np.number).columns.tolist(), help="Select the Features you want to Apply transformation operation on it")
    with col2:
        op = st.selectbox("Select Operation", ["Select", "Addition +", "Subtraction -", "Multiplication *", "Division /"], key='feat_trans_op', help="Select the operation you want to apply on the feature")
    with col3:
        value = st.text_input("Enter Value", key='feat_trans_val', help="Enter the value you want to apply the operation on it")
    
    if op != "Select" and value != "":
        col1, col2, col3 = st.columns([1, 0.7, 1])
        if col2.button("Transform Feature"):
            if op == "Addition +":
                df[feat_trans] = df[feat_trans] + float(value)
            elif op == "Subtraction -":
                df[feat_trans] = df[feat_trans] - float(value)
            elif op == "Multiplication *":
                df[feat_trans] = df[feat_trans] * float(value)
            elif op == "Division /":
                df[feat_trans] = df[feat_trans] / float(value)
            
            st.session_state['df'] = df
            st.success(f"The Features **`{feat_trans}`** have been transformed using {op} with the value **`{value}`**.")
    
    return df

def select_feature(df):
    st.markdown("#### Feature Selection", unsafe_allow_html=True)
    
    feat_sel = st.multiselect("Select Feature/s", df.columns.tolist(), key='feat_sel', help="Select the Features you want to keep in the dataset")
    
    if feat_sel:
        col1, col2, col3 = st.columns([1, 0.7, 1])
        if col2.button("Select Features"):
            df = df[feat_sel]
            st.session_state['df'] = df
            st.success(f"The Features **`{feat_sel}`** have been selected.")
    
    return df

def show_dataframe(df):
    col1, col2, col3 = st.columns([0.15,1,0.15])
    col2.divider()
    col1, col2, col3 = st.columns([0.9, 0.6, 1])
    with col2:
        show_df = st.button("Show DataFrame", key="feat_eng_show_df", help="Click to show the DataFrame.")
    
    if show_df:
        st.dataframe(df, use_container_width=True)
