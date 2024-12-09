# data_splitting.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import new_line

def display_splitting_options(df):
    st.markdown("### 🪚 Data Splitting", unsafe_allow_html=True)
    
    target, sets = None, None
    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("Select Target Variable", df.columns.tolist(), key='target', help="Target Variable is the variable that you want to predict.")
        st.session_state['target_variable'] = target
    with col2:
        sets = st.selectbox("Select The Split Sets", ["Select", "Train and Test", "Train, Validation, and Test"], key='sets', help="Train Set is the data used to train the model. Validation Set is the data used to validate the model. Test Set is the data used to test the model.")
        st.session_state['split_sets'] = sets
    
    return target, sets

def train_test_split_ui(df, target, train_size, test_size):
    col1, col2, col3 = st.columns([1, 0.5, 1])
    with col2:
        if st.button("Split Data"):
            st.session_state.all_the_process += f"""
# Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('{target}', axis=1), df['{target}'], train_size={train_size}, random_state=42)
\n """
            X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], train_size=train_size, random_state=42)
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.success("Data Splitting Done!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Train Set")
                st.write("X Train Shape: ", X_train.shape)
                st.write("Y Train Shape: ", y_train.shape)
                train = pd.concat([X_train, y_train], axis=1)
                train_csv = train.to_csv(index=False).encode('utf-8')
                st.download_button("Download Train Set", train_csv, "train.csv", "text/csv", key='train2')
            with col2:
                st.write("Test Set")
                st.write("X Test Shape: ", X_test.shape)
                st.write("Y Test Shape: ", y_test.shape)
                test = pd.concat([X_test, y_test], axis=1)
                test_csv = test.to_csv(index=False).encode('utf-8')
                st.download_button("Download Test Set", test_csv, "test.csv", "text/csv", key='test2')

def train_val_test_split_ui(df, target, train_size, val_size, test_size):
    if float(train_size + val_size + test_size) != 1.0:
        st.error(f"The sum of Train, Validation, and Test sizes must be equal to 1.0, your sum is: **train** + **validation** + **test** = **{train_size}** + **{val_size}** + **{test_size}** = **{sum([train_size, val_size, test_size])}**")
    else:
        col1, col2, col3 = st.columns([1, 0.5, 1])
        with col2:
            if st.button("Split Data", use_container_width=True):
                st.session_state.all_the_process += f"""
# Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_rem, y_train, y_rem = train_test_split(df.drop('{target}', axis=1), df['{target}'], train_size={train_size}, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size= {val_size} / (1.0 - {train_size}),random_state=42)
\n """
                X_train, X_rem, y_train, y_rem = train_test_split(df.drop(target, axis=1), df[target], train_size=train_size, random_state=42)
                X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size=val_size / (1.0 - train_size), random_state=42)
                st.session_state['X_train'] = X_train
                st.session_state['X_val'] = X_val
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_val'] = y_val
                st.session_state['y_test'] = y_test
                st.success("Data Splitting Done!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("Train Set")
                    st.write("X Train Shape: ", X_train.shape)
                    st.write("Y Train Shape: ", y_train.shape)
                    train = pd.concat([X_train, y_train], axis=1)
                    train_csv = train.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Train Set", train_csv, "train.csv", "text/csv", key='train3')
                with col2:
                    st.write("Validation Set")
                    st.write("X Validation Shape: ", X_val.shape)
                    st.write("Y Validation Shape: ", y_val.shape)
                    val = pd.concat([X_val, y_val], axis=1)
                    val_csv = val.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Validation Set", val_csv, "validation.csv", key='val3')
                with col3:
                    st.write("Test Set")
                    st.write("X Test Shape: ", X_test.shape)
                    st.write("Y Test Shape: ", y_test.shape)
                    test = pd.concat([X_test, y_test], axis=1)
                    test_csv = test.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Test Set", test_csv, "test.csv", key='test3')

def split_data(df):
    target, sets = display_splitting_options(df)
    
    if sets and target:
        if sets == "Train and Test":
            col1, col2 = st.columns(2)
            with col1:
                train_size = st.number_input("Train Size", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key='train_size')
            with col2:
                test_size = 1.0 - train_size
            train_test_split_ui(df, target, train_size, test_size)

        elif sets == "Train, Validation, and Test":
            col1, col2, col3 = st.columns(3)
            with col1:
                train_size = st.number_inpu
