import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from utils import new_line

def handle_categorical_data(st, df):

    # Encoding
    new_line()
    st.markdown("### 🔠 Handling Categorical Data", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Encoding"):
        new_line()

        # Explain Encoding
        exp_enc = st.checkbox("Explain Encoding", value=False, key='exp_enc')
        if exp_enc:
            col1, col2 = st.columns([0.8, 1])
            with col1:
                st.markdown("<h6 align='center'>Ordinal Encoding</h6>", unsafe_allow_html=True)
                cola, colb = st.columns(2)
                with cola:
                    st.write("Before Encoding")
                    st.dataframe(pd.DataFrame(np.array(['a', 'b', 'c', 'b', 'a'])), width=120, height=200)
                with colb:
                    st.write("After Encoding")
                    st.dataframe(pd.DataFrame(np.array([0, 1, 2, 1, 0])), width=120, height=200)

            with col2:
                st.markdown("<h6 align='center'>One Hot Encoding</h6>", unsafe_allow_html=True)
                cola, colb = st.columns([0.7, 1])
                with cola:
                    st.write("Before Encoding")
                    st.dataframe(pd.DataFrame(np.array(['a', 'b', 'c', 'b', 'a'])), width=150, height=200)
                with colb:
                    st.write("After Encoding")
                    st.dataframe(pd.DataFrame(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])), width=200, height=200)

            col1, col2, col3 = st.columns([0.5, 1, 0.5])
            with col2:
                new_line()
                st.markdown("<h6 align='center'>Count Frequency Encoding</h6>", unsafe_allow_html=True)
                cola, colb = st.columns([0.8, 1])
                with cola:
                    st.write("Before Encoding")
                    st.dataframe(pd.DataFrame(np.array(['a', 'b', 'c', 'b', 'a'])), width=150, height=200)
                with colb:
                    st.write("After Encoding")
                    st.dataframe(pd.DataFrame(np.array([0.4, 0.4, 0.2, 0.4, 0.4])), width=200, height=200)

            new_line()

        # Show Categorical Features
        show_cat = st.checkbox("Show Categorical Features", value=False, key='show_cat')
        if show_cat:
            col1, col2 = st.columns(2)
            col1.dataframe(df.select_dtypes(include=[object]), height=250, use_container_width=True)
            if len(df.select_dtypes(include=[object]).columns.tolist()) > 1:
                tmp = df.select_dtypes(include=[object])
                tmp = tmp.apply(lambda x: x.unique())
                tmp = tmp.to_frame()
                tmp.columns = ['Unique Values']
                col2.dataframe(tmp, height=250, use_container_width=True)

        # Further Analysis
        further_analysis = st.checkbox("Further Analysis", value=False, key='further_analysis')
        if further_analysis:
            col1, col2 = st.columns([0.5, 1])
            with col1:
                new_line()
                st.markdown("<h6 align='left'> Number of Unique Values</h6>", unsafe_allow_html=True)
                unique_values = pd.DataFrame(df.select_dtypes(include=[object]).nunique())
                unique_values.columns = ['# Unique Values']
                unique_values = unique_values.sort_values(by='# Unique Values', ascending=False)
                st.dataframe(unique_values, width=200, height=300)

            with col2:
                new_line()
                st.markdown("<h6 align='center'> Plot for the Count of Unique Values </h6>", unsafe_allow_html=True)
                unique_values = pd.DataFrame(df.select_dtypes(include=[object]).nunique())
                unique_values.columns = ['# Unique Values']
                unique_values = unique_values.sort_values(by='# Unique Values', ascending=False)
                unique_values['Feature'] = unique_values.index
                fig = px.bar(unique_values, x='Feature', y='# Unique Values', color='# Unique Values', height=350)
                st.plotly_chart(fig, use_container_width=True)

        # Input for encoding
        col1, col2 = st.columns(2)
        with col1:
            enc_feat = st.multiselect("Select Features", df.select_dtypes(include=[object]).columns.tolist(), key='encoding_feat', help="Select the categorical features to encode.")
        with col2:
            encoding = st.selectbox("Select Encoding", ["Select", "Ordinal Encoding", "One Hot Encoding", "Count Frequency Encoding"], key='encoding', help="Select the encoding method.")

        if enc_feat and encoding != "Select":
            new_line()
            col1, col2, col3 = st.columns([1, 0.5, 1])
            if col2.button("Apply", key='encoding_apply', use_container_width=True, help="Click to apply encoding."):
                # Perform encoding based on the selected method
                apply_encoding(df, enc_feat, encoding, st)

        # Show DataFrame Button
        col1, col2, col3 = st.columns([0.15, 1, 0.15])
        col2.divider()
        col1, col2, col3 = st.columns([1, 0.7, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="cat_show_df", help="Click to show the DataFrame.")
        if show_df:
            st.dataframe(df, use_container_width=True)

def apply_encoding(df, features, method, st):
    if method == "Ordinal Encoding":
        from sklearn.preprocessing import OrdinalEncoder
        encoder = OrdinalEncoder()
        df[features] = encoder.fit_transform(df[features])
        st.success(f"The Categories of the features **`{features}`** have been encoded using Ordinal Encoding.")
        
    elif method == "One Hot Encoding":
        df = pd.get_dummies(df, columns=features)
        st.success(f"The Categories of the features **`{features}`** have been encoded using One Hot Encoding.")

    elif method == "Count Frequency Encoding":
        for feature in features:
            freq = df[feature].value_counts() / len(df)
            df[feature] = df[feature].map(freq)
        st.success(f"The Categories of the features **`{features}`** have been encoded using Count Frequency Encoding.")
