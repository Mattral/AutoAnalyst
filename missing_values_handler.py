import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import SimpleImputer
from utils import new_line

def handle_missing_values(df):
    # Missing Values
    new_line()
    st.markdown("### ⚠️ Missing Values", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Missing Values"):

        # Further Analysis
        new_line()
        missing = st.checkbox("Further Analysis", value=False, key='missing')
        new_line()
        if missing:

            col1, col2 = st.columns(2, gap='medium')
            with col1:
                # Number of Null Values
                st.markdown("<h6 align='center'> Number of Null Values", unsafe_allow_html=True)
                st.dataframe(df.isnull().sum().sort_values(ascending=False), height=300, use_container_width=True)

            with col2:
                # Percentage of Null Values
                st.markdown("<h6 align='center'> Percentage of Null Values", unsafe_allow_html=True)
                null_percentage = pd.DataFrame(round(df.isnull().sum()/df.shape[0]*100, 2))
                null_percentage.columns = ['Percentage']
                null_percentage['Percentage'] = null_percentage['Percentage'].map('{:.2f} %'.format)
                null_percentage = null_percentage.sort_values(by='Percentage', ascending=False)
                st.dataframe(null_percentage, height=300, use_container_width=True)

            # Heatmap
            col1, col2, col3 = st.columns([0.1,1,0.1])
            with col2:
                new_line()
                st.markdown("<h6 align='center'> Plot for the Null Values ", unsafe_allow_html=True)
                null_values = df.isnull().sum()
                null_values = null_values[null_values > 0]
                null_values = null_values.sort_values(ascending=False)
                null_values = null_values.to_frame()
                null_values.columns = ['Count']
                null_values.index.names = ['Feature']
                null_values['Feature'] = null_values.index
                fig = px.bar(null_values, x='Feature', y='Count', color='Count', height=350)
                st.plotly_chart(fig, use_container_width=True)


        # INPUT
        col1, col2 = st.columns(2)
        with col1:
            missing_df_cols = df.columns[df.isnull().any()].tolist()
            if missing_df_cols:
                add_opt = ["All Numerical Features (Default Feature)", "All Categorical Feature (Default Feature)"]
            else:
                add_opt = []
            fill_feat = st.multiselect("Select Features",  missing_df_cols + add_opt ,  help="Select Features to fill missing values")

        with col2:
            strategy = st.selectbox("Select Missing Values Strategy", ["Select", "Drop Rows", "Drop Columns", "Fill with Mean", "Fill with Median", "Fill with Mode (Most Frequent)", "Fill with ffill, bfill"], help="Select Missing Values Strategy")


        if fill_feat and strategy != "Select":

            new_line()
            col1, col2, col3 = st.columns([1,0.5,1])
            if col2.button("Apply", use_container_width=True, key="missing_apply", help="Apply Missing Values Strategy"):

                progress_bar()
                
                # All Numerical Features
                if "All Numerical Features (Default Feature)" in fill_feat:
                    fill_feat.remove("All Numerical Features (Default Feature)")
                    fill_feat += df.select_dtypes(include=np.number).columns.tolist()

                # All Categorical Features
                if "All Categorical Feature (Default Feature)" in fill_feat:
                    fill_feat.remove("All Categorical Feature (Default Feature)")
                    fill_feat += df.select_dtypes(include=np.object).columns.tolist()

                
                # Drop Rows
                if strategy == "Drop Rows":
                    st.session_state.all_the_process += f"""
# Drop Rows
df[{fill_feat}] = df[{fill_feat}].dropna(axis=0)
\n """
                    df[fill_feat] = df[fill_feat].dropna(axis=0)
                    st.session_state['df'] = df
                    st.success(f"Missing values have been dropped from the DataFrame for the features **`{fill_feat}`**.")


                # Drop Columns
                elif strategy == "Drop Columns":
                    st.session_state.all_the_process += f"""
# Drop Columns
df[{fill_feat}] = df[{fill_feat}].dropna(axis=1)
\n """
                    df[fill_feat] = df[fill_feat].dropna(axis=1)
                    st.session_state['df'] = df
                    st.success(f"The Columns **`{fill_feat}`** have been dropped from the DataFrame.")


                # Fill with Mean
                elif strategy == "Fill with Mean":
                    st.session_state.all_the_process += f"""
# Fill with Mean
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='mean')
df[{fill_feat}] = num_imputer.fit_transform(df[{fill_feat}])
\n """
                    from sklearn.impute import SimpleImputer
                    num_imputer = SimpleImputer(strategy='mean')
                    df[fill_feat] = num_imputer.fit_transform(df[fill_feat])

                    null_cat = df[missing_df_cols].select_dtypes(include=np.object).columns.tolist()
                    if null_cat:
                        st.session_state.all_the_process += f"""
# Fill with Mode
from sklearn.impute import SimpleImputer
cat_imputer = SimpleImputer(strategy='most_frequent')
df[{null_cat}] = cat_imputer.fit_transform(df[{null_cat}])
\n """
                        cat_imputer = SimpleImputer(strategy='most_frequent')
                        df[null_cat] = cat_imputer.fit_transform(df[null_cat])

                    st.session_state['df'] = df
                    if df.select_dtypes(include=np.object).columns.tolist():
                        st.success(f"The Columns **`{fill_feat}`** has been filled with the mean. And the categorical columns **`{null_cat}`** has been filled with the mode.")
                    else:
                        st.success(f"The Columns **`{fill_feat}`** has been filled with the mean.")
                    

                # Fill with Median
                elif strategy == "Fill with Median":
                    st.session_state.all_the_process += f"""
# Fill with Median
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='median')
df[{fill_feat}] = pd.DataFrame(num_imputer.fit_transform(df[{fill_feat}]), columns=df[{fill_feat}].columns)
\n """
                    from sklearn.impute import SimpleImputer
                    num_imputer = SimpleImputer(strategy='median')
                    df[fill_feat] = pd.DataFrame(num_imputer.fit_transform(df[fill_feat]), columns=df[fill_feat].columns)

                    null_cat = df[missing_df_cols].select_dtypes(include=np.object).columns.tolist()
                    if null_cat:
                        st.session_state.all_the_process += f"""
# Fill with Mode
from sklearn.impute import SimpleImputer
cat_imputer = SimpleImputer(strategy='most_frequent')
df[{null_cat}] = cat_imputer.fit_transform(df[{null_cat}])
\n """
                        cat_imputer = SimpleImputer(strategy='most_frequent')
                        df[null_cat] = cat_imputer.fit_transform(df[null_cat])

                    st.session_state['df'] = df
                    if df.select_dtypes(include=np.object).columns.tolist():
                        st.success(f"The Columns **`{fill_feat}`** has been filled with the Median. And the categorical columns **`{null_cat}`** has been filled with the mode.")
                    else:
                        st.success(f"The Columns **`{fill_feat}`** has been filled with the Median.")


                # Fill with Mode
                elif strategy == "Fill with Mode (Most Frequent)":
                    st.session_state.all_the_process += f"""
# Fill with Mode
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
df[{fill_feat}] = imputer.fit_transform(df[{fill_feat}])
\n """
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='most_frequent')
                    df[fill_feat] = imputer.fit_transform(df[fill_feat])

                    st.session_state['df'] = df
                    st.success(f"The Columns **`{fill_feat}`** has been filled with the Mode.")


                # Fill with ffill, bfill
                elif strategy == "Fill with ffill, bfill":
                    st.session_state.all_the_process += f"""
# Fill with ffill, bfill
df[{fill_feat}] = df[{fill_feat}].fillna(method='ffill').fillna(method='bfill')
\n """
                    df = df.fillna(method='ffill').fillna(method='bfill')
                    st.session_state['df'] = df
                    st.success("The DataFrame has been filled with ffill, bfill.")
        
        # Show DataFrame Button
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="missing_show_df")
        if show_df:
            st.dataframe(df, use_container_width=True)
