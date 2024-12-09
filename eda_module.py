import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from utils import new_line

def show_eda(df):
    # Assuming necessary imports are available like numpy, pandas, seaborn, matplotlib, etc.
    st.markdown("### 🕵️‍♂️ Exploratory Data Analysis", unsafe_allow_html=True)
    st.write("")  # To add a new line or space, if needed
    

    with st.expander("Show EDA"):
        new_line()

        # Head
        head = st.checkbox("Show First 5 Rows", value=False)    
        new_line()
        if head:
            st.dataframe(df.head(), use_container_width=True)

        # Tail
        tail = st.checkbox("Show Last 5 Rows", value=False)
        new_line()
        if tail:
            st.dataframe(df.tail(), use_container_width=True)

        # Shape
        shape = st.checkbox("Show Shape", value=False)
        new_line()
        if shape:
            st.write(f"This DataFrame has **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
            new_line()

        # Columns
        columns = st.checkbox("Show Columns", value=False)
        new_line()
        if columns:
            st.write(pd.DataFrame(df.columns, columns=['Columns']).T)
            new_line()

        if st.checkbox("Check Data Types", value=False):
            st.write(df.dtypes)
            new_line()

        new_line()  
        if st.checkbox("Show Skewness and Kurtosis", value=False):
            skew_kurt = pd.DataFrame(data={
                'Skewness': df.skew(),
                'Kurtosis': df.kurtosis()
            })
            st.write(skew_kurt)
            new_line()

        new_line()  
        # Describe Numerical
        describe = st.checkbox("Show Description **(Numerical Features)**", value=False)
        new_line()
        if describe:
            st.dataframe(df.describe(), use_container_width=True)
            new_line()

        if st.checkbox("Unique Value Count", value=False):
            unique_counts = pd.DataFrame(df.nunique()).rename(columns={0: 'Unique Count'})
            st.write(unique_counts)
            new_line()

        new_line()  
        # Describe Categorical
        describe_cat = st.checkbox("Show Description **(Categorical Features)**", value=False)
        new_line()
        if describe_cat:
            if df.select_dtypes(include=[object, 'string']).columns.tolist():
                st.dataframe(df.describe(include=['object']), use_container_width=True)
                new_line()
            else:
                st.info("There is no Categorical Features.")
                new_line()

        # Correlation Matrix using heatmap seabron
        corr = st.checkbox("Show Correlation", value=False)
        new_line()
        if corr:

            if df.corr().columns.tolist():
                fig, ax = plt.subplots()
                sns.heatmap(df.corr(), cmap='Blues', annot=True, ax=ax)
                st.pyplot(fig)
                new_line()
            else:
                st.info("There is no Numerical Features.")
            

        # Missing Values
        missing = st.checkbox("Show Missing Values", value=False)
        new_line()
        if missing:

            col1, col2 = st.columns([0.4,1])
            with col1:
                st.markdown("<h6 align='center'> Number of Null Values", unsafe_allow_html=True)
                st.dataframe(df.isnull().sum().sort_values(ascending=False),height=350, use_container_width=True)

            with col2:
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

            new_line()
                 

        # Delete Columns
        delete = st.checkbox("Delete Columns", value=False)
        new_line()
        if delete:
            col_to_delete = st.multiselect("Select Columns to Delete", df.columns)
            new_line()
            
            col1, col2, col3 = st.columns([1,0.7,1])
            if col2.button("Delete", use_container_width=True):
                st.session_state.all_the_process += f"""
# Delete Columns
df.drop(columns={col_to_delete}, inplace=True)
\n """
                progress_bar()
                df.drop(columns=col_to_delete, inplace=True)
                st.session_state.df = df
                st.success(f"The Columns **`{col_to_delete}`** are Deleted Successfully!")


        # Show DataFrame Button
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([1, 0.7, 1])
        if col2.button("Show DataFrame", use_container_width=True):
            st.dataframe(df, use_container_width=True)

        #start point

        # Histograms for Numerical Features
        hist = st.checkbox("Show Histograms", value=False)
        new_line()
        if hist:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            col_for_hist = st.selectbox("Select Column for Histogram", options=numeric_cols)
            num_bins = st.slider("Select Number of Bins", min_value=10, max_value=100, value=30)
            fig, ax = plt.subplots()
            df[col_for_hist].hist(bins=num_bins, ax=ax, color='skyblue')
            ax.set_title(f'Histogram of {col_for_hist}')
            st.pyplot(fig)
            new_line()
        
        # Box Plots for Numerical Features
        boxplot = st.checkbox("Show Box Plots", value=False)
        new_line()
        if boxplot:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            col_for_box = st.selectbox("Select Column for Box Plot", options=numeric_cols)
            fig, ax = plt.subplots()
            df.boxplot(column=[col_for_box], ax=ax)
            ax.set_title(f'Box Plot of {col_for_box}')
            st.pyplot(fig)
            new_line()
        
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Scatter Plots for Numerical Features
        scatter = st.checkbox("Show Scatter Plots", value=False)
        new_line()
        if scatter:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            x_col = st.selectbox("Select X-axis Column", options=numeric_cols, index=0)
            y_col = st.selectbox("Select Y-axis Column", options=numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            fig, ax = plt.subplots()
            df.plot(kind='scatter', x=x_col, y=y_col, ax=ax, color='red')
            ax.set_title(f'Scatter Plot between {x_col} and {y_col}')
            st.pyplot(fig)
            new_line()
        
        # Pair Plots for Numerical Features
        pairplot = st.checkbox("Show Pair Plots", value=False)
        new_line()
        if pairplot:
            sns.pairplot(df.select_dtypes(include=np.number))
            st.pyplot()
        
        # Count Plots for Categorical Data
        countplot = st.checkbox("Show Count Plots", value=False)
        new_line()
        if countplot:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            col_for_count = st.selectbox("Select Column for Count Plot", options=categorical_cols)
            fig, ax = plt.subplots()
            sns.countplot(x=df[col_for_count], data=df, ax=ax)
            ax.set_title(f'Count Plot of {col_for_count}')
            st.pyplot(fig)
            new_line()
        
        # Pie Charts for Categorical Data
        pie_chart = st.checkbox("Show Pie Charts", value=False)
        new_line()
        if pie_chart:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            col_for_pie = st.selectbox("Select Column for Pie Chart", options=categorical_cols)
            pie_data = df[col_for_pie].value_counts()
            fig, ax = plt.subplots()
            ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax.set_title(f'Pie Chart of {col_for_pie}')
            st.pyplot(fig)
            new_line()
        
        new_line()
        if st.checkbox("Identify Outliers", value=False):
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            col_for_outliers = st.selectbox("Select Column to Check Outliers", options=numeric_cols)
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col_for_outliers], ax=ax)
            ax.set_title(f'Outliers in {col_for_outliers}')
            st.pyplot(fig)
            new_line()

        new_line()
        if st.checkbox("Show Cross-tabulations", value=False):
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            x_col = st.selectbox("Select X-axis Column for Cross-tab", options=categorical_cols, index=0)
            y_col = st.selectbox("Select Y-axis Column for Cross-tab", options=categorical_cols, index=1 if len(categorical_cols) > 1 else 0)
            cross_tab = pd.crosstab(df[x_col], df[y_col])
            st.write(cross_tab)
            new_line()

        new_line()
        if st.checkbox("Segmented Analysis", value=False):
            segments = st.selectbox("Select Segment", options=df.columns)
            segment_values = df[segments].dropna().unique()
            selected_segment = st.selectbox("Choose Segment Value", options=segment_values)
            segmented_data = df[df[segments] == selected_segment]
            st.write(segmented_data)
            new_line()

        new_line()
        if st.checkbox("Temporal Analysis", value=False):
            date_col_options = df.select_dtypes(include=[np.datetime64]).columns.tolist()
            value_col_options = df.select_dtypes(include=np.number).columns.tolist()
            
            if not date_col_options:
                st.error("No datetime columns found in the DataFrame.")
            elif not value_col_options:
                st.error("No numeric columns found in the DataFrame.")
            else:
                date_col = st.selectbox("Select Date Column", options=date_col_options)
                value_col = st.selectbox("Select Value Column", options=value_col_options)
                
                fig, ax = plt.subplots()
                df.set_index(date_col)[value_col].plot(ax=ax)
                ax.set_title(f'Trend Over Time - {value_col}')
                st.pyplot(fig)

        new_line()
        if st.checkbox("Show Word Cloud", value=False):
            # Get the list of object-type columns for user to choose from
            text_col_options = df.select_dtypes(include=[object, 'string']).columns.tolist()
            
            if text_col_options:
                # Let the user select a text column
                text_col = st.selectbox("Select Text Column for Word Cloud", options=text_col_options)
                
                # Collect text data, dropping NA values and joining them into a single string
                text_data = ' '.join(df[text_col].dropna()).strip()
                
                if text_data:  # Check if there is any text data to use
                    try:
                        wordcloud = WordCloud(width=800, height=400).generate(text_data)
                        fig, ax = plt.subplots()
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                    except ValueError as e:
                        st.error("Failed to generate word cloud: " + str(e))
                else:
                    st.error("No words available to create a word cloud. Please check the selected text data.")
            else:
                st.error("No suitable text columns found for creating a word cloud.")


        new_line()    
        # Interactive Data Tables
        interactive_table = st.checkbox("Show Interactive Data Table", value=False)
        new_line()
        if interactive_table:
            st.dataframe(df)
            new_line()

