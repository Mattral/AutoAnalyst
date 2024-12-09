import time
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from utils import new_line

def progress_bar():
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.0002)
        my_bar.progress(percent_complete + 1)

def display_scaling_options(st, df):
        # Scaling Methods
        scaling_methods = st.checkbox("Explain Scaling Methods", value=False, key='scaling_methods')
        if scaling_methods:
            new_line()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h6 align='center'> Standard Scaling </h6>" ,unsafe_allow_html=True)
                st.latex(r'''z = \frac{x - \mu}{\sigma}''')
                new_line()
                # Values Ranges for the output of Standard Scaling in general
                st.latex(r'''z \in [-3,3]''')   

            with col2:
                st.markdown("<h6 align='center'> MinMax Scaling </h6>", unsafe_allow_html=True)
                st.latex(r'''z = \frac{x - min(x)}{max(x) - min(x)}''')
                new_line()
                # Values Ranges for the output of MinMax Scaling in general
                st.latex(r'''z \in [0,1]''')
                
            with col3:
                st.markdown("<h6 align='center'> Robust Scaling </h6>", unsafe_allow_html=True)
                st.latex(r'''z = \frac{x - Q_1}{Q_3 - Q_1}''')
                # Values Ranges for the output of Robust Scaling in general
                new_line()
                st.latex(r'''z \in [-2,2]''')

            # write z in the range for the output in latex
            st.latex(r''' **  Z = The\ Scaled\ Value  ** ''')

            new_line()


        # Ranges for the numeric features
        feat_range = st.checkbox("Further Analysis", value=False, key='feat_range')
        if feat_range:
            new_line()
            st.write("The Ranges for the numeric features:")
            col1, col2, col3 = st.columns([0.05,1, 0.05])
            with col2:
                 st.dataframe(df.describe().T, width=700)
            
            new_line()

        # INPUT
        new_line()
        new_line()
        col1, col2 = st.columns(2)
        with col1:
            scale_feat = st.multiselect("Select Features", df.select_dtypes(include=np.number).columns.tolist(), help="Select the features to be scaled.")

        with col2:
            scaling = st.selectbox("Select Scaling", ["Select", "Standard Scaling", "MinMax Scaling", "Robust Scaling"], help="Select the scaling method.")


        if scale_feat and scaling != "Select":       
                new_line()
                col1, col2, col3 = st.columns([1, 0.5, 1])
                
                if col2.button("Apply", key='scaling_apply',use_container_width=True ,help="Click to apply scaling."):

                    progress_bar()
    
                    # Standard Scaling
                    if scaling == "Standard Scaling":
                        st.session_state.all_the_process += f"""
# Standard Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[{scale_feat}] = pd.DataFrame(scaler.fit_transform(df[{scale_feat}]), columns=df[{scale_feat}].columns)
\n """
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        df[scale_feat] = pd.DataFrame(scaler.fit_transform(df[scale_feat]), columns=df[scale_feat].columns)
                        st.session_state['df'] = df
                        st.success(f"The Features **`{scale_feat}`** have been scaled using Standard Scaling.")
    
                    # MinMax Scaling
                    elif scaling == "MinMax Scaling":
                        st.session_state.all_the_process += f"""
# MinMax Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[{scale_feat}] = pd.DataFrame(scaler.fit_transform(df[{scale_feat}]), columns=df[{scale_feat}].columns)
\n """
                        from sklearn.preprocessing import MinMaxScaler
                        scaler = MinMaxScaler()
                        df[scale_feat] = pd.DataFrame(scaler.fit_transform(df[scale_feat]), columns=df[scale_feat].columns)
                        st.session_state['df'] = df
                        st.success(f"The Features **`{scale_feat}`** have been scaled using MinMax Scaling.")
    
                    # Robust Scaling
                    elif scaling == "Robust Scaling":
                        st.session_state.all_the_process += f"""
# Robust Scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df[{scale_feat}] = pd.DataFrame(scaler.fit_transform(df[{scale_feat}]), columns=df[{scale_feat}].columns)
\n """
                        from sklearn.preprocessing import RobustScaler
                        scaler = RobustScaler()
                        df[scale_feat] = pd.DataFrame(scaler.fit_transform(df[scale_feat]), columns=df[scale_feat].columns)
                        st.session_state['df'] = df
                        st.success(f"The Features **`{scale_feat}`** have been scaled using Robust Scaling.")

        # Show DataFrame Button
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="scaling_show_df", help="Click to show the DataFrame.")
        if show_df:
            st.dataframe(df, use_container_width=True)
