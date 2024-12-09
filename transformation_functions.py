import streamlit as st
import numpy as np
from utils import new_line

def progress_bar():
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.0002)
        my_bar.progress(percent_complete + 1)

def display_transformation_options(st, df):
    # Data Transformation
    new_line()
    st.markdown("### 🧬 Data Transformation", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Data Transformation"):
        new_line()

        # Transformation Methods
        trans_methods = st.checkbox("Explain Transformation Methods", key="trans_methods", value=False)
        if trans_methods:
            new_line()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("<h6 align='center'> Log <br> Transformation</h6>", unsafe_allow_html=True)
                st.latex(r'''z = log(x)''')

            with col2:
                st.markdown("<h6 align='center'> Square Root Transformation </h6>", unsafe_allow_html=True)
                st.latex(r'''z = \sqrt{x}''')

            with col3:
                st.markdown("<h6 align='center'> Cube Root Transformation </h6>", unsafe_allow_html=True)
                st.latex(r'''z = \sqrt[3]{x}''')

            with col4:
                st.markdown("<h6 align='center'> Exponential Transformation </h6>", unsafe_allow_html=True)
                st.latex(r'''z = e^x''')

        # Input
        new_line()
        col1, col2 = st.columns(2)
        with col1:
            trans_feat = st.multiselect("Select Features", df.select_dtypes(include=np.number).columns.tolist(), help="Select the features you want to transform.", key="transformation features")

        with col2:
            trans = st.selectbox("Select Transformation", ["Select", "Log Transformation", "Square Root Transformation", "Cube Root Transformation", "Exponential Transformation"],
                                 help="Select the transformation you want to apply.", 
                                 key="transformation")

        # Apply Transformation
        if trans_feat and trans != "Select":
            new_line()
            col1, col2, col3 = st.columns([1, 0.5, 1])
            if col2.button("Apply", key='trans_apply',use_container_width=True ,help="Click to apply transformation."):

                progress_bar()

                # Depending on the transformation selected
                if trans == "Log Transformation":
                    df[trans_feat] = np.log1p(df[trans_feat])
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Log Transformation.")
                elif trans == "Square Root Transformation":
                    df[trans_feat] = np.sqrt(df[trans_feat])
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Square Root Transformation.")
                elif trans == "Cube Root Transformation":
                    df[trans_feat] = np.cbrt(df[trans_feat])
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Cube Root Transformation.")
                elif trans == "Exponential Transformation":
                    df[trans_feat] = np.exp(df[trans_feat])
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Exponential Transformation.")

        # Show DataFrame Button
        new_line()
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="trans_show_df", help="Click to show the DataFrame.")
        
        if show_df:
            st.dataframe(df, use_container_width=True)

