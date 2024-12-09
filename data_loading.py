import streamlit as st
import pandas as pd

def load_data(upd_file):
    if upd_file.name.endswith('.csv'):
        return pd.read_csv(upd_file)
    elif upd_file.name.endswith('.xlsx') or upd_file.name.endswith('.xls'):
        return pd.read_excel(upd_file)
    else:
        raise ValueError("Unsupported file format. Only CSV and Excel files are supported.")
