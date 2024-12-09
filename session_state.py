import streamlit as st
import pandas as pd

def initial_state():
    state_keys = {
        'df': None, 'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None,
        'X_val': None, 'y_val': None, 'model': None, 'trained_model': False,
        'trained_model_bool': False, 'problem_type': None, 'metrics_df': pd.DataFrame(),
        'is_train': False, 'is_test': False, 'is_val': False, 'show_eval': False,
        'all_the_process': '', 'all_the_process_predictions': False,
        'y_pred_train': None, 'y_pred_test': None, 'y_pred_val': None,
        'uploading_way': None, 'lst_models': [], 'lst_models_predictions': [],
        'models_with_eval': {}, 'reset_1': False
    }
    for key, initial_value in state_keys.items():
        if key not in st.session_state:
            st.session_state[key] = initial_value
