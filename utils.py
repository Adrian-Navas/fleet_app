import streamlit as st

def make_popover(label, content):
    with st.popover(label):
        st.markdown(content)

def plot_metrics_table(metrics_list):
    """
    metrics_list: list of dicts [{'Model': 'LR', 'MAE': 10, ...}]
    """
    st.table(metrics_list)
