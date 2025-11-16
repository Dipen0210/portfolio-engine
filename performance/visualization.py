# performance/visualization.py
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from utils.formatting import format_percentage


def plot_equity_curve(portfolio_state):
    df = portfolio_state.portfolio_value_history.copy()
    if df.empty:
        st.warning("No portfolio history yet.")
        return
    plt.figure(figsize=(8, 4))
    plt.plot(df["Date"], df["Value"], label="Portfolio Value", linewidth=2)
    plt.title("📈 Portfolio Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Value ($)")
    plt.legend()
    st.pyplot(plt)


def display_metrics_table(metrics_dict):
    df = pd.DataFrame(metrics_dict, index=["Metrics"]).T
    df.columns = ["Value"]
    df["Value"] = df["Value"].apply(format_percentage)
    st.table(df)
