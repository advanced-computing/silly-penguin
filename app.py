import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="EIA Grid Monitor", page_icon="⚡")
st.title("⚡ California (CISO) Grid Monitor")
st.subheader("Xingyi Wang, Wuhao Xia") 
st.markdown("Real-time analysis of **Actual Demand** vs. **Day-ahead Forecast**.")

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("EIA_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["EIA_API_KEY"]
    except FileNotFoundError:
        st.error("API Key not found. Please set it in .env or Streamlit Secrets.")
        st.stop()


@st.cache_data(ttl=3600)
def get_eia_data(api_key):
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": "CISO",
        "facets[type][]": ["D", "DF"],
        "start": "2025-01-01T00",
        "end": "2026-02-01T00",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()["response"]["data"]
        df = pd.DataFrame(data)

        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["period"] = pd.to_datetime(df["period"])
        return df
    else:
        st.error(f"API Error: {response.status_code}")
        return pd.DataFrame()


with st.spinner("Fetching data from EIA API..."):
    df = get_eia_data(api_key)

if not df.empty:
    df_pivot = df.pivot(index="period", columns="type-name", values="value")

    st.sidebar.header("Filter Options")
    days_to_show = st.sidebar.slider("Days to visualize", 1, 30, 7)

    df_display = df_pivot.head(days_to_show * 24)

    last_actual = df_display["Demand"].iloc[0]
    last_forecast = df_display["Day-ahead demand forecast"].iloc[0]
    delta = last_actual - last_forecast

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Actual Demand", f"{last_actual:,.0f} MWh")
    col2.metric("Latest Forecast", f"{last_forecast:,.0f} MWh")
    col3.metric("Forecast Error (Delta)", f"{delta:,.0f} MWh", delta_color="inverse")

    st.subheader(f"Demand vs. Forecast (Last {days_to_show} Days)")

    fig, ax = plt.subplots(figsize=(10, 5))
    df_display.plot(ax=ax, linewidth=2)
    ax.set_ylabel("Megawatthours (MWh)")
    ax.set_xlabel("Time")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(["Forecast", "Actual Demand"])

    st.pyplot(fig)

    with st.expander("See Raw Data"):
        st.dataframe(df_display)

else:
    st.warning("No data available. Please check API Key or connection.")
