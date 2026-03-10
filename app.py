import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from data_processing import calculate_grid_kpis, merge_daily_demand_weather

st.set_page_config(page_title="EIA Grid Monitor", page_icon="⚡", layout="wide")

load_dotenv()
api_key = os.getenv("EIA_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["EIA_API_KEY"]
    except (FileNotFoundError, KeyError):
        st.error("API Key not found. Please set it in .env (Local) or Streamlit Secrets (Cloud).")
        st.stop()


@st.cache_data(ttl=3600)
def get_eia_data(api_key):
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": "CISO",
        "facets[type][]": ["D", "DF"],  # D=Actual, DF=Forecast
        "start": "2025-01-01T00",
        "end": "2026-02-01T00",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
    }
    response = requests.get(url, params=params)

    param_used_below = 200
    if response.status_code == param_used_below:
        data = response.json()["response"]["data"]
        df = pd.DataFrame(data)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["period"] = pd.to_datetime(df["period"])
        return df
    else:
        st.error(f"EIA API Error: {response.status_code}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_weather_data():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 34.05,
        "longitude": -118.24,
        "start_date": "2025-01-01",
        "end_date": "2026-02-01",
        "daily": ["temperature_2m_max", "temperature_2m_min", "weathercode"],
        "timezone": "America/Los_Angeles",
    }
    response = requests.get(url, params=params)

    param_used_below = 200

    if response.status_code == param_used_below:
        data = response.json()
        daily = data["daily"]
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(daily["time"]),
                "max_temp": daily["temperature_2m_max"],
                "min_temp": daily["temperature_2m_min"],
            }
        )
        df["avg_temp"] = (df["max_temp"] + df["min_temp"]) / 2
        return df
    else:
        st.error("Weather API Error")
        return pd.DataFrame()


# ==========================================
# Sidebar Navigation
# ==========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", ["⚡ Real-time Grid Monitor", "🌡️ Weather Impact Analysis", "📄 Project Proposal"]
)

# ==========================================
# PAGE 1: Real-time Grid Monitor
# ==========================================
if page == "⚡ Real-time Grid Monitor":
    st.title("⚡ California (CISO) Grid Monitor")
    st.subheader("Xingyi Wang, Wuhao Xia")
    st.markdown("Real-time analysis of **Actual Demand** vs. **Day-ahead Forecast**.")

    with st.spinner("Fetching data from EIA API..."):
        df = get_eia_data(api_key)

    if not df.empty:
        df_pivot = df.pivot_table(index="period", columns="type-name", values="value")

        st.sidebar.header("Filter Options")
        days_to_show = st.sidebar.slider("Days to visualize", 1, 30, 7)
        df_display = df_pivot.head(days_to_show * 24)

        last_actual, last_forecast, delta = calculate_grid_kpis(df_display)

        if last_actual is not None:
            col1, col2, col3 = st.columns(3)
            col1.metric("Latest Actual Demand", f"{last_actual:,.0f} MWh")
            col2.metric("Latest Forecast", f"{last_forecast:,.0f} MWh")
            col3.metric("Forecast Error (Delta)", f"{delta:,.0f} MWh", delta_color="inverse")
        else:
            st.warning("Data incomplete for KPI calculation.")

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

# ==========================================
# PAGE 2: Weather Impact Analysis
# ==========================================
elif page == "🌡️ Weather Impact Analysis":
    st.title("🌡️ Weather vs. Demand Analysis")
    st.markdown(
        "This page combines **EIA Electricity Data** with **Open-Meteo Weather Data** to "
        "explore the correlation between temperature and energy consumption."
    )

    eia_df = get_eia_data(api_key)
    weather_df = get_weather_data()

    # [REFACTORED] Data Processing: Filter, Resample, and Merge
    merged_df = merge_daily_demand_weather(eia_df, weather_df)

    if not merged_df.empty:
        st.subheader("Temperature & Electricity Demand Trend")

        fig2, ax1 = plt.subplots(figsize=(10, 5))

        color = "tab:blue"
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Daily Avg Demand (MWh)", color=color)
        ax1.plot(
            merged_df["date"],
            merged_df["avg_demand_mwh"],
            color=color,
            linewidth=2,
            label="Demand",
        )
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("Avg Temperature (°C)", color=color)
        ax2.plot(
            merged_df["date"],
            merged_df["avg_temp"],
            color=color,
            linestyle="--",
            linewidth=2,
            label="Temperature",
        )
        ax2.tick_params(axis="y", labelcolor=color)

        plt.title("Correlation: Electricity Demand vs. Temperature (Los Angeles)")
        st.pyplot(fig2)

        st.subheader("Scatter Plot: Temperature Sensitivity")
        fig3, ax3 = plt.subplots()
        ax3.scatter(merged_df["avg_temp"], merged_df["avg_demand_mwh"], alpha=0.6, c="purple")
        ax3.set_xlabel("Temperature (°C)")
        ax3.set_ylabel("Electricity Demand (MWh)")
        ax3.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig3)

        st.caption("Data Sources: EIA API (Electricity) & Open-Meteo API (Weather)")

    else:
        st.error("Failed to load data for analysis.")

# ==========================================
# PAGE 3: Project Proposal
# ==========================================

elif page == "📄 Project Proposal":
    st.markdown("---")
    st.caption("Course Project | Advanced Computing for Policy")
    st.title("Project Proposal")
    st.subheader("Electricity Demand Forecasting and Grid Stress Analysis")

    st.header("1. Dataset")

    st.write("""
    **Dataset Name:** Balancing Authority Areas Hourly Operating Data

    **Source:** U.S. Energy Information Administration (EIA)

    This dataset provides hourly electricity system data across more than 50
    U.S. balancing authorities, including:

    - Actual electricity demand
    - Day-ahead demand forecasts
    - Net electricity generation
    - Power interchange between regions

    The data is updated hourly and is accessible through the EIA API.
    """)

    st.header("2. Research Questions")

    st.write("""
    Our project investigates electricity demand forecasting and grid stress.

    **Main Questions:**

    1. How does the forecast error between day-ahead demand forecasts and
       actual electricity demand change during extreme temperature periods?

    2. Do balancing authorities rely more heavily on interregional
       electricity interchange during periods of high demand or
       large prediction errors?

    3. Are there systematic differences in prediction accuracy
       across balancing authorities?
    """)

    st.header("3. Target Visualizations")

    st.write("""
    The project will produce several visualizations:

    - Time-series line chart comparing **forecasted vs actual demand**
    - Temperature vs electricity demand analysis
    - Forecast error trends over time
    - Scatter plots of temperature vs demand
    - Map visualization of balancing authorities
    showing prediction error and interchange reliance
    """)

    st.header("4. Known Unknowns")

    st.write("""
    Some uncertainties remain in the dataset:

    - Not all balancing authorities may report generation breakdown
      consistently across the full time period.

    - Electricity demand can be influenced by factors not captured in the data,
      such as local holidays, outages, or policy changes.

    - Weather data is approximated using representative cities
      rather than full regional weather systems.
    """)

    st.header("5. Anticipated Challenges")

    st.write("""
    Several technical challenges are expected:

    - Aligning timestamps across multiple regions and datasets
    - Managing large volumes of hourly electricity data
    - Integrating weather data with electricity demand data
    - Designing visualizations that clearly communicate grid stress
    """)

    st.header("6. Proposed Dashboard")

    st.write("""
    The final Streamlit dashboard will contain:

    1. **Real-time Grid Monitor**
       - Forecast vs actual demand
       - Demand trends

    2. **Weather Impact Analysis**
       - Temperature vs demand
       - Forecast error analysis

    3. **Map Dashboard (planned)**
       - Regional prediction error
       - Interchange dependence
    """)

    st.header("7. Proposal Updates After Initial Implementation")

    st.write("""
    After implementing the first version of the Streamlit dashboard,
    we refined the project focus.

    Originally the proposal included several broad research questions.
    After exploring the data, we narrowed the scope to focus on:

    1. Forecast error in electricity demand prediction
    2. The relationship between grid stress and interregional electricity interchange

    We also decided to add a **map-based visualization**
    to allow users to explore regional differences
    in prediction accuracy and grid reliance.
    """)
