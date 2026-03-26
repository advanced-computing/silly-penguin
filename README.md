# ⚡ US Grid Monitor

**How Do Power Systems Respond to Uncertainty and Shocks?**

A Streamlit dashboard analyzing US electricity grid behavior across three dimensions: forecast uncertainty, system flexibility through interregional power trade, and fuel substitution in response to energy price shocks.

🔗 **Live App**: [silly-penguin.streamlit.app](https://silly-penguin-hv5ae2e4zodut2mc2oarrh.streamlit.app/)

---

## Research Question

Power systems face constant uncertainty — from unpredictable demand swings to volatile fuel prices. This project investigates how the US electricity grid responds, using hourly operational data from the Energy Information Administration (EIA).

We analyze **10 major balancing authorities** (CISO, ERCO, PJM, MISO, NYIS, ISNE, SWPP, SOCO, TVA, BPAT) across three dimensions:

| Dimension | Question | Data Source |
|-----------|----------|-------------|
| 🎯 **Forecast Uncertainty** | How accurate are day-ahead demand forecasts? When do the largest errors occur? | EIA Region Data (D, DF) |
| 🔄 **System Flexibility** | Do regions rely more on power imports during high demand? | EIA Interchange Data |
| ⛽ **Fuel Substitution** | How does the generation mix shift when natural gas prices change? | EIA Fuel Type Data + Henry Hub Prices |

## Dashboard Pages

1. **📊 Executive Overview** — KPI cards, MAPE ranking, interchange trends, generation mix snapshot
2. **🎯 Forecast Uncertainty** — Forecast vs actual time series, error heatmaps, cross-BA accuracy comparison, temperature sensitivity analysis
3. **🔄 System Flexibility** — Net interchange trends, hourly import/export patterns, demand-interchange correlation
4. **⛽ Fuel Substitution** — Generation mix stacked area charts, fuel share trends, natural gas price vs generation response, multi-fuel price elasticity
5. **🗺️ Geographic Dashboard** — US map visualization of MAPE, demand, interchange, and renewable share by region
6. **📄 Research & Methodology** — Data sources, methodology, and limitations

## Data Sources

- **EIA Demand & Forecast** — `electricity/rto/region-data` (hourly)
- **EIA Interchange** — `electricity/rto/interchange-data` (hourly)
- **EIA Generation by Fuel** — `electricity/rto/fuel-type-data` (hourly)
- **EIA Natural Gas Price** — `natural-gas/pri/fut` (daily, Henry Hub spot)
- **Open-Meteo Weather** — Archive API (daily temperature per BA)

## Project Structure

```
├── app.py                  # Streamlit dashboard (6 pages)
├── data_fetching.py        # EIA + weather API fetching
├── data_processing.py      # Data transformation & metrics
├── validation.py           # Pandera schemas for all data sources
├── load_to_bigquery.py     # One-time data loader to BigQuery
├── requirements.txt
├── pyproject.toml           # Ruff config
└── tests/
    └── test_data_processing.py
```

## Setup

### Prerequisites

- Python 3.11+
- A [Google Cloud](https://console.cloud.google.com/) project with BigQuery enabled
- An [EIA API key](https://www.eia.gov/opendata/register.php)

### Installation

```bash
git clone https://github.com/advanced-computing/silly-penguin.git
cd silly-penguin
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Load Data to BigQuery

```bash
# Authenticate
gcloud auth application-default login

# Set your EIA key in .env
echo 'EIA_API_KEY=your-key-here' > .env

# Run the loader (one-time)
python load_to_bigquery.py
```

### Run Locally

```bash
# Set up .streamlit/secrets.toml with your GCP service account key
streamlit run app.py
```

### Deploy

The app reads all data from BigQuery via a service account. Configure secrets in Streamlit Cloud under **Settings → Secrets**.

## Team

**Xingyi Wang & Wuhao Xia** — SIPA, Columbia University

Course: DSPC7160 Advanced Computing for Policy
