# ⚡ Grid Intelligence Platform

**Open-source power market intelligence dashboard for monitoring US electricity system behavior across major balancing authorities.**

A Streamlit dashboard analyzing US electricity grid operations across five practical dimensions: anomaly detection in forecast errors, interchange-based arbitrage signals, renewable investment opportunity scoring, automated compliance summaries, and real-time market overview.

🔗 **Live App**: [silly-penguin.streamlit.app](https://silly-penguin-hv5ae2e4zodut2mc2oarrh.streamlit.app/)

---

## Research Question

Power systems operate under constant uncertainty — from fluctuating demand to changing generation portfolios and shifting interregional electricity flows. This project investigates how the US electricity grid behaves under these conditions using operational data from the U.S. Energy Information Administration (EIA), combined with weather data from Open-Meteo.

We analyze **10 major balancing authorities** (CISO, ERCO, PJM, MISO, NYIS, ISNE, SWPP, SOCO, TVA, BPAT) across five dimensions:

| Dimension | Question | Data Source |
|-----------|----------|-------------|
| 🚨 **Anomaly Detection** | Where are demand forecast errors unusually large relative to historical patterns? | EIA Region Data (Demand, Forecast) |
| 💰 **Arbitrage Signals** | Which BA-to-BA interchange routes show persistent directional flow and low volatility? | EIA Interchange Data |
| 🌱 **Renewable Investment Scoring** | Which balancing authorities appear most attractive for future renewable investment? | EIA Demand, Interchange, Fuel Type Data |
| 📋 **Compliance Reports** | How can BA-level operational metrics be summarized in a structured reporting format? | EIA Demand, Forecast, Interchange, Fuel Type Data |
| 📊 **Market Overview** | What is the latest snapshot of demand, forecast accuracy, interchange, and generation mix? | EIA Region Data, Interchange Data, Fuel Type Data |

## Dashboard Pages

1. **📊 Market Overview** — KPI cards, latest demand and forecast, BA MAPE ranking, demand trends, generation and interchange context  
2. **🚨 Anomaly Detection** — alert cards, control charts, historical error thresholds, cross-BA distribution comparison  
3. **💰 Arbitrage Signals** — top interchange routes, consistency scores, hourly heatmaps, selected route profiles  
4. **🌱 Renewable Investment Scoring** — BA ranking table, composite score breakdown, radar chart, regional map  
5. **📋 Compliance Reports** — auto-generated BA summaries covering demand, forecast accuracy, interchange, and fuel mix  

## Data Sources

- **EIA Demand & Forecast** — `electricity/rto/region-data` (hourly)
- **EIA Interchange** — `electricity/rto/interchange-data` (hourly)
- **EIA Generation by Fuel** — `electricity/rto/fuel-type-data` (hourly)
- **EIA Natural Gas Price** — `natural-gas/pri/fut` (daily, Henry Hub)
- **Open-Meteo Weather** — Archive API (daily temperature per BA)

## Project Structure

```text
├── app.py                   # Streamlit dashboard with multi-page interface
├── data_fetching.py         # EIA and Open-Meteo API fetching functions
├── data_processing.py       # Data transformation, metrics, scoring, summaries
├── validation.py            # Pandera schemas for source validation
├── load_to_bigquery.py      # ETL pipeline to BigQuery
├── requirements.txt
├── pyproject.toml           # Ruff config
└── tests/
    └── test_data_processing.py

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

### Methodology Highlights
- **Anomaly Detection** uses rolling forecast error monitoring with BA-specific historical percentile thresholds
- **Arbitrage Signals** score interchange routes based on directional consistency and stability
- **Renewable Investment Scoring** combines demand growth, renewable headroom, import dependence, and fossil transition opportunity
- **Compliance Reports** generate structured operational summaries for selected balancing authorities
- **Market Overview** provides a real-time operational snapshot across key grid indicators

## Team

**Xingyi Wang & Wuhao Xia** — SIPA, Columbia University

Course: DSPC7160 Advanced Computing for Policy
