# Grid Intelligence Platform

**Open-source power market intelligence dashboard for monitoring US electricity system behavior across major balancing authorities.**

A Streamlit dashboard for investigating U.S. grid operations through one connected workflow: executive briefing, anomaly detection, arbitrage signals, transition scoring, and compliance reporting.

**Live App**: [silly-penguin.streamlit.app](https://silly-penguin-hv5ae2e4zodut2mc2oarrh.streamlit.app/)

---

## Research Question

Power systems operate under constant uncertainty — from fluctuating demand to changing generation portfolios and shifting interregional electricity flows. This project investigates how the US electricity grid behaves under these conditions using operational data from the U.S. Energy Information Administration (EIA), combined with weather data from Open-Meteo.

We analyze **10 major balancing authorities** (CISO, ERCO, PJM, MISO, NYIS, ISNE, SWPP, SOCO, TVA, BPAT) across five core dimensions:

| Dimension | Question | Data Source |
|-----------|----------|-------------|
| **Executive Briefing** | What is the current cross-module picture for one balancing authority? | EIA, LBNL, gridstatus |
| **Anomaly Detection** | Where are demand forecast errors or LMP signals unusually abnormal? | EIA Region Data, gridstatus |
| **Arbitrage Signals** | Which BA-to-BA interchange routes show persistent directional flow and low volatility? | EIA Interchange Data |
| **Transition Scoring** | Which balancing authorities appear most attractive for renewable transition and project deployment? | EIA, LBNL, NREL |
| **Compliance Reports** | How can BA-level operational metrics be summarized in a structured reporting format? | EIA Demand, Forecast, Interchange, Fuel Type Data |

## Dashboard Pages

1. **Executive Briefing** — the main investigation hub with KPI cards, interpretations, drill-down actions, and cross-module context
2. **Anomaly Detection** — BA alert panels, control charts, cross-BA comparison, LMP anomaly triage, and drill-down into selected ISO locations
3. **Arbitrage Signals** — top interchange routes, route interpretation, hourly heatmaps, route profiles, and cross-module actions
4. **Transition Scoring** — composite ranking, factor breakdown, queue pipeline analysis, NREL resource map, and BA action shortcuts
5. **Compliance Reports** — structured BA summaries with direct links back to anomaly, transition, interchange, and generation views
6. **About** — methodology, architecture, data provenance, limitations, and platform framing

## Workflow Features

- Shared investigation context across modules using Streamlit session state
- Cross-page navigation that preserves BA, route, ISO, LMP location, and focus area
- Drill-down buttons from summary metrics into deeper diagnostics
- Interpretation panels for anomaly status, transition score, demand, and route signals
- Compliance `§5 Cross-Module Signals` as a navigation hub instead of a dead-end summary

## Data Sources

- **EIA Demand & Forecast** — `electricity/rto/region-data` (hourly)
- **EIA Interchange** — `electricity/rto/interchange-data` (hourly)
- **EIA Generation by Fuel** — `electricity/rto/fuel-type-data` (hourly)
- **EIA Natural Gas Price** — `natural-gas/pri/fut` (daily, Henry Hub)
- **Open-Meteo Weather** — Archive API (daily temperature per BA)
- **gridstatus ISO LMP** — day-ahead hourly zonal and hub LMPs
- **LBNL Queued Up** — interconnection queue project pipeline
- **NREL Solar Resource API** — sampled solar resource quality points

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

### BigQuery Access Notes

The dashboard reads its analytics tables from BigQuery. Local runs require credentials with:

- `roles/bigquery.jobUser`
- `roles/bigquery.dataViewer`

If you authenticate with user ADC instead of a service account, also set the quota project:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project sipa-adv-c-silly-penguin
```

### Deploy

The app reads all data from BigQuery via a service account. Configure secrets in Streamlit Cloud under **Settings → Secrets**.

### Methodology Highlights
- **Anomaly Detection** uses rolling forecast error monitoring with BA-specific historical percentile thresholds
- **Arbitrage Signals** score interchange routes based on directional consistency and stability
- **Transition Scoring** combines demand growth, renewable headroom, import dependence, fossil transition opportunity, queue activity, and queue completion
- **Compliance Reports** generate structured operational summaries for selected balancing authorities
- **Executive Briefing** acts as the cross-module entry point into the rest of the dashboard

## Team

**Xingyi Wang & Wuhao Xia** — SIPA, Columbia University

Course: DSPC7160 Advanced Computing for Policy
