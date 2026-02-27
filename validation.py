# validation.py
from __future__ import annotations

import pandera as pa
from pandera import Check, Column, DataFrameSchema

# ----------------------------
# Constants (so rules are explicit)
# ----------------------------
VALID_TYPE_NAMES = {"Demand", "Day-ahead demand forecast"}

TEMP_MIN_C = -30.0
TEMP_MAX_C = 55.0

# demand should never be negative; upper bound is just a sanity check
DEMAND_MIN = 0.0
DEMAND_MAX = 200_000.0

# avg_temp should match (max+min)/2 within tolerance (API can have floats/rounding)
AVG_TEMP_TOL = 0.2


# ----------------------------
# 1) EIA hourly schema (raw)
# Columns of interest: period, type-name, value
# ----------------------------
eia_schema = DataFrameSchema(
    {
        # (1) period is datetime and not null
        "period": Column(pa.DateTime, nullable=False),
        # (2) type-name is string and must be one of allowed categories
        "type-name": Column(
            pa.String,
            nullable=False,
            checks=Check.isin(VALID_TYPE_NAMES),
        ),
        # (3) value is numeric, non-null, non-negative
        # (4) value has a reasonable upper bound (sanity check)
        "value": Column(
            pa.Float,
            nullable=False,
            checks=[
                Check.ge(DEMAND_MIN),
                Check.le(DEMAND_MAX),
            ],
        ),
    },
    # allow extra columns returned by API
    strict=False,
    # (5) no duplicate (period, type-name) pairs (helps pivot/pivot_table logic)
    checks=Check(
        lambda df: ~df.duplicated(subset=["period", "type-name"]).any(),
        error="Duplicate (period, type-name) rows found.",
    ),
)


# ----------------------------
# 2) Weather daily schema (raw)
# Columns of interest: date, max_temp, min_temp, avg_temp
# ----------------------------
weather_schema = DataFrameSchema(
    {
        # (6) date is datetime and not null
        "date": Column(pa.DateTime, nullable=False),
        # (7) max_temp within realistic bounds
        "max_temp": Column(pa.Float, nullable=False, checks=Check.between(TEMP_MIN_C, TEMP_MAX_C)),
        # (8) min_temp within realistic bounds
        "min_temp": Column(pa.Float, nullable=False, checks=Check.between(TEMP_MIN_C, TEMP_MAX_C)),
        # (9) avg_temp within realistic bounds
        "avg_temp": Column(pa.Float, nullable=False, checks=Check.between(TEMP_MIN_C, TEMP_MAX_C)),
    },
    strict=False,
    checks=[
        # (10) date unique (daily data should have 1 row per date)
        Check(lambda df: df["date"].is_unique, error="Weather 'date' is not unique."),
        # (11) max_temp must be >= min_temp
        Check(
            lambda df: (df["max_temp"] >= df["min_temp"]).all(), error="max_temp < min_temp found."
        ),
        # (12) avg_temp ~= (max+min)/2 within tolerance
        Check(
            lambda df: (
                (df["avg_temp"] - (df["max_temp"] + df["min_temp"]) / 2).abs() <= AVG_TEMP_TOL
            ).all(),
            error="avg_temp deviates from (max_temp+min_temp)/2 beyond tolerance.",
        ),
    ],
)


# ----------------------------
# 3) Merged daily schema (post-merge)
# Columns of interest: date, avg_demand_mwh, avg_temp
# ----------------------------
merged_schema = DataFrameSchema(
    {
        # (13) date datetime and not null
        "date": Column(pa.DateTime, nullable=False),
        # (14) avg_demand_mwh numeric, not null, non-negative and sane upper bound
        "avg_demand_mwh": Column(
            pa.Float,
            nullable=False,
            checks=[Check.ge(DEMAND_MIN), Check.le(DEMAND_MAX)],
        ),
        # (15) avg_temp numeric, not null, within bounds
        "avg_temp": Column(
            pa.Float,
            nullable=False,
            checks=Check.between(TEMP_MIN_C, TEMP_MAX_C),
        ),
    },
    strict=False,
    checks=[
        # (16) merged date unique (1 row per day)
        Check(lambda df: df["date"].is_unique, error="Merged 'date' is not unique."),
    ],
)


# ----------------------------
# Convenience helpers
# ----------------------------
def validate_eia(df):
    """Validate raw EIA dataframe."""
    return eia_schema.validate(df)


def validate_weather(df):
    """Validate raw weather dataframe."""
    return weather_schema.validate(df)


def validate_merged(df):
    """Validate merged dataframe."""
    return merged_schema.validate(df)
