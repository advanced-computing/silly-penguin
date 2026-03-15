"""Pandera validation schemas for all data sources."""

from __future__ import annotations

import pandera as pa
from pandera import Check, Column, DataFrameSchema

# ----------------------------
# Constants
# ----------------------------
VALID_DEMAND_TYPES = {"Demand", "Day-ahead demand forecast"}

TEMP_MIN_C = -50.0
TEMP_MAX_C = 60.0

DEMAND_MIN = 0.0
DEMAND_MAX = 500_000.0  # higher bound for large BAs like PJM/MISO

INTERCHANGE_MIN = -200_000.0
INTERCHANGE_MAX = 200_000.0

GENERATION_MIN = 0.0
GENERATION_MAX = 500_000.0

NG_PRICE_MIN = 0.0
NG_PRICE_MAX = 50.0  # $/MMBtu — extreme but possible

AVG_TEMP_TOL = 0.2


# ----------------------------
# 1) EIA Demand Schema
# ----------------------------
demand_schema = DataFrameSchema(
    {
        "period": Column(pa.DateTime, nullable=False),
        "type-name": Column(
            pa.String,
            nullable=False,
            checks=Check.isin(VALID_DEMAND_TYPES),
        ),
        "value": Column(
            pa.Float,
            nullable=False,
            checks=[Check.ge(DEMAND_MIN), Check.le(DEMAND_MAX)],
        ),
        "respondent": Column(pa.String, nullable=False),
    },
    strict=False,
)


# ----------------------------
# 2) EIA Interchange Schema
# ----------------------------
interchange_schema = DataFrameSchema(
    {
        "period": Column(pa.DateTime, nullable=False),
        "value": Column(
            pa.Float,
            nullable=False,
            checks=[Check.ge(INTERCHANGE_MIN), Check.le(INTERCHANGE_MAX)],
        ),
        "fromba": Column(pa.String, nullable=False),
        "toba": Column(pa.String, nullable=False),
    },
    strict=False,
)


# ----------------------------
# 3) EIA Fuel Type Schema
# ----------------------------
fuel_type_schema = DataFrameSchema(
    {
        "period": Column(pa.DateTime, nullable=False),
        "value": Column(
            pa.Float,
            nullable=False,
            checks=[Check.ge(GENERATION_MIN), Check.le(GENERATION_MAX)],
        ),
        "respondent": Column(pa.String, nullable=False),
        "fueltype": Column(pa.String, nullable=False),
    },
    strict=False,
)


# ----------------------------
# 4) Natural Gas Price Schema
# ----------------------------
ng_price_schema = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "ng_price": Column(
            pa.Float,
            nullable=False,
            checks=[Check.ge(NG_PRICE_MIN), Check.le(NG_PRICE_MAX)],
        ),
    },
    strict=False,
)


# ----------------------------
# 5) Weather Schema
# ----------------------------
weather_schema = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "max_temp": Column(pa.Float, nullable=False, checks=Check.between(TEMP_MIN_C, TEMP_MAX_C)),
        "min_temp": Column(pa.Float, nullable=False, checks=Check.between(TEMP_MIN_C, TEMP_MAX_C)),
        "avg_temp": Column(pa.Float, nullable=False, checks=Check.between(TEMP_MIN_C, TEMP_MAX_C)),
        "ba": Column(pa.String, nullable=False),
    },
    strict=False,
    checks=[
        Check(
            lambda df: (df["max_temp"] >= df["min_temp"]).all(),
            error="max_temp < min_temp found.",
        ),
    ],
)


# ----------------------------
# 6) Merged Daily Schema
# ----------------------------
merged_schema = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "avg_demand_mwh": Column(
            pa.Float,
            nullable=False,
            checks=[Check.ge(DEMAND_MIN), Check.le(DEMAND_MAX)],
        ),
        "avg_temp": Column(
            pa.Float,
            nullable=False,
            checks=Check.between(TEMP_MIN_C, TEMP_MAX_C),
        ),
    },
    strict=False,
)


# ----------------------------
# Convenience helpers
# ----------------------------
def validate_demand(df):
    """Validate raw EIA demand dataframe."""
    return demand_schema.validate(df)


def validate_interchange(df):
    """Validate raw EIA interchange dataframe."""
    return interchange_schema.validate(df)


def validate_fuel_type(df):
    """Validate raw EIA fuel type dataframe."""
    return fuel_type_schema.validate(df)


def validate_ng_price(df):
    """Validate natural gas price dataframe."""
    return ng_price_schema.validate(df)


def validate_weather(df):
    """Validate raw weather dataframe."""
    return weather_schema.validate(df)


def validate_merged(df):
    """Validate merged dataframe."""
    return merged_schema.validate(df)
