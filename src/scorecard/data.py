"""Data loading, target definition, exclusions and sample splits."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "loan_portfolio.csv"

# Fields known only after origination (model outputs, outcomes, losses). Never used as features.
LEAKAGE_COLUMNS: tuple[str, ...] = (
    "pd_annual",
    "lgd",
    "el",
    "unexpected_loss",
    "rwa",
    "survival_months",
    "recovery_rate",
    "loss_given_default",
    "defaulted",
    "default_date",
)

# Identifiers / dates that are not predictive features.
NON_FEATURE_COLUMNS: tuple[str, ...] = ("loan_id", "origination_date", "maturity_date")

NUMERIC_FEATURES: tuple[str, ...] = (
    "maturity_months",
    "credit_score",
    "ead",
    "coupon_rate",
    "leverage",
    "interest_coverage",
    "debt_to_equity",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("sector", "loan_type", "collateral", "initial_rating")
CANDIDATE_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "bad_12m"
PERFORMANCE_WINDOW_MONTHS = 12

DEVELOPMENT_YEARS = (2015, 2016, 2017, 2018)
EXCLUDED_YEARS = (2019, 2020)  # COVID-shock cohorts
OOT_YEARS = (2021, 2022, 2023)

RANDOM_STATE = 42
HOLDOUT_SHARE = 0.30


def load_raw(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["origination_date", "default_date"])


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """bad_12m = 1 if the loan defaulted within 12 calendar months of origination.

    Dates in the source are month-starts, so 12 calendar months is 365 days, or 366 when the
    window spans 29 February. Using a calendar-month offset keeps that leap-year case in.
    """
    out = df.copy()
    horizon = out["origination_date"] + pd.DateOffset(months=PERFORMANCE_WINDOW_MONTHS)
    out[TARGET] = (out["default_date"].notna() & (out["default_date"] <= horizon)).astype(int)
    out["cohort_year"] = out["origination_date"].dt.year
    return out


def make_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return train / holdout (random 70/30 within 2015-2018, stratified) and OOT (2021-2023)."""
    dev = df[df["cohort_year"].isin(DEVELOPMENT_YEARS)]
    oot = df[df["cohort_year"].isin(OOT_YEARS)]
    train, holdout = train_test_split(
        dev, test_size=HOLDOUT_SHARE, random_state=RANDOM_STATE, stratify=dev[TARGET]
    )
    return {"train": train.copy(), "holdout": holdout.copy(), "oot": oot.copy()}


def cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("cohort_year")[TARGET].agg(n_loans="size", n_bad_12m="sum", bad_rate_12m="mean")
    role = np.select(
        [g.index.isin(DEVELOPMENT_YEARS), g.index.isin(EXCLUDED_YEARS), g.index.isin(OOT_YEARS)],
        ["development", "excluded (COVID shock)", "out-of-time"],
        default="unused",
    )
    g.insert(0, "sample", role)
    return g.reset_index()
