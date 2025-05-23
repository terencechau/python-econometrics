import pandas as pd
import numpy as np
import pyfixest as pf
import re

def prep_dynamic_did_data(panel_path, control_event_time_value, drop_terms):
    df = pd.read_csv(panel_path)

    # Assign event_time
    # The if-else statement bakes in the interaction
    df["event_time"] = np.where(
        df["ever_treated"] == 1,
        df["time"] - df["treatment_time"],
        control_event_time_value
    )

    # Create dummies and rename
    # Note: writing a formula like "y ~ i(event_time, ref = -1) | unit + time", skips this step, but that leads to messy coefficient names
    leads_lags = pd.get_dummies(df["event_time"].astype(int), prefix="T")
    leads_lags.columns = [col.replace("-", "minus_") for col in leads_lags.columns]

    # Drop reference + optional control group dummy
    leads_lags = leads_lags.drop(columns=drop_terms, errors="ignore")

    # Add to data
    df = pd.concat([df, leads_lags], axis=1)
    return df

def run_dynamic_did(panel_path, control_event_time_value, drop_terms):
    df = prep_dynamic_did_data(panel_path, control_event_time_value, drop_terms)

    # Construct formula
    rhs_terms = " + ".join(df.filter(regex="^T_").columns)
    formula = f"y ~ {rhs_terms} | unit + time"

    return pf.feols(formula, data=df, vcov={"CRV1": "unit_fe"})