# This code estimates static and dynamic two way fixed effects regressions when treatment is simultaneous and when treatment is staggered.
# For dynamic regressions, it compares three methods: have never treated units get -1 event time, 0 event time, and 999 event time (and omit that 999 indicator). All methods omit the -1 indicator.

import pandas as pd
import numpy as np
import pyfixest as pf
import re

def run_dynamic_did(panel, control_event_time_value, drop_terms):
    df = panel.copy()

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

    # Construct formula
    rhs_terms = " + ".join(leads_lags.columns)
    formula = f"y ~ {rhs_terms} | unit + time"

    return pf.feols(formula, data=df, vcov={"CRV1": "unit_fe"})

def generate_did_table(panel_path):
    panel = pd.read_csv(panel_path)

    # Estimate static DID
    # CRV1 matches R's fixest standard error
    static_did = pf.feols(
        fml="y ~ treated|unit + time ", 
        data=panel, 
        vcov={"CRV1": "unit_fe"}
    )

    # Estimate dynamic DID
    dynamic_did_minus_1 = run_dynamic_did(
        panel, control_event_time_value=-1, 
        drop_terms=["T_minus_1"]
    )
    dynamic_did_0 = run_dynamic_did(
        panel, control_event_time_value=0, 
        drop_terms=["T_minus_1"]
    )
    dynamic_did_999 = run_dynamic_did(
        panel, control_event_time_value=999, 
        drop_terms=["T_minus_1", "T_999"]
    )

    return pf.etable([static_did, dynamic_did_minus_1, dynamic_did_0, dynamic_did_999])

# Estimate for simultaneous treatment
generate_did_table("output/panel_data_simultaneous.csv")

# Re-estimate for staggered treatment
generate_did_table("output/panel_data_staggered.csv")

# Why are they equivalent?
# A. Are the datasets equivalent?
def prep_dynamic_did_data(panel_path, control_event_time_value, drop_terms):
    panel = pd.read_csv(panel_path)
    df = panel.copy()

    # Assign event_time
    df["event_time"] = np.where(
        df["ever_treated"] == 1,
        df["time"] - df["treatment_time"],
        control_event_time_value
    )

    # Create dummies and rename
    leads_lags = pd.get_dummies(df["event_time"].astype(int), prefix="T")
    leads_lags.columns = [col.replace("-", "minus_") for col in leads_lags.columns]

    # Drop reference + optional control group dummy
    leads_lags = leads_lags.drop(columns=drop_terms, errors="ignore")

    # Add to data and estimate
    df = pd.concat([df, leads_lags], axis=1)
    return df

df_minus_1 = prep_dynamic_did_data(
    panel_path="output/panel_data_simultaneous.csv", 
    control_event_time_value=-1, 
    drop_terms=["T_minus_1"]
)
df_minus_1 = df_minus_1.filter(regex="^y$|^unit$|^time$|^T_")

df_0 = prep_dynamic_did_data(
    panel_path="output/panel_data_simultaneous.csv", 
    control_event_time_value=0, 
    drop_terms=["T_minus_1"]
)
df_0 = df_0.filter(regex="^y$|^unit$|^time$|^T_")

df_999 = prep_dynamic_did_data(
    panel_path="output/panel_data_simultaneous.csv", 
    control_event_time_value=999, 
    drop_terms=["T_minus_1", "T_999"]
)
df_999 = df_999.filter(regex="^y$|^unit$|^time$|^T_")

df_minus_1.equals(df_0)
df_minus_1.equals(df_999)
df_0.equals(df_999)

# The -1 and 999 datasets are equivalent, but not the 0 (which has a lot more 1s in the T_0 indicator), so it makes sense that the -1 and 999 lead to identical estimates. Next, we have to show why the 999 dataset and the 0 dataset lead to the same estimates, despite being different.

# B. Is it that never treateds only identify time fixed effects?
# Proposed test: partial out time fixed effects from y and from the leads and lags, then estimate residual y on residual leads and lags for both datasets.

def partial_out_fe(panel, var, fe):
    model = pf.feols(fml=f"{var} ~ 1|{fe}", data=panel)
    return panel[var] - model.predict()

def test_fe_residuals(panel, fe):
    df = panel.copy()

    # Partial out time effects from y
    df["y_residual"] = partial_out_fe(df, "y", fe)

    # Partial out time effects from each lead and lag
    leads_lags = [col for col in df.columns if col.startswith("T_")]
    for col in leads_lags:
        df[f"{col}_residual"] = partial_out_fe(df, col, fe)

    # Regress residual y on residual leads and lags
    rhs_terms = (" + ".join(df.filter(regex=r"^T_.*_residual$")
    .columns))

    residual_formula = f"y_residual ~ {rhs_terms}"

    res_on_res = pf.feols(
        fml=residual_formula,
        data=df
    )
    return res_on_res

res_on_res_time_0 = test_fe_residuals(df_0, "time")
res_on_res_time_999 = test_fe_residuals(df_999, "time")

pf.etable([res_on_res_time_0, res_on_res_time_999])

# Partialing out time variation leads to almost identical estimates for each lead and lag except for the T_0 residual, as expected.

# Next test: then, do some similar check for group fixed effects, there has to be something about never treateds having no remaining variation in T_0 or something

res_on_res_unit_0 = test_fe_residuals(df_0, "unit")
res_on_res_unit_999 = test_fe_residuals(df_999, "unit")

pf.etable([res_on_res_unit_0, res_on_res_unit_999])

# Side note:
# Using linearmodels doesn't match R's fixest + is clunkier
# from linearmodels.panel import PanelOLS
# panel = panel.set_index(['unit', 'time'])

# static_did_lm = PanelOLS.from_formula("y ~ treated + EntityEffects + TimeEffects", data=panel)
# static_did_lm = static_did_lm.fit(cov_type='clustered', cluster_entity=True)

# print(static_did_lm.summary)