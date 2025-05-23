import pandas as pd
import numpy as np
import pyfixest as pf
import re

from event_study_functions import *

# Interesting partial regression quirk: for dynamic regressions, people usually assign never treateds to event time -999 or some other bogus value and omit that lead/lag. You can actually assign never treated units to any event time value, even ones that include ever treated units, like -1 or 0. The regressions will be the same because those values have no variation within fixed effects. Example:

def generate_did_table(panel_path):
    df = pd.read_csv(panel_path)

    # Estimate static DID
    # CRV1 matches R's fixest standard error
    static_did = pf.feols(
        fml="y ~ treated|unit + time ", 
        data=df, 
        vcov={"CRV1": "unit_fe"}
    )

    # Estimate dynamic DID
    dynamic_did_minus_1 = run_dynamic_did(
        panel_path, control_event_time_value=-1, 
        drop_terms=["T_minus_1"]
    )
    dynamic_did_0 = run_dynamic_did(
        panel_path, control_event_time_value=0, 
        drop_terms=["T_minus_1"]
    )
    dynamic_did_999 = run_dynamic_did(
        panel_path, control_event_time_value=999, 
        drop_terms=["T_minus_1", "T_999"]
    )

    return pf.etable([static_did, dynamic_did_minus_1, dynamic_did_0, dynamic_did_999])

# Estimate for simultaneous treatment
generate_did_table("output/panel_data_simultaneous.csv")

# Re-estimate for staggered treatment
generate_did_table("output/panel_data_staggered.csv")

# Why are they equivalent?
# First, note that the datasets for -1 and 999 are equivalent. In both cases, never treateds are being lumped into a lead or lag that is omitted from the regression, so all their lead/lag values are blocks of zeros.
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

df_minus_1.equals(df_999)
df_0.equals(df_999)

# Then, for the ones where leads/lags are blocks of ones for never treated units, those will be collinear with group fixed effects. You can see this by partialing out time and group fixed effects separately from y and from the leads and lags, then estimate residual y on residual leads and lags for both datasets.

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

# Partialing out time variation leads to almost identical estimates for each lead and lag except for the T_0 residual, as expected. Partialing out unit fixed effects leads to identical regressions.

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